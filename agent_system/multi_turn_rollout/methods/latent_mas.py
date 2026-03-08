from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import default_agents
from ..models import ModelWrapper, _past_length
from ..prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from ..utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
from ..reading_policy import ReadingPolicy
import torch
import argparse
from vllm import SamplingParams
import pdb


@dataclass
class PreJudgerState:
    """
    Cached state produced by running the non-judger agents (Planner, Critic,
    Refiner) once.  The state is reused across G GRPO rollouts so that only
    the Judger needs to be re-run for each gate sample.

    HF path  : ``past_kv`` + ``agent_kv_ranges`` are populated.
    vLLM path: ``embedding_record`` + ``curr_prompt_emb`` + ``len_of_left``
               are populated instead.
    """
    items: List[Dict]
    judger_prompts: List[str]
    # Query and block summaries: pre-computed once, identical across rollouts
    query: torch.Tensor                         # [B, D_q]
    block_summaries: List[torch.Tensor]         # N × [B, D_s]

    # HF path
    judger_ids: Optional[torch.Tensor] = None   # [B, L_judger]
    judger_mask: Optional[torch.Tensor] = None  # [B, L_judger]
    past_kv: Optional[Tuple] = None
    agent_kv_ranges: Optional[List[Tuple[int, int]]] = None

    # vLLM path
    embedding_record: Optional[List[torch.Tensor]] = None
    curr_prompt_emb: Optional[torch.Tensor] = None   # [B, L_judger, H]
    len_of_left: Optional[List[int]] = None

    is_vllm: bool = False

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=args.max_new_tokens,
        )
        self.task = args.task

        # ---- Reading Policy (optional) ----
        # The backbone LLM is kept frozen; only GateNet parameters are updated.
        # Gates are suppressed when sequential_info_only/latent_only is active
        # (in those modes the accumulated KV cache is truncated per agent, so
        #  per-agent range tracking is not meaningful on the HF path).
        self.reading_policy: Optional[ReadingPolicy] = None
        if bool(getattr(args, "use_reading_policy", False)):
            n_upstream = sum(1 for a in self.agents if a.role != "judger")
            self.reading_policy = ReadingPolicy(
                n_agents=n_upstream,
                temperature=float(getattr(args, "rp_temperature", 1.0)),
                hard_threshold=float(getattr(args, "rp_hard_threshold", 0.5)),
                sparsity_lambda=float(getattr(args, "rp_sparsity_lambda", 0.01)),
                ema_momentum=float(getattr(args, "rp_ema_momentum", 0.99)),
                lr=float(getattr(args, "rp_lr", 1e-3)),
            )

        # Stored after each forward pass for the external REINFORCE training loop
        self._last_gates: Optional[torch.Tensor] = None
        self._last_logits: Optional[torch.Tensor] = None

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    # ------------------------------------------------------------------
    # GRPO helper: cache non-judger state, then run judger G times
    # ------------------------------------------------------------------

    def _build_prompt_messages(self, agent_role: str, items: List[Dict]) -> List[List[Dict]]:
        """Build prompt messages for a given agent role and items list."""
        if self.args.prompt == "sequential":
            return [
                build_agent_message_sequential_latent_mas(
                    role=agent_role, question=item["question"],
                    context="", method=self.method_name, args=self.args
                )
                for item in items
            ]
        else:
            return [
                build_agent_message_hierarchical_latent_mas(
                    role=agent_role, question=item["question"],
                    context="", method=self.method_name, args=self.args
                )
                for item in items
            ]

    @torch.no_grad()
    def collect_pre_judger_state(self, items: List[Dict]) -> PreJudgerState:
        """
        Run the non-judger agents (Planner, Critic, Refiner) once and cache
        their KV / embedding state.  Also pre-compute:

        * The Judger's query representation ``q`` (mean-pooled prompt embeddings).
        * Per-agent block summaries ``s_i`` (fixed for all G rollouts).

        Only the Judger needs to be called for each gate sample in the GRPO
        loop, making this ~G× faster than running the full pipeline G times.

        Requires ``self.reading_policy is not None``.

        Returns
        -------
        PreJudgerState
            Immutable snapshot of the non-judger state.  ``past_kv`` /
            ``agent_kv_ranges`` are set on the HF path; ``embedding_record`` /
            ``curr_prompt_emb`` / ``len_of_left`` are set on the vLLM path.
        """
        assert self.reading_policy is not None, \
            "collect_pre_judger_state requires use_reading_policy=True"

        batch_size = len(items)
        is_vllm = self.model.use_vllm

        past_kv: Optional[Tuple] = None
        agent_kv_ranges: List[Tuple[int, int]] = []
        embedding_record: List[torch.Tensor] = []

        judger_prompts: List[str] = []

        # ---- Run non-judger agents ----
        for agent in self.agents:
            batch_messages = self._build_prompt_messages(agent.role, items)
            prompts, _, _, _ = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                wrapped_prompts = [f"{p}<think>" for p in prompts] if self.args.think else prompts
                wrapped_enc = self.model.tokenizer(
                    wrapped_prompts, return_tensors="pt",
                    padding=True, add_special_tokens=False,
                )

                if is_vllm:
                    wrapped_ids  = wrapped_enc["input_ids"].to(self.model.HF_device)
                    wrapped_mask = wrapped_enc["attention_mask"].to(self.model.HF_device)
                    prev_len = _past_length(past_kv)
                    past_kv, emb = self.model.generate_latent_batch_hidden_state(
                        wrapped_ids, attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps, past_key_values=past_kv,
                    )
                    if self.sequential_info_only or self.latent_only:
                        new_len = _past_length(past_kv)
                        keep = self.latent_steps if self.latent_only else (new_len - prev_len)
                        past_kv = self._truncate_past(past_kv, keep)
                    if self.latent_only and self.latent_steps > 0:
                        emb = emb[:, -self.latent_steps:, :]
                    embedding_record.append(emb)
                    if self.sequential_info_only or self.latent_only:
                        embedding_record = embedding_record[-1:]
                else:
                    wrapped_ids  = wrapped_enc["input_ids"].to(self.model.device)
                    wrapped_mask = wrapped_enc["attention_mask"].to(self.model.device)
                    prev_len = _past_length(past_kv)
                    past_kv = self.model.generate_latent_batch(
                        wrapped_ids, attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps, past_key_values=past_kv,
                    )
                    after_len = _past_length(past_kv)
                    agent_kv_ranges.append((prev_len, after_len))
                    if self.sequential_info_only or self.latent_only:
                        new_len = _past_length(past_kv)
                        keep = self.latent_steps if self.latent_only else (new_len - prev_len)
                        past_kv = self._truncate_past(past_kv, keep)
            else:
                # Judger: only prepare the prompt, do not run yet
                judger_prompts = [f"{p}<think>" for p in prompts] if self.args.think else prompts

        # ---- Prepare judger token IDs / masks ----
        judger_enc = self.model.tokenizer(
            judger_prompts, return_tensors="pt",
            padding=True, add_special_tokens=False,
        )

        if is_vllm:
            judger_ids_for_rp  = judger_enc["input_ids"].to(self.model.HF_device)
            judger_mask_for_rp = judger_enc["attention_mask"].to(self.model.HF_device)
            judger_ids  = None
            judger_mask = None

            # Build curr_prompt_emb (fixed) and len_of_left
            curr_prompt_emb = self.model.embedding_layer(judger_ids_for_rp).to(self.vllm_device)
            assert "qwen" in self.args.model_name.lower(), \
                "vLLM judger insertion currently requires a Qwen model"
            len_of_left = []
            for p in judger_prompts:
                idx = p.find("<|im_start|>user\n")
                left = p[: idx + len("<|im_start|>user\n")]
                len_of_left.append(len(self.model.tokenizer(left)["input_ids"]))

            # Block summaries from embedding_record (vLLM path)
            block_summaries = [
                ReadingPolicy.extract_block_summary_from_embeddings(emb)
                for emb in embedding_record
            ]
            query = ReadingPolicy.extract_query_repr(
                judger_ids_for_rp, self.model.embedding_layer,
                attention_mask=judger_mask_for_rp,
            )
        else:
            judger_ids  = judger_enc["input_ids"].to(self.model.device)
            judger_mask = judger_enc["attention_mask"].to(self.model.device)
            judger_ids_for_rp  = judger_ids
            judger_mask_for_rp = judger_mask
            curr_prompt_emb = None
            len_of_left = None

            # Block summaries from KV-cache keys (HF path)
            block_summaries = [
                ReadingPolicy.extract_block_summary_from_kv(past_kv, start, end)
                for start, end in agent_kv_ranges
            ]
            query = ReadingPolicy.extract_query_repr(
                judger_ids_for_rp,
                self.model.model.get_input_embeddings(),
                attention_mask=judger_mask_for_rp,
            )

        return PreJudgerState(
            items=items,
            judger_prompts=judger_prompts,
            query=query,
            block_summaries=block_summaries,
            judger_ids=judger_ids,
            judger_mask=judger_mask,
            past_kv=past_kv,
            agent_kv_ranges=agent_kv_ranges if not is_vllm else None,
            embedding_record=embedding_record if is_vllm else None,
            curr_prompt_emb=curr_prompt_emb,
            len_of_left=len_of_left,
            is_vllm=is_vllm,
        )

    def run_judger_with_new_gates(
        self,
        state: PreJudgerState,
        training: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Sample a fresh set of gate values and run only the Judger.

        This is the inner loop of the GRPO rollout: the non-judger KV state
        is fixed (from ``collect_pre_judger_state``); only the gate sampling
        and Judger generation are repeated G times.

        Gate computation is wrapped in ``torch.enable_grad()`` so that the
        returned ``logits`` retain their ``grad_fn`` even when this method is
        called from inside a ``torch.no_grad()`` scope.

        Parameters
        ----------
        state:
            Pre-computed non-judger state.
        training:
            If ``True`` use Gumbel-Sigmoid; if ``False`` use hard threshold.

        Returns
        -------
        texts : List[str]
            Judger outputs for each item in the batch.
        gates : Tensor [B, N]
            Sampled gate values (detached).
        logits : Tensor [B, N]
            Raw GateNet logits (with ``grad_fn`` for REINFORCE/GRPO loss).
        """
        assert self.reading_policy is not None

        # Sample gates — enable_grad so logits keep their grad_fn for the
        # policy-gradient loss even if the caller is inside @torch.no_grad().
        with torch.enable_grad():
            gates, logits = self.reading_policy.compute_gates(
                state.query, state.block_summaries, training=training
            )

        if state.is_vllm:
            # Apply gates to embedding record
            gated_record = ReadingPolicy.apply_gates_vllm(
                state.embedding_record, gates.detach()
            )
            past_embedding = torch.cat(gated_record, dim=1).to(self.vllm_device)

            B, _, H = state.curr_prompt_emb.shape
            whole_prompt_emb_list = []
            for b in range(B):
                ins = state.len_of_left[b]
                combined = torch.cat([
                    state.curr_prompt_emb[b, :ins],
                    past_embedding[b],
                    state.curr_prompt_emb[b, ins:],
                ], dim=0)
                whole_prompt_emb_list.append(combined)
            max_len = max(e.shape[0] for e in whole_prompt_emb_list)
            whole_prompt_emb = torch.stack([
                torch.cat([e, torch.zeros(max_len - e.shape[0], H, device=e.device)], dim=0)
                for e in whole_prompt_emb_list
            ])
            inputs = [{"prompt_embeds": emb} for emb in whole_prompt_emb]
            with torch.no_grad():
                outputs = self.model.vllm_engine.generate(inputs, self.sampling_params)
            texts = [out.outputs[0].text.strip() for out in outputs]
        else:
            # Apply gates to HF KV cache (clones internally — cached past_kv unchanged)
            gated_kv = ReadingPolicy.apply_gates_hf(
                state.past_kv, state.agent_kv_ranges, gates.detach()
            )
            with torch.no_grad():
                texts, _ = self.model.generate_text_batch(
                    state.judger_ids,
                    state.judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=gated_kv,
                )

        return texts, gates.detach(), logits

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        # Track per-agent KV ranges for the Reading Policy (HF path).
        # Only valid when we are NOT truncating past_kv (sequential_info_only /
        # latent_only discard earlier agents' context, so per-agent gating is
        # not applicable on this path).
        rp_active_hf = (
            self.reading_policy is not None
            and not self.sequential_info_only
            and not self.latent_only
        )
        agent_kv_ranges: List[Tuple[int, int]] = []

        for agent in self.agents:

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]


            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )

                # Record this agent's KV segment range (before any truncation)
                if rp_active_hf:
                    after_past_len = _past_length(past_kv)
                    agent_kv_ranges.append((prev_past_len, after_past_len))

                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:

                past_for_decoding = past_kv if self.latent_steps > 0 else None

                if self.args.think:
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    judger_prompts = prompts

                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # ---- Reading Policy: gate each agent's KV segment (HF path) ----
                if rp_active_hf and past_for_decoding is not None and agent_kv_ranges:
                    embed_layer = self.model.model.get_input_embeddings()
                    # Compute Judger's query representation (mean of prompt embeddings)
                    query = ReadingPolicy.extract_query_repr(
                        judger_ids, embed_layer, attention_mask=judger_mask
                    )
                    # Compute block summaries from the accumulated KV cache
                    block_summaries = [
                        ReadingPolicy.extract_block_summary_from_kv(
                            past_for_decoding, start, end
                        )
                        for start, end in agent_kv_ranges
                    ]
                    # Gate computation (enable_grad so logits keep their grad_fn)
                    with torch.enable_grad():
                        gates, logits = self.reading_policy.compute_gates(
                            query, block_summaries,
                            training=self.reading_policy.training,
                        )
                    self._last_gates = gates.detach()
                    self._last_logits = logits   # retains grad_fn for REINFORCE
                    # Apply gates to the KV cache
                    past_for_decoding = ReadingPolicy.apply_gates_hf(
                        past_for_decoding, agent_kv_ranges, gates.detach()
                    )

                generated_batch, _ = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
                        }
                    )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
                # print(f'=========================================')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None
            
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results
    
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record: List[torch.Tensor] = []
        for agent in self.agents:
            
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
                
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                # to wrap all latent thoughts from previous agents
                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]
                
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:

                # ---- Reading Policy: gate each agent's embedding block (vLLM path) ----
                active_embedding_record = embedding_record
                if self.reading_policy is not None and embedding_record:
                    judger_encoded_for_rp = self.model.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    judger_ids_for_rp = judger_encoded_for_rp["input_ids"].to(self.model.HF_device)
                    judger_mask_for_rp = judger_encoded_for_rp["attention_mask"].to(self.model.HF_device)
                    # Block summaries: mean-pool each agent's embedding segment
                    block_summaries = [
                        ReadingPolicy.extract_block_summary_from_embeddings(emb)
                        for emb in embedding_record
                    ]
                    # Query representation: mean of judger's prompt token embeddings
                    query = ReadingPolicy.extract_query_repr(
                        judger_ids_for_rp,
                        self.model.embedding_layer,
                        attention_mask=judger_mask_for_rp,
                    )
                    # Gate computation (gradients enabled for REINFORCE)
                    gates, logits = self.reading_policy.compute_gates(
                        query, block_summaries,
                        training=self.reading_policy.training,
                    )
                    self._last_gates = gates.detach()
                    self._last_logits = logits   # retains grad_fn for REINFORCE
                    # Apply gates to the embedding blocks
                    active_embedding_record = ReadingPolicy.apply_gates_vllm(
                        embedding_record, gates.detach()
                    )

                # A stack of [B, L_i, H]
                past_embedding = torch.cat(active_embedding_record, dim=1).to(self.vllm_device)
                
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ) 
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                # Get current prompt embedding
                curr_prompt_emb = self.model.embedding_layer(judger_encoded).squeeze(0).to(self.vllm_device)
                
                # assert Qwen model
                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                # handle latent embedding insertion position    
                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\n")
                    # Get the text up to and including "<|im_start|>user\n"
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
                    
                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape  # assume shape consistency
                    
                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)

                # Pad back to max length if needed
                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack([
                    torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                    for x in whole_prompt_emb_list
                ])

                # else:
                    # Get full prompt embedding from cat with previous ones 
                    # B L H B L H
                    # whole_prompt_emb = torch.cat([past_embedding, curr_prompt_emb], dim=1)
                
                # pdb.set_trace()              
                
                # Use vLLM 
                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb 
                ]
                
                
                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]
                    
                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )


        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item["gold"]
            ok = (pred == gold) if (pred and gold) else False
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
