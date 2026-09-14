"""
Reading Policy (RP) for LatentMAS-RP — chunk-level latent gating.

Implements a ChunkKV-inspired gating mechanism over latent tokens only.
Each of the 3 upstream agents (Planner, Critic, Refiner) generates 16 latent
KV tokens, divided into 4 chunks of 4 tokens each → 12 gates total
(P1–P4, C1–C4, R1–R4).  Prompt and context tokens are always retained.

Key design choices:
  - Chunk-level granularity: each gate controls 4 consecutive latent positions.
  - GateNet input: [q ; s_j ; e_agent ; e_chunk] where q is the judger's
    mean-pooled prompt embedding, s_j is the mean of key vectors from the
    middle Transformer layer over the chunk, and e_agent/e_chunk are learnable
    identity embeddings that supply agent and intra-agent position context.
  - Hard Bernoulli sampling during training; hard threshold at evaluation.
  - Sparsity penalty: mean keep ratio — avoids 4× penalty scaling vs. sum.
  - Decoupled training: GateNet updated via REINFORCE + within-question
    group-relative advantages + entropy bonus; backbone LLM is frozen.
    Loss: L = −E[log π(g) · A] − β·H(π)
  - Lazy initialization: GateNet is built on its first forward call so that
    the actual input dimension is determined automatically.
  - HF path only: vLLM does not expose per-position KV cache and therefore
    cannot apply chunk-level latent-only gating.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GateNet
# ---------------------------------------------------------------------------

class GateNet(nn.Module):
    """
    Lazily-initialized 3-layer MLP that maps [q ; s_j ; e_agent ; e_chunk] → scalar logit.

    Architecture (built on first forward call):
        Linear(dim_in, dim_in // 2) → ReLU
        Linear(dim_in // 2, dim_in // 4) → ReLU
        Linear(dim_in // 4, 1)

    The final layer's bias is initialised to 0.0 so that sigmoid(logit) = 0.5
    at initialisation, i.e. gates start half-open (within the target 0.5–1.0 range).
    """

    def __init__(self) -> None:
        super().__init__()
        self._built: bool = False
        self.layers: Optional[nn.Sequential] = None

    # ------------------------------------------------------------------
    def _build(self, dim_in: int, device: torch.device, dtype: torch.dtype) -> None:
        hidden1 = max(dim_in // 2, 1)
        hidden2 = max(hidden1 // 2, 1)
        self.layers = nn.Sequential(
            nn.Linear(dim_in, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        ).to(device=device, dtype=dtype)
        # Bias init: gates start half-open (σ(0) = 0.5)
        nn.init.constant_(self.layers[-1].bias, 0.0)
        self._built = True

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, dim_q + dim_s]`` — concatenation of query and block summary.
        Returns:
            logit: ``[B, 1]``
        """
        if not self._built:
            self._build(x.shape[-1], x.device, x.dtype)
        elif self.layers[0].weight.device != x.device or self.layers[0].weight.dtype != x.dtype:
            self.layers = self.layers.to(device=x.device, dtype=x.dtype)
        return self.layers(x)


# ---------------------------------------------------------------------------
# ReadingPolicy
# ---------------------------------------------------------------------------

class ReadingPolicy(nn.Module):
    """
    Learnable chunk-level reading policy for LatentMAS-RP.

    One GateNet per (agent, chunk) pair.  With n_agents=3 and chunks_per_agent=4
    there are 12 gates total: P1–P4, C1–C4, R1–R4.  Each gate controls 4
    consecutive latent KV positions; prompt/context tokens are always preserved.

    Parameters
    ----------
    n_agents:
        Number of upstream agents (Planner, Critic, Refiner → 3 by default).
    chunks_per_agent:
        Number of latent chunks per agent (default 4, each chunk = 4 tokens).
    embed_dim:
        Dimension of the learnable agent and chunk position embeddings.
    entropy_coef:
        β for the entropy bonus term: L = L_policy − β·H(π). Default 0.01.
    temperature:
        Bernoulli gate temperature (unused in hard sampling; kept for compat).
    hard_threshold:
        Threshold δ used for hard gating at evaluation time.
    sparsity_lambda:
        Weight λ for the mean keep-ratio sparsity penalty.
    ema_momentum:
        EMA momentum α for the legacy EMA baseline (used by update()).
    lr:
        Adam learning rate for the GateNet parameters.
    """

    def __init__(
        self,
        n_agents: int = 3,
        chunks_per_agent: int = 4,
        embed_dim: int = 32,
        entropy_coef: float = 0.01,
        temperature: float = 1.0,
        hard_threshold: float = 0.5,
        sparsity_lambda: float = 0.05,
        ema_momentum: float = 0.99,
        lr: float = 3e-5,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.chunks_per_agent = chunks_per_agent
        self.n_gates = n_agents * chunks_per_agent
        self.entropy_coef = entropy_coef
        self.temperature = temperature
        self.hard_threshold = hard_threshold
        self.sparsity_lambda = sparsity_lambda
        self.ema_momentum = ema_momentum

        # One independent GateNet per (agent, chunk) pair
        self.gate_nets = nn.ModuleList([GateNet() for _ in range(self.n_gates)])

        # Learnable agent and chunk-position identity embeddings
        self.agent_embed = nn.Embedding(n_agents, embed_dim)
        self.chunk_embed = nn.Embedding(chunks_per_agent, embed_dim)

        # EMA baseline (scalar, non-trainable)
        self.register_buffer("baseline", torch.tensor(0.0))

        self._lr = lr
        self._optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    # Lazy optimizer (built after GateNets are lazily initialised)
    # ------------------------------------------------------------------

    def get_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            params = list(self.parameters())
            if not params:
                raise RuntimeError(
                    "GateNet has no parameters yet — call get_optimizer() only after "
                    "the first forward pass (which triggers lazy initialization)."
                )
            self._optimizer = torch.optim.Adam(params, lr=self._lr)
        return self._optimizer

    def build_gate_nets_from_state_dict(self, state_dict: dict) -> None:
        """Pre-build GateNets from a checkpoint state_dict (for checkpoint resume)."""
        for i, gate_net in enumerate(self.gate_nets):
            key = f"gate_nets.{i}.layers.0.weight"
            if key in state_dict and not gate_net._built:
                weight = state_dict[key]
                dim_in = weight.shape[1]
                device = weight.device if isinstance(weight, torch.Tensor) else torch.device("cpu")
                dtype = weight.dtype if isinstance(weight, torch.Tensor) else torch.float32
                gate_net._build(dim_in, device, dtype)

    @staticmethod
    def _to_legacy(past_kv):
        """Convert DynamicCache (transformers 5.x) to indexable list-of-tuples."""
        if past_kv is None:
            return None
        if hasattr(past_kv, "get_seq_length"):
            # DynamicCache is iterable; each element is (key, value, [metadata...])
            return list(past_kv)
        return past_kv

    # ------------------------------------------------------------------
    # Block-summary extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_block_summary_from_kv(
        past_kv: Tuple,
        start: int,
        end: int,
        mid_layer: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute a block summary by mean-pooling the key vectors of the middle
        Transformer layer over the agent's KV segment.

        Matches Eq. (3) in the design:
            s_i = mean_{t=start..end} K̂^{(i)}_{:,t,:}
        where K̂ is the key tensor reshaped to [B, L_i, H_kv * d_h].

        Args:
            past_kv: Tuple of ``(key, value)`` tensors per Transformer layer.
                     Each key has shape ``[B, H_kv, L_total, d_h]``.
            start: First KV position belonging to this agent's segment.
            end: One past the last position (exclusive).
            mid_layer: Which layer to use; defaults to ``len(past_kv) // 2``.

        Returns:
            summary: ``[B, H_kv * d_h]`` in float32.
        """
        past_kv = ReadingPolicy._to_legacy(past_kv)
        n_layers = len(past_kv)
        if mid_layer is None:
            mid_layer = n_layers // 2

        keys = past_kv[mid_layer][0]          # [B, H_kv, L_total, d_h]
        seg = keys[:, :, start:end, :]        # [B, H_kv, L_i,     d_h]
        B, H_kv, L_i, d_h = seg.shape
        # Reshape to [B, L_i, D] where D = H_kv * d_h
        seg = seg.permute(0, 2, 1, 3).reshape(B, L_i, H_kv * d_h)
        return seg.mean(dim=1).float()         # [B, H_kv * d_h]

    @staticmethod
    def extract_block_summary_from_embeddings(
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a block summary by mean-pooling an agent's embedding sequence.

        Used on the vLLM path where ``embedding_record[i]`` contains the
        concatenated input embeddings produced by the agent (including latent
        step vectors).

        Args:
            embeddings: ``[B, L_i, H]``

        Returns:
            summary: ``[B, H]`` in float32.
        """
        return embeddings.float().mean(dim=1)

    @staticmethod
    def extract_chunk_summaries(
        past_kv,
        latent_start: int,
        n_chunks: int = 4,
        chunk_size: int = 4,
        mid_layer: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Extract per-chunk key-vector summaries from the latent KV range only.

        For chunk j, mean-pools key vectors from the middle Transformer layer
        over positions [latent_start + j*chunk_size, latent_start + (j+1)*chunk_size).
        Positions before latent_start (prompt/context) are never accessed.

        Args:
            past_kv: Accumulated HF KV cache.
            latent_start: L_before[i] — first latent token position for agent i.
            n_chunks: Number of chunks (default 4).
            chunk_size: Tokens per chunk (default 4).
            mid_layer: Which layer to use; defaults to len(past_kv) // 2.

        Returns:
            List of n_chunks tensors, each ``[B, n_kv_heads * head_dim]`` (float32).
        """
        legacy_kv = ReadingPolicy._to_legacy(past_kv)
        n_layers = len(legacy_kv)
        if mid_layer is None:
            mid_layer = n_layers // 2

        keys = legacy_kv[mid_layer][0]  # [B, H_kv, L_total, d_h]
        summaries: List[torch.Tensor] = []
        for j in range(n_chunks):
            start = latent_start + j * chunk_size
            end = start + chunk_size
            seg = keys[:, :, start:end, :]          # [B, H_kv, chunk_size, d_h]
            B, H_kv, _, d_h = seg.shape
            seg = seg.permute(0, 2, 1, 3).reshape(B, chunk_size, H_kv * d_h)
            summaries.append(seg.mean(dim=1).float())  # [B, H_kv * d_h]
        return summaries

    @staticmethod
    def extract_chunk_summaries_from_embeddings(
        embeddings: torch.Tensor,  # [B, L_i, H]
        n_chunks: int = 4,
    ) -> List[torch.Tensor]:
        """
        Extract n_chunks summaries by splitting the embedding sequence evenly.
        Used on the vLLM path where KV cache is not directly accessible.

        Returns:
            List of n_chunks tensors, each ``[B, H]`` (float32).
        """
        L = embeddings.shape[1]
        base = max(1, L // n_chunks)
        summaries: List[torch.Tensor] = []
        for j in range(n_chunks):
            start = j * base
            end = start + base if j < n_chunks - 1 else L
            end = max(end, start + 1)
            seg = embeddings[:, start:end, :]
            summaries.append(seg.float().mean(dim=1))
        return summaries

    @staticmethod
    def extract_query_repr(
        judger_ids: torch.Tensor,
        embedding_layer: nn.Embedding,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the Judger's query representation by mean-pooling its input
        token embeddings (Eq. 4 in the design).

        Args:
            judger_ids: ``[B, L_judger]`` token IDs.
            embedding_layer: The model's input embedding layer.
            attention_mask: Optional ``[B, L_judger]`` mask (1 = real token).
                            When provided, padding tokens are excluded from
                            the mean.

        Returns:
            query: ``[B, H]`` in float32.
        """
        with torch.no_grad():
            emb = embedding_layer(judger_ids).float()   # [B, L, H]

        if attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1).to(emb.device)  # [B, L, 1]
            query = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            query = emb.mean(dim=1)

        return query  # [B, H]

    # ------------------------------------------------------------------
    # Gate computation — Eq. (5) training / Eq. (6) evaluation
    # ------------------------------------------------------------------

    def compute_gates(
        self,
        query: torch.Tensor,                       # [B, D_q]
        chunk_summaries_flat: List[torch.Tensor],  # n_gates tensors, each [B, D_s]
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gate values for all (agent, chunk) pairs.

        For gate idx = i * chunks_per_agent + j, the GateNet input is:
            [q ; s_{i,j} ; e_agent[i] ; e_chunk[j]]

        chunk_summaries_flat is ordered as:
            [agent0_chunk0, agent0_chunk1, ..., agent0_chunk3,
             agent1_chunk0, ..., agent2_chunk3]  (12 tensors total for 3×4)

        During training each gate is a hard Bernoulli(sigmoid(logit)) sample {0, 1};
        at evaluation a hard threshold on sigmoid(logit) is applied.

        Args:
            query: Judger query representation ``[B, D_q]``.
            chunk_summaries_flat: Flat list of n_gates chunk summary tensors,
                each ``[B, D_s]``.
            training: If True use hard Bernoulli sampling; else use hard threshold.

        Returns:
            gates:  ``[B, n_gates]`` — gate values in {0, 1} (float32).
            logits: ``[B, n_gates]`` — raw logits with grad_fn (for REINFORCE).
        """
        gates_list: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []

        q = query.float()
        B = q.shape[0]
        device = q.device

        # Migrate learnable embeddings to the query device on first GPU use
        if self.agent_embed.weight.device != device:
            self.agent_embed = self.agent_embed.to(device)
            self.chunk_embed = self.chunk_embed.to(device)

        for idx in range(self.n_gates):
            i = idx // self.chunks_per_agent   # agent index
            j = idx % self.chunks_per_agent    # chunk index within agent

            s = chunk_summaries_flat[idx].float().to(device)  # [B, D_s]

            e_a = self.agent_embed(torch.tensor([i], device=device)).expand(B, -1)  # [B, embed_dim]
            e_c = self.chunk_embed(torch.tensor([j], device=device)).expand(B, -1)  # [B, embed_dim]

            x = torch.cat([q, s, e_a, e_c], dim=-1)    # [B, D_q + D_s + 2*embed_dim]
            logit = self.gate_nets[idx](x).squeeze(-1)   # [B]
            logits_list.append(logit)

            if training:
                # Hard Bernoulli sample — straight-through via REINFORCE log-prob
                g = torch.distributions.Bernoulli(logits=logit).sample()      # [B] ∈ {0, 1}
            else:
                # Hard threshold at evaluation
                g = (torch.sigmoid(logit) > self.hard_threshold).float()      # [B]

            gates_list.append(g)

        gates = torch.stack(gates_list, dim=-1)    # [B, n_gates]
        logits = torch.stack(logits_list, dim=-1)  # [B, n_gates]
        return gates, logits

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    @staticmethod
    def apply_gates_hf(
        past_kv,
        latent_starts: List[int],
        gates: torch.Tensor,           # [B, n_gates]
        chunks_per_agent: int = 4,
        chunk_size: int = 4,
    ):
        """
        Scale only the latent KV positions for each (agent, chunk) pair.
        Prompt/context tokens at positions < latent_starts[i] are never touched.

        Gate idx = i*chunks_per_agent + j scales positions
        [latent_starts[i] + j*chunk_size, latent_starts[i] + (j+1)*chunk_size)
        across ALL Transformer layers.

        Args:
            past_kv: Accumulated HF KV cache.
            latent_starts: L_before[i] for each agent i (len = n_agents).
            gates: ``[B, n_gates]`` gate values.
            chunks_per_agent: Chunks per agent (default 4).
            chunk_size: Tokens per chunk (default 4).

        Returns:
            New past_kv with gated latent segments; prompt tokens unchanged.
        """
        n_agents = len(latent_starts)
        n_gates = n_agents * chunks_per_agent

        # DynamicCache path (transformers 5.x)
        if hasattr(past_kv, "get_seq_length"):
            gated_cache = copy.deepcopy(past_kv)
            for layer in gated_cache.layers:
                for idx in range(n_gates):
                    i = idx // chunks_per_agent
                    j = idx % chunks_per_agent
                    start = latent_starts[i] + j * chunk_size
                    end = start + chunk_size
                    g = gates[:, idx].view(-1, 1, 1, 1).to(layer.keys.dtype).to(layer.keys.device)
                    layer.keys[:, :, start:end, :].mul_(g)
                    layer.values[:, :, start:end, :].mul_(g)
            return gated_cache

        # Legacy tuple-of-tuples path (transformers 4.x)
        new_layers = []
        for layer_kv in past_kv:
            k = layer_kv[0].clone()   # [B, H_kv, L_total, d_h]
            v = layer_kv[1].clone()
            for idx in range(n_gates):
                i = idx // chunks_per_agent
                j = idx % chunks_per_agent
                start = latent_starts[i] + j * chunk_size
                end = start + chunk_size
                g = gates[:, idx].view(-1, 1, 1, 1).to(k.dtype).to(k.device)
                k[:, :, start:end, :] *= g
                v[:, :, start:end, :] *= g
            new_layers.append((k, v))
        return tuple(new_layers)

    @staticmethod
    def apply_gates_vllm(
        embedding_record: List[torch.Tensor],
        gates: torch.Tensor,   # [B, N]
    ) -> List[torch.Tensor]:
        """
        Scale each agent's embedding block by the corresponding gate value.

        Used on the vLLM path before concatenating the embeddings and feeding
        them to the vLLM engine.

        Args:
            embedding_record: Per-agent embedding tensors (each ``[B, L_i, H]``).
            gates: ``[B, N]`` gate values.

        Returns:
            List of gated tensors with the same shapes.
        """
        return [
            emb * gates[:, i].detach().view(-1, 1, 1).to(device=emb.device, dtype=emb.dtype)
            for i, emb in enumerate(embedding_record)
        ]

    # ------------------------------------------------------------------
    # Log-probabilities — Eq. (10)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_log_probs(
        gates: torch.Tensor,    # [B, N]  — treated as Bernoulli samples
        logits: torch.Tensor,   # [B, N]  — must have grad_fn for REINFORCE
    ) -> torch.Tensor:
        """
        Bernoulli log-probability of the sampled gates under the current policy:

            log π(g_i) = g_i · log σ(l_i) + (1 − g_i) · log(1 − σ(l_i))

        Sum over agents to get one scalar per batch element.

        Returns:
            log_probs: ``[B]`` with grad_fn w.r.t. GateNet parameters.
        """
        lls = (
            gates.detach() * F.logsigmoid(logits)
            + (1.0 - gates.detach()) * F.logsigmoid(-logits)
        )
        return lls.sum(dim=-1)   # [B]

    @staticmethod
    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        """
        Mean Bernoulli entropy over all (batch, gate) positions.

            H(Bernoulli(p)) = -p·log p - (1-p)·log(1-p)

        Args:
            logits: ``[B, n_gates]`` or ``[G, B, n_gates]`` — any shape.

        Returns:
            Scalar mean entropy (float32).
        """
        return torch.distributions.Bernoulli(logits=logits.float()).entropy().mean()

    # ------------------------------------------------------------------
    # Augmented reward — Eq. (8)
    # ------------------------------------------------------------------

    def compute_augmented_rewards(
        self,
        task_rewards: torch.Tensor,   # [B]
        gates: torch.Tensor,          # [B, N]
    ) -> torch.Tensor:
        """
        R = R_task − λ · mean_i |g_i|    (mean keep ratio, not sum)
        """
        sparsity_penalty = gates.detach().abs().mean(dim=-1)   # [B]  mean keep ratio
        return task_rewards - self.sparsity_lambda * sparsity_penalty

    # ------------------------------------------------------------------
    # REINFORCE update
    # ------------------------------------------------------------------

    def reinforce_step(
        self,
        log_probs: torch.Tensor,    # [B*G] — with grad_fn w.r.t. GateNet
        advantages: torch.Tensor,   # [B*G] — pre-computed, will be detached
        logits: torch.Tensor,       # any shape with n_gates in last dim — for entropy
    ) -> torch.Tensor:
        """
        Combined policy-gradient + entropy-bonus loss:

            L_policy = −(1/BG) Σ_{b,r} A_{b,r} · log π_θ(g_{b,r} | x_b)

            H(π)  = −(1/BGM) Σ_{b,r,j} [p log p + (1−p) log(1−p)]

            loss  = L_policy − entropy_coef · H(π)

        Args:
            log_probs:  Per-sample log π(g), shape ``[B*G]``.
            advantages: Pre-computed within-question group-relative advantages,
                        same shape as log_probs.  Detached inside this method.
            logits:     Raw GateNet logits used to compute H(π).  Shape can be
                        ``[B*G, n_gates]`` or ``[G, B, n_gates]`` — any shape
                        whose last dimension is n_gates.

        Returns:
            Scalar loss tensor with grad_fn.
        """
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy = self.compute_entropy(logits)
        return policy_loss - self.entropy_coef * entropy

    def update(
        self,
        log_probs_list: List[torch.Tensor],    # G × [B] — each with grad_fn
        task_rewards: torch.Tensor,             # [B, G]
        gates_list: List[torch.Tensor],         # G × [B, n_gates]
        logits_list: List[torch.Tensor],        # G × [B, n_gates] — each with grad_fn
    ) -> Dict:
        """
        Full REINFORCE update with within-question group-relative baseline and
        entropy bonus.

        For G rollouts on the same B questions:
            A_{b,r} = R_aug_{b,r} − (1/G) Σ_{r'} R_aug_{b,r'}

        Args:
            log_probs_list: One ``[B]`` log-prob tensor per rollout.
            task_rewards:   Task rewards, shape ``[B, G]``.
            gates_list:     One ``[B, n_gates]`` gate tensor per rollout.
            logits_list:    One ``[B, n_gates]`` logit tensor per rollout.

        Returns:
            Dictionary with training statistics.
        """
        G = len(log_probs_list)

        # Augmented rewards: R_aug = R_task − λ · mean|g|
        aug_per_k = [
            self.compute_augmented_rewards(task_rewards[:, k], gates_list[k])
            for k in range(G)
        ]
        aug_rewards = torch.stack(aug_per_k, dim=1)  # [B, G]

        # Within-question group-relative advantages: zero-sum across rollouts per question
        advantages = aug_rewards - aug_rewards.mean(dim=1, keepdim=True)  # [B, G]

        # Flatten to [B*G] — ordering: rollout 0 (all B), rollout 1 (all B), …
        flat_log_probs  = torch.cat(log_probs_list, dim=0)                         # [B*G]
        flat_advantages = torch.cat([advantages[:, k] for k in range(G)], dim=0)  # [B*G]
        all_logits      = torch.stack(logits_list)                                 # [G, B, n_gates]

        loss = self.reinforce_step(flat_log_probs, flat_advantages, all_logits)

        opt = self.get_optimizer()
        opt.zero_grad()
        loss.backward()
        opt.step()

        return {
            "rp_loss":            loss.item(),
            "mean_task_reward":   task_rewards.detach().mean().item(),
            "mean_aug_reward":    aug_rewards.detach().mean().item(),
            "mean_gate_value":    torch.stack(gates_list).detach().mean().item(),
            "entropy":            self.compute_entropy(all_logits.detach()).item(),
            "ema_baseline":       self.baseline.item(),
        }
