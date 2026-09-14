#!/usr/bin/env python3
"""
GRPO Training Script for the Reading Policy (GateNet).

This script trains the LatentMAS-RP Reading Policy using Group Relative Policy
Optimization (GRPO).  The backbone LLM (Planner / Critic / Refiner / Judger) is
kept **frozen** throughout; only the lightweight GateNet parameters are updated.

Algorithm
---------
For each batch of B questions:
  1. ``collect_pre_judger_state`` — run the three non-judger agents once,
     caching their KV/embedding state.  Pre-compute the fixed block summaries
     and judger query representation.
  2. Repeat G times (GRPO rollouts):
     - Sample a fresh gate vector via Gumbel-Sigmoid (stochastic).
     - Apply the gates to the cached state.
     - Run only the Judger to produce a text answer.
     - Evaluate correctness → task reward R ∈ {0, 1}.
     - Compute augmented reward  R_aug = R − λ · Σ|g_i|.
  3. GRPO advantage normalisation (group = G rollouts of the same question):
       A_{b,k} = (R_aug_{b,k} − mean_k) / (std_k + ε)
  4. Policy-gradient loss (single update step, no PPO clipping):
       L = −(1 / B·G) · Σ_{b,k} log π_φ(g_{b,k}) · A_{b,k}
  5. Gradient step on the GateNet optimizer.

Usage
-----
::

    python -m verl.trainer.ppo.train_reading_policy \\
        --model_name ./models/Qwen3-4B \\
        --train_file   data/GSM8K/train.parquet \\
        --val_file     data/GSM8K/test.parquet  \\
        --task gsm8k   \\
        --n_rollouts 8 \\
        --batch_size 16 \\
        --max_steps 500 \\
        --output_dir checkpoints/reading_policy

Checkpoints
-----------
The GateNet ``state_dict``, optimizer state, and training metadata are saved
to ``{output_dir}/checkpoint_step{N}.pt`` every ``--save_freq`` steps, and
to ``{output_dir}/checkpoint_latest.pt`` after every step.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.utils as nn_utils

# ---------------------------------------------------------------------------
# Lazy imports so the script can be syntax-checked without vLLM / transformers
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ===========================================================================
# Reward / answer checking helpers
# ===========================================================================

def _extract_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} answer from a model output."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None


def _normalize(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return str(float(s))
    except ValueError:
        return s.lower()


def _check_gsm8k(text: str, gold: str) -> float:
    """Return 1.0 if the prediction matches the GSM8K gold answer.

    Extraction priority for predictions: \\boxed{} first, then #### format.
    This mirrors the evaluation path and avoids rewarding non-attempts as 0
    just because the model used a different but valid format.
    """
    # Extract gold answer (typically "... #### 42" or bare number)
    m = re.search(r"####\s*(\S+)", gold)
    gold_val = _normalize(m.group(1) if m else gold.strip())

    # Extract prediction: prefer \boxed{}, then #### format
    pred_str = _extract_boxed(text)
    if pred_str is None:
        m2 = re.search(r"####\s*(\S+)", text)
        if m2:
            pred_str = m2.group(1)
    pred_val = _normalize(pred_str)

    return 1.0 if (pred_val is not None and pred_val == gold_val) else 0.0


def _check_mcq(text: str, gold: str) -> float:
    """Return 1.0 if the boxed prediction matches a single-letter MCQ answer."""
    pred = _extract_boxed(text)
    if pred is None:
        return 0.0
    return 1.0 if pred.strip().upper() == gold.strip().upper() else 0.0


REWARD_FN = {
    "gsm8k":        _check_gsm8k,
    "aime2024":     _check_gsm8k,
    "aime2025":     _check_gsm8k,
    "arc_easy":     _check_mcq,
    "arc_challenge": _check_mcq,
    "gpqa":         _check_mcq,
    "medqa":        _check_mcq,
}


def compute_task_reward(text: str, gold: str, task: str) -> float:
    fn = REWARD_FN.get(task, _check_gsm8k)
    return fn(text, gold)


# ===========================================================================
# Data loading
# ===========================================================================

def load_dataset(path: str, question_col: str, answer_col: str) -> List[Dict]:
    """Load a parquet or JSONL dataset and return a list of dicts."""
    p = Path(path)
    if p.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        cols = df.columns.tolist()
        q_col = question_col if question_col in cols else cols[0]
        a_col = answer_col   if answer_col   in cols else cols[1]
        records = []
        for _, row in df.iterrows():
            records.append({"question": str(row[q_col]), "gold": str(row[a_col])})
        return records
    elif p.suffix in (".jsonl", ".json"):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append({
                    "question": str(obj.get(question_col, obj.get("problem", ""))),
                    "gold":     str(obj.get(answer_col,   obj.get("answer",  ""))),
                })
        return records
    else:
        raise ValueError(f"Unsupported data format: {p.suffix}  (expected .parquet or .jsonl)")


# ===========================================================================
# GRPO core functions
# ===========================================================================

def compute_grpo_advantages(
    rewards: torch.Tensor,   # [B, G]  augmented rewards
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Group-relative advantage normalization.

    For each question b, normalize the G rollout rewards to zero mean and
    unit variance within the group:

        A_{b,k} = (R_{b,k} - mean_k R_{b,.}) / (std_k R_{b,.} + ε)

    Parameters
    ----------
    rewards : Tensor [B, G]
    epsilon : float
        Numerical stability term.

    Returns
    -------
    advantages : Tensor [B, G]
    """
    mean = rewards.mean(dim=1, keepdim=True)   # [B, 1]
    std  = rewards.std(dim=1, keepdim=True)    # [B, 1]
    return (rewards - mean) / (std + epsilon)


def grpo_loss(
    logits_list: List[torch.Tensor],   # G × [B, N]  — with grad_fn
    gates_list:  List[torch.Tensor],   # G × [B, N]  — detached
    advantages:  torch.Tensor,         # [B, G]       — detached
) -> torch.Tensor:
    """
    Single-step GRPO policy-gradient loss (no PPO clipping):

        L = −(1 / B·G) · Σ_{b,k} log π_φ(g_{b,k}) · A_{b,k}

    ``logits_list[k]`` must have ``grad_fn`` connected to the GateNet so that
    ``loss.backward()`` correctly updates only the GateNet parameters.
    """
    from agent_system.multi_turn_rollout.reading_policy import ReadingPolicy

    G   = len(logits_list)
    total = torch.tensor(0.0)

    for k in range(G):
        # log π(g_{b,k}) summed over agents  →  [B]
        lp = ReadingPolicy.compute_log_probs(gates_list[k], logits_list[k])
        # Make sure we stay on the same device as logits
        if total.device != lp.device:
            total = total.to(lp.device)
        adv_k = advantages[:, k].to(lp.device)        # [B]
        total = total - (lp * adv_k).sum()

    return total / (len(advantages) * G)


# ===========================================================================
# λ warmup schedule
# ===========================================================================

def compute_lambda(step: int, args) -> float:
    """Three-phase λ schedule.

    Phase 1  [0, phase1_end):    linear ramp from phase1_start → phase1_stop
    Phase 2  [phase1_end, phase2_end): fixed at phase2_val
    Phase 3  [phase2_end, ∞):    fixed at phase3_val

    Falls back to args.rp_sparsity_lambda when no warmup is configured
    (i.e. rp_lambda_phase1_end == 0).
    """
    if args.rp_lambda_phase1_end == 0:
        return args.rp_sparsity_lambda

    if step < args.rp_lambda_phase1_end:
        t = step / args.rp_lambda_phase1_end
        return args.rp_lambda_phase1_start + t * (args.rp_lambda_phase1_stop - args.rp_lambda_phase1_start)

    if args.rp_lambda_phase2_end > 0 and step < args.rp_lambda_phase2_end:
        return args.rp_lambda_phase2_val

    return args.rp_lambda_phase3_val


# ===========================================================================
# Evaluation
# ===========================================================================

@torch.no_grad()
def evaluate(
    method,
    dataset: List[Dict],
    args: argparse.Namespace,
    n_samples: int = 200,
) -> Dict:
    """
    Run the Reading Policy in *eval* mode (hard gates) on a random subset of
    ``dataset`` and report accuracy and mean gate sparsity.
    """
    from agent_system.multi_turn_rollout.reading_policy import ReadingPolicy

    method.reading_policy.eval()
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    n_correct = 0
    gate_sums = []

    for start in range(0, len(indices), args.batch_size):
        batch_idx = indices[start: start + args.batch_size]
        items = [dataset[i] for i in batch_idx]

        state = method.collect_pre_judger_state(items)
        texts, gates, _ = method.run_judger_with_new_gates(state, training=False)

        gate_sums.append(gates.mean().item())
        for i, (text, item) in enumerate(zip(texts, items)):
            if compute_task_reward(text, item["gold"], args.task) > 0.5:
                n_correct += 1

    method.reading_policy.train()
    return {
        "eval_accuracy": n_correct / len(indices),
        "eval_mean_gate": float(np.mean(gate_sums)) if gate_sums else 0.0,
    }


# ===========================================================================
# Main training loop
# ===========================================================================

def train(args: argparse.Namespace) -> None:
    """End-to-end GRPO training loop for the Reading Policy."""
    # -----------------------------------------------------------------------
    # 0. Logging / seeding
    # -----------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "train.log")),
        ],
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate chunk_size consistency (must be exact integer division)
    _chunk_size = args.latent_steps // args.rp_chunks_per_agent
    assert args.latent_steps == args.rp_chunks_per_agent * _chunk_size, (
        f"latent_steps ({args.latent_steps}) must equal "
        f"chunks_per_agent ({args.rp_chunks_per_agent}) × chunk_size ({_chunk_size}). "
        f"Choose a latent_steps that is divisible by chunks_per_agent."
    )
    logger.info(
        f"chunk_size={_chunk_size} "
        f"(latent_steps={args.latent_steps} / chunks_per_agent={args.rp_chunks_per_agent})"
    )

    # -----------------------------------------------------------------------
    # 1. Build LatentMAS args namespace expected by ModelWrapper / LatentMASMethod
    # -----------------------------------------------------------------------
    latent_args = argparse.Namespace(
        model_name            = args.model_name,
        device                = args.device,
        device2               = args.device2,
        latent_steps          = args.latent_steps,
        max_new_tokens        = args.judger_max_new_tokens,
        temperature           = args.temperature,
        top_p                 = args.top_p,
        prompt                = args.prompt_style,
        think                 = args.think,
        latent_space_realign       = args.latent_space_realign,
        realign_matrix_is_identity = args.realign_matrix_is_identity,
        use_vllm              = args.use_vllm,
        use_second_HF_model   = args.use_second_hf_model,
        tensor_parallel_size  = args.tensor_parallel_size,
        gpu_memory_utilization= args.gpu_memory_utilization,
        enable_prefix_caching = False,
        method                = "latent_mas",
        task                  = args.task,
        text_mas_context_length = -1,
        latent_only           = False,
        sequential_info_only  = False,
        # Reading Policy flags
        use_reading_policy    = True,
        rp_temperature        = args.rp_temperature,
        rp_hard_threshold     = args.rp_hard_threshold,
        rp_sparsity_lambda    = args.rp_sparsity_lambda,
        rp_ema_momentum       = args.rp_ema_momentum,
        rp_lr                 = args.rp_lr,
        rp_chunks_per_agent   = args.rp_chunks_per_agent,
        rp_entropy_coef       = args.rp_entropy_coef,
    )

    # -----------------------------------------------------------------------
    # 2. Initialize ModelWrapper and LatentMASMethod
    # -----------------------------------------------------------------------
    from agent_system.multi_turn_rollout.models import ModelWrapper
    from agent_system.multi_turn_rollout.methods.latent_mas import LatentMASMethod
    from agent_system.multi_turn_rollout.reading_policy import ReadingPolicy

    logger.info(f"Loading model: {args.model_name}")
    model_wrapper = ModelWrapper(
        model_name=args.model_name,
        device=torch.device(args.device),
        use_vllm=args.use_vllm,
        args=latent_args,
    )
    for p in model_wrapper.model.parameters():
        p.requires_grad_(False)

    method = LatentMASMethod(
        model=model_wrapper,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.judger_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        generate_bs=args.batch_size,
        args=latent_args,
    )
    assert method.reading_policy is not None, \
        "ReadingPolicy was not created — check use_reading_policy flag"

    reading_policy: ReadingPolicy = method.reading_policy
    reading_policy.train()

    # -----------------------------------------------------------------------
    # 3. Optionally load a checkpoint
    # -----------------------------------------------------------------------
    start_step = 0
    _resume_optimizer_state = None
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        reading_policy.build_gate_nets_from_state_dict(ckpt["reading_policy"])
        reading_policy.load_state_dict(ckpt["reading_policy"])
        _resume_optimizer_state = ckpt.get("optimizer", None)
        start_step = ckpt.get("step", 0) + 1
        logger.info(f"Resumed from {args.resume_from} at step {start_step}")

    optimizer = None  # lazily created after GateNet's first forward pass

    # -----------------------------------------------------------------------
    # 4. Data
    # -----------------------------------------------------------------------
    logger.info(f"Loading training data: {args.train_file}")
    train_data = load_dataset(args.train_file, args.question_col, args.answer_col)
    val_data   = load_dataset(args.val_file,   args.question_col, args.answer_col) \
                 if args.val_file else None
    logger.info(f"Train size: {len(train_data)},  Val size: {len(val_data) if val_data else 0}")

    # -----------------------------------------------------------------------
    # 5. Optional wandb
    # -----------------------------------------------------------------------
    use_wandb = args.wandb_project and not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"rp_{args.task}_{time.strftime('%m%d_%H%M')}",
                config=vars(args),
            )
        except ImportError:
            logger.warning("wandb not installed; disabling")
            use_wandb = False

    # -----------------------------------------------------------------------
    # 6. Training loop
    # -----------------------------------------------------------------------
    global_step = start_step
    running_loss         = 0.0
    running_reward       = 0.0
    running_gate         = 0.0
    running_entropy      = 0.0
    running_reward_std   = 0.0
    running_adv_abs      = 0.0
    log_interval = args.log_freq

    agent_names = [a.role for a in method.agents if a.role != "judger"]

    logger.info(
        f"Training: batch_size={args.batch_size}, n_rollouts(G)={args.n_rollouts}, "
        f"max_steps={args.max_steps}, baseline=within_question (group-relative), "
        f"entropy_coef={reading_policy.entropy_coef}, sparsity_lambda={args.rp_sparsity_lambda}"
    )

    # Early-stopping counters (per-step, not per log-interval)
    _es_high_keep   = 0   # consecutive steps with mean_keep_ratio > 0.95
    _es_low_keep    = 0   # consecutive steps with mean_keep_ratio < 0.05
    _es_degenerate  = 0   # consecutive steps with reward_std≈0 AND adv_abs≈0 AND |loss|≈0
    _ES_WINDOW      = 10
    _ES_DEGEN_THR   = 1e-3   # threshold for "approximately zero"
    _early_stop_reason: Optional[str] = None

    while global_step < args.max_steps:
        # Sample a batch of questions
        batch = random.sample(train_data, min(args.batch_size, len(train_data)))

        # ------------------------------------------------------------------
        # 6a. Cache non-judger state (single LLM pass for all 3 upstream agents)
        # ------------------------------------------------------------------
        state = method.collect_pre_judger_state(batch)

        # ------------------------------------------------------------------
        # 6b. GRPO rollouts  —  G samples per question
        # ------------------------------------------------------------------
        gates_list:  List[torch.Tensor] = []   # G × [B, N]
        logits_list: List[torch.Tensor] = []   # G × [B, N]
        rewards_raw = torch.zeros(len(batch), args.n_rollouts)  # task rewards
        sparsity    = torch.zeros(len(batch), args.n_rollouts)

        for k in range(args.n_rollouts):
            texts, gates, logits = method.run_judger_with_new_gates(state, training=True)

            # Lazily build optimizer on first rollout (GateNet is now initialized)
            if optimizer is None:
                optimizer = reading_policy.get_optimizer()
                if _resume_optimizer_state is not None:
                    try:
                        optimizer.load_state_dict(_resume_optimizer_state)
                    except Exception as e:
                        logger.warning(f"Could not restore optimizer state: {e}")
                    _resume_optimizer_state = None

            gates_list.append(gates)
            logits_list.append(logits)

            for b, (text, item) in enumerate(zip(texts, batch)):
                rewards_raw[b, k] = compute_task_reward(text, item["gold"], args.task)
                sparsity[b, k]    = gates[b].abs().mean().item()

        # ------------------------------------------------------------------
        # 6c. Augmented reward  R_aug = R_task − λ · mean(|g|)
        # ------------------------------------------------------------------
        current_lambda = compute_lambda(global_step, args)
        augmented_rewards = rewards_raw - current_lambda * sparsity   # [B, G]

        # ------------------------------------------------------------------
        # 6d. Policy-gradient loss with within-question group-relative baseline
        #     and entropy bonus
        #
        #   A_{b,r} = R_aug_{b,r} − (1/G) Σ_{r'} R_aug_{b,r'}
        #   L = −E[log π(g) · A] − β · H(π)
        # ------------------------------------------------------------------
        flat_log_probs = torch.cat([
            reading_policy.compute_log_probs(gates_list[k], logits_list[k])
            for k in range(args.n_rollouts)
        ], dim=0)                                                       # [B*G]

        # Within-question group-relative advantages (always used)
        advantages = (
            augmented_rewards - augmented_rewards.mean(dim=1, keepdim=True)
        )                                                               # [B, G]
        flat_advantages = torch.cat(
            [advantages[:, k] for k in range(args.n_rollouts)], dim=0
        ).to(flat_log_probs.device)                                     # [B*G]

        policy_loss = -(flat_log_probs * flat_advantages.detach()).mean()

        # Entropy bonus: H(Bernoulli(σ(logit))) over all gates, questions, rollouts
        all_logits_stacked = torch.stack(logits_list)                  # [G, B, n_gates]
        entropy = reading_policy.compute_entropy(all_logits_stacked)
        loss = policy_loss - reading_policy.entropy_coef * entropy

        # ------------------------------------------------------------------
        # 6e. Gradient step
        # ------------------------------------------------------------------
        assert optimizer is not None, "Optimizer should be initialized after first rollout"
        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            nn_utils.clip_grad_norm_(reading_policy.parameters(), args.max_grad_norm)
        optimizer.step()

        # ------------------------------------------------------------------
        # 6f. Monitoring metrics
        # ------------------------------------------------------------------
        all_gates_tensor = torch.stack(gates_list).float()             # [G, B, n_gates]
        per_gate_keep    = all_gates_tensor.mean(dim=[0, 1])           # [n_gates]
        n_ag  = reading_policy.n_agents
        cpa   = reading_policy.chunks_per_agent

        # Per-(agent, chunk) keep rates  e.g. keep_P1 … keep_R4
        per_gate_metrics: Dict[str, float] = {}
        for ag_i in range(n_ag):
            prefix = agent_names[ag_i][0].upper() if ag_i < len(agent_names) else f"A{ag_i}"
            for ch_j in range(cpa):
                gate_idx = ag_i * cpa + ch_j
                per_gate_metrics[f"keep_{prefix}{ch_j + 1}"] = per_gate_keep[gate_idx].item()

        # Active chunks per sample (mean number of open gates)
        active_per_sample = all_gates_tensor.sum(dim=-1).mean().item()

        # Gate-pattern diversity: fraction of unique {0,1}^n_gates patterns in B×G
        flat_binary = all_gates_tensor.reshape(-1, reading_policy.n_gates).int()
        gate_patterns = [tuple(row.tolist()) for row in flat_binary]
        gate_diversity = len(set(gate_patterns)) / max(len(gate_patterns), 1)

        reward_std         = rewards_raw.float().std().item()
        advantage_abs_mean = advantages.float().abs().mean().item()

        mean_task_reward = rewards_raw.mean().item()
        mean_gate_value  = all_gates_tensor.mean().item()

        running_loss         += loss.item()
        running_reward       += mean_task_reward
        running_gate         += mean_gate_value
        running_entropy      += entropy.item()
        running_reward_std   += reward_std
        running_adv_abs      += advantage_abs_mean

        if (global_step + 1) % log_interval == 0:
            avg_loss    = running_loss    / log_interval
            avg_reward  = running_reward  / log_interval
            avg_gate    = running_gate    / log_interval
            avg_entropy = running_entropy / log_interval

            metrics: Dict[str, float] = {
                "step":               global_step + 1,
                "loss":               avg_loss,
                "loss_policy":        policy_loss.item(),
                "mean_task_reward":   avg_reward,
                "mean_gate_value":    avg_gate,
                "active_chunks":      active_per_sample,
                "gate_diversity":     gate_diversity,
                "reward_std":         running_reward_std / log_interval,
                "advantage_abs_mean": running_adv_abs   / log_interval,
                "entropy":            avg_entropy,
                "rp_ema_baseline":    reading_policy.baseline.item(),
                "lambda":             current_lambda,
                **per_gate_metrics,
            }
            keep_str = "  ".join(
                f"{agent_names[ag_i][0].upper()}:["
                + " ".join(f"{per_gate_keep[ag_i * cpa + j].item():.2f}" for j in range(cpa))
                + "]"
                for ag_i in range(n_ag)
            )
            logger.info(
                f"[step {global_step+1:>5}]  loss={avg_loss:.4f}  "
                f"reward={avg_reward:.3f}  gate={avg_gate:.3f}  "
                f"active={active_per_sample:.1f}  H={avg_entropy:.3f}  λ={current_lambda:.4f}"
            )
            logger.info(
                f"          keep: {keep_str}  "
                f"diversity={gate_diversity:.2f}  "
                f"reward_std={running_reward_std/log_interval:.3f}  "
                f"adv_abs={running_adv_abs/log_interval:.3f}"
            )
            if use_wandb:
                wandb.log(metrics, step=global_step + 1)

            running_loss = running_reward = running_gate = 0.0
            running_entropy = running_reward_std = running_adv_abs = 0.0

        # ------------------------------------------------------------------
        # 6g. Early stopping (checked every step, independent of log_interval)
        # ------------------------------------------------------------------
        _keep = mean_gate_value   # gates ∈ {0,1} → mean = keep ratio
        if _keep > 0.95:
            _es_high_keep += 1;  _es_low_keep = 0
        elif _keep < 0.05:
            _es_low_keep  += 1;  _es_high_keep = 0
        else:
            _es_high_keep = _es_low_keep = 0

        _is_degen = (
            reward_std         < _ES_DEGEN_THR and
            advantage_abs_mean < _ES_DEGEN_THR and
            abs(loss.item())   < _ES_DEGEN_THR
        )
        _es_degenerate = _es_degenerate + 1 if _is_degen else 0

        if _es_high_keep >= _ES_WINDOW:
            _early_stop_reason = (
                f"mean_keep_ratio={_keep:.3f} > 0.95 "
                f"for {_es_high_keep} consecutive steps — all gates open (trivial keep-all)"
            )
        elif _es_low_keep >= _ES_WINDOW:
            _early_stop_reason = (
                f"mean_keep_ratio={_keep:.3f} < 0.05 "
                f"for {_es_low_keep} consecutive steps — all gates closed (trivial discard-all)"
            )
        elif _es_degenerate >= _ES_WINDOW:
            _early_stop_reason = (
                f"reward_std={reward_std:.2e}, advantage_abs_mean={advantage_abs_mean:.2e}, "
                f"|loss|={abs(loss.item()):.2e} all < {_ES_DEGEN_THR} "
                f"for {_es_degenerate} consecutive steps — degenerate / no-reward-signal"
            )

        if _early_stop_reason is not None:
            logger.warning(f"EARLY STOP at step {global_step + 1}: {_early_stop_reason}")
            global_step += 1
            break

        # ------------------------------------------------------------------
        # 6i. Validation
        # ------------------------------------------------------------------
        if val_data and (global_step + 1) % args.eval_freq == 0:
            eval_metrics = evaluate(method, val_data, args, n_samples=args.eval_samples)
            logger.info(
                f"[eval  step {global_step+1}]  "
                f"accuracy={eval_metrics['eval_accuracy']:.3f}  "
                f"mean_gate={eval_metrics['eval_mean_gate']:.3f}"
            )
            if use_wandb:
                wandb.log(eval_metrics, step=global_step + 1)

        # ------------------------------------------------------------------
        # 6j. Checkpointing
        # ------------------------------------------------------------------
        ckpt_payload = {
            "step":           global_step,
            "reading_policy": reading_policy.state_dict(),
            "optimizer":      optimizer.state_dict() if optimizer is not None else None,
            "args":           vars(args),
        }
        latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
        torch.save(ckpt_payload, latest_path)

        if (global_step + 1) % args.save_freq == 0:
            step_path = os.path.join(args.output_dir, f"checkpoint_step{global_step+1}.pt")
            torch.save(ckpt_payload, step_path)
            logger.info(f"Saved checkpoint → {step_path}")

        global_step += 1

    # -----------------------------------------------------------------------
    # 7. Run summary
    # -----------------------------------------------------------------------
    if _early_stop_reason:
        logger.warning(f"Run ended early at step {global_step} — {_early_stop_reason}")
    else:
        logger.info(f"Run completed normally at step {global_step}.")

    # -----------------------------------------------------------------------
    # 8. Final evaluation
    # -----------------------------------------------------------------------
    if val_data:
        final_metrics = evaluate(method, val_data, args, n_samples=min(1000, len(val_data)))
        logger.info(
            f"Final eval — accuracy={final_metrics['eval_accuracy']:.3f}  "
            f"mean_gate={final_metrics['eval_mean_gate']:.3f}"
        )
        if use_wandb:
            wandb.log({"final_" + k: v for k, v in final_metrics.items()})

    if use_wandb:
        wandb.finish()

    logger.info("Training complete.")


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO training for the LatentMAS-RP Reading Policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model_name",       required=True,
                   help="Path or HuggingFace name of the backbone LLM")
    p.add_argument("--device",           default="cuda",
                   help="Primary device (backbone HF model)")
    p.add_argument("--device2",          default="cuda:1",
                   help="Second device (used only when use_second_hf_model=True)")
    p.add_argument("--use_vllm",         action="store_true",
                   help="Use vLLM for Judger text generation")
    p.add_argument("--use_second_hf_model", action="store_true",
                   help="Load a second HF model for latent embedding (vLLM path)")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # LatentMAS
    p.add_argument("--latent_steps",          type=int,   default=16)
    p.add_argument("--judger_max_new_tokens",  type=int,   default=512)
    p.add_argument("--temperature",            type=float, default=0.7)
    p.add_argument("--do_sample",              action="store_true", default=False,
                   help="Use sampling for Judger generation (default: greedy)")
    p.add_argument("--top_p",                  type=float, default=0.95)
    p.add_argument("--prompt_style",           default="sequential",
                   choices=["sequential", "hierarchical"])
    p.add_argument("--think",                  action="store_true")
    p.add_argument("--latent_space_realign",        action="store_true")
    p.add_argument("--realign_matrix_is_identity",  action="store_true",
                   help="When set, use identity matrix (norm-rescale only) even if latent_space_realign=True")
    p.add_argument("--task",                   default="gsm8k",
                   choices=list(REWARD_FN.keys()))

    # Reading Policy hyper-parameters
    p.add_argument("--rp_temperature",      type=float, default=1.0,
                   help="Gumbel-Sigmoid temperature τ")
    p.add_argument("--rp_hard_threshold",   type=float, default=0.5,
                   help="Hard gate threshold δ at evaluation")
    p.add_argument("--rp_sparsity_lambda",  type=float, default=0.05,
                   help="Mean keep-ratio sparsity penalty weight λ (fixed). "
                        "Ablation: {0.02, 0.05, 0.10}. Ignored if "
                        "--rp_lambda_warmup_steps > 0.")
    p.add_argument("--rp_lambda_phase1_end",   type=int,   default=0,
                   help="Step at which phase-1 λ ramp ends (0 = no warmup).")
    p.add_argument("--rp_lambda_phase1_start", type=float, default=0.00,
                   help="λ at step 0 of phase-1 ramp.")
    p.add_argument("--rp_lambda_phase1_stop",  type=float, default=0.02,
                   help="λ at end of phase-1 ramp.")
    p.add_argument("--rp_lambda_phase2_end",   type=int,   default=0,
                   help="Step at which phase-2 fixed λ ends (0 = no phase 2).")
    p.add_argument("--rp_lambda_phase2_val",   type=float, default=0.05,
                   help="λ during phase 2.")
    p.add_argument("--rp_lambda_phase3_val",   type=float, default=0.10,
                   help="λ from phase-3 onward.")
    p.add_argument("--rp_ema_momentum",     type=float, default=0.99,
                   help="EMA momentum (used only by ReadingPolicy.update; "
                        "the GRPO loop uses group-relative normalisation instead)")
    p.add_argument("--rp_lr",              type=float, default=3e-5,
                   help="Adam learning rate for GateNet")
    p.add_argument("--rp_chunks_per_agent", type=int,  default=4,
                   help="Number of latent chunks per agent; chunk_size = latent_steps / chunks_per_agent")
    p.add_argument("--rp_baseline",        default="within_question",
                   choices=["within_question", "ema"],
                   help="Baseline for REINFORCE: 'within_question' (default) uses "
                        "per-question mean of G rollout rewards; 'ema' uses EMA scalar")
    p.add_argument("--rp_entropy_coef",    type=float, default=0.01,
                   help="β for entropy bonus: L_total = L_policy − β·H(π). "
                        "Set 0 to disable. Recommended range: 0.005–0.01")
    p.add_argument("--max_grad_norm",      type=float, default=1.0,
                   help="Gradient clipping norm (0 = disabled)")

    # GRPO
    p.add_argument("--n_rollouts",  type=int,   default=2,
                   help="G — number of gate samples per question per step")
    p.add_argument("--batch_size",  type=int,   default=2,
                   help="Number of questions per gradient step")
    p.add_argument("--max_steps",   type=int,   default=500,
                   help="Total gradient steps")

    # Data
    p.add_argument("--train_file",   required=True,
                   help="Training dataset (.parquet or .jsonl)")
    p.add_argument("--val_file",     default=None,
                   help="Validation dataset (.parquet or .jsonl)")
    p.add_argument("--question_col", default="problem",
                   help="Column / key holding the question text")
    p.add_argument("--answer_col",   default="answer",
                   help="Column / key holding the ground-truth answer")

    # Output & logging
    p.add_argument("--output_dir",   default="checkpoints/reading_policy")
    p.add_argument("--save_freq",    type=int, default=50,
                   help="Save a named checkpoint every N steps")
    p.add_argument("--log_freq",     type=int, default=10,
                   help="Log metrics every N steps")
    p.add_argument("--eval_freq",    type=int, default=50,
                   help="Run validation every N steps")
    p.add_argument("--eval_samples", type=int, default=200,
                   help="Number of validation samples per evaluation")
    p.add_argument("--resume_from",  default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--seed",         type=int, default=42)

    # wandb
    p.add_argument("--wandb_project",  default=None,
                   help="wandb project name (None = disabled)")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_wandb",       action="store_true",
                   help="Disable wandb even if --wandb_project is set")

    return p.parse_args()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
