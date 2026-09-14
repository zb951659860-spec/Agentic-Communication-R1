#!/usr/bin/env python3
"""
Diagnostic experiments for LatentMAS-RP reward debugging.

Three controlled experiments (50 steps each, B=2 questions per step):

  Exp 1 — Full-latent baseline:   all 12 gates = 1  (keep every latent token)
  Exp 2 — Bypass gating:          pass raw KV to Judger (no gate applied at all)
  Exp 3 — No-latent baseline:     all 12 gates = 0  (discard every latent token)

Results are written to:
  /tmp/rp_test_run/exp1_full_latent.log
  /tmp/rp_test_run/exp2_bypass_gates.log
  /tmp/rp_test_run/exp3_no_latent.log
"""

import argparse
import logging
import os
import random
import re
import sys
from typing import List, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Reward helpers (copied from train_reading_policy.py)
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> Optional[str]:
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

def check_gsm8k(text: str, gold: str) -> float:
    m = re.search(r"####\s*(\S+)", gold)
    gold_val = _normalize(m.group(1) if m else gold.strip())
    pred_str = _extract_boxed(text)
    if pred_str is None:
        m2 = re.search(r"####\s*(\S+)", text)
        if m2:
            pred_str = m2.group(1)
    pred_val = _normalize(pred_str)
    return 1.0 if (pred_val is not None and pred_val == gold_val) else 0.0

def load_dataset(path, qcol="problem", acol="answer"):
    import pandas as pd
    df = pd.read_parquet(path)
    return [{"question": row[qcol], "gold": row[acol]} for _, row in df.iterrows()]

# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    method,
    train_data,
    exp_name: str,
    n_steps: int,
    batch_size: int,
    gate_mode: str,     # 'full' | 'bypass' | 'none'
    log_path: str,
    seed: int = 42,
):
    """
    gate_mode:
      'full'   — forced_gates = ones (keep all latent tokens)
      'bypass' — bypass_gating=True (raw KV, no apply_gates called)
      'none'   — forced_gates = zeros (drop all latent tokens)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    log = logging.getLogger(f"diag.{exp_name}")
    log.setLevel(logging.INFO)
    log.propagate = False
    log.addHandler(fh)
    log.addHandler(logging.StreamHandler())

    n_gates = method.reading_policy.n_gates

    # --- Config header ---
    mw = method.model
    _args        = getattr(mw, "args", None)
    realign_en   = bool(getattr(_args, "latent_space_realign",       False)) if _args else False
    mat_identity = bool(
        (not realign_en) or getattr(_args, "realign_matrix_is_identity", False)
    ) if _args else True
    tied         = getattr(mw, "input_output_embeddings_tied", False)
    log.info(f"=== Experiment: {exp_name}  gate_mode={gate_mode}  n_steps={n_steps}  B={batch_size} ===")
    log.info(f"    realign_enabled={realign_en}  matrix_is_identity={mat_identity}  "
             f"input_output_embeddings_tied={tied}")

    # reset relative-change history for this experiment
    mw.relative_change_history.clear()

    total_reward = 0.0
    nonzero_steps = 0
    step_dt_list: List[float] = []

    for step in range(n_steps):
        batch = random.sample(train_data, min(batch_size, len(train_data)))

        # Snapshot history length before this step so we can isolate this step's d_t values
        hist_before = len(mw.relative_change_history)

        # Collect non-judger state (Planner / Critic / Refiner)
        state = method.collect_pre_judger_state(batch)
        B = len(batch)

        # Build forced gates or set bypass flag
        if gate_mode == "full":
            fg = torch.ones(B, n_gates)
            bypass = False
        elif gate_mode == "none":
            fg = torch.zeros(B, n_gates)
            bypass = False
        else:  # "bypass"
            fg = None
            bypass = True

        # Run Judger
        texts, gates, _ = method.run_judger_with_new_gates(
            state,
            training=False,
            forced_gates=fg,
            bypass_gating=bypass,
        )

        # Score
        rewards = [check_gsm8k(t, item["gold"]) for t, item in zip(texts, batch)]
        mean_r  = float(np.mean(rewards))
        total_reward += mean_r
        if mean_r > 0:
            nonzero_steps += 1

        # Per-step mean d_t (averaged over all latent calls this step)
        hist_after = mw.relative_change_history[hist_before:]
        step_dt = float(np.mean(hist_after)) if hist_after else 0.0
        step_dt_list.append(step_dt)

        keep_ratio = gates.float().mean().item()
        log.info(
            f"[step {step+1:>3}]  reward={mean_r:.3f}  "
            f"keep_ratio={keep_ratio:.3f}  "
            f"d_t={step_dt:.4f}  "
            f"rewards={[round(r,2) for r in rewards]}"
        )

        # Brief output sample for first step
        if step == 0:
            for i, (t, item) in enumerate(zip(texts, batch)):
                log.info(f"  Q[{i}]: {item['question'][:80]}...")
                log.info(f"  A[{i}] (gold): {item['gold'][-60:].strip()}")
                log.info(f"  A[{i}] (pred): {t[:120].strip()}")

    avg_reward   = total_reward / n_steps
    hit_rate     = nonzero_steps / n_steps
    mean_dt      = float(np.mean(step_dt_list)) if step_dt_list else 0.0
    log.info(
        f"=== DONE {exp_name}: avg_reward={avg_reward:.4f}  "
        f"steps_with_reward={nonzero_steps}/{n_steps}  hit_rate={hit_rate:.1%}  "
        f"delta_from_identity={mean_dt:.6f} ==="
    )
    return avg_reward, hit_rate

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--train_file", default="data/gsm8k/train.parquet")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--n_steps",    type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--latent_steps",       type=int,   default=16)
    p.add_argument("--rp_chunks_per_agent",type=int,   default=4)
    p.add_argument("--judger_max_new_tokens", type=int, default=512)
    p.add_argument("--output_dir", default="/tmp/rp_test_run")
    p.add_argument("--seed",       type=int, default=42)
    # Single-experiment mode: pass --gate_mode and optionally --exp_name
    p.add_argument("--gate_mode",  default=None,
                   choices=["full", "bypass", "none"],
                   help="Run a single experiment with this gate mode instead of all three")
    p.add_argument("--exp_name",   default=None,
                   help="Log filename stem when using --gate_mode (default: derived from gate_mode)")
    p.add_argument("--latent_space_realign",       action="store_true",
                   help="Enable latent realignment pipeline in ModelWrapper")
    p.add_argument("--realign_matrix_is_identity", action="store_true",
                   help="Use identity matrix (norm-rescale only) within the realignment pipeline")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    logger = logging.getLogger("diag.main")

    # --- Build latent_args for ModelWrapper / LatentMASMethod ---
    import argparse as _ap
    latent_args = _ap.Namespace(
        model_name            = args.model_name,
        device                = args.device,
        device2               = args.device,
        latent_steps          = args.latent_steps,
        max_new_tokens        = args.judger_max_new_tokens,
        temperature           = 0.7,
        top_p                 = 0.95,
        prompt                = "sequential",
        think                 = False,
        latent_space_realign       = args.latent_space_realign,
        realign_matrix_is_identity = args.realign_matrix_is_identity,
        use_vllm              = False,
        use_second_HF_model   = False,
        tensor_parallel_size  = 1,
        gpu_memory_utilization= 0.9,
        enable_prefix_caching = False,
        method                = "latent_mas",
        task                  = "gsm8k",
        text_mas_context_length = -1,
        latent_only           = False,
        sequential_info_only  = False,
        use_reading_policy    = True,
        rp_temperature        = 1.0,
        rp_hard_threshold     = 0.5,
        rp_sparsity_lambda    = 0.05,
        rp_ema_momentum       = 0.99,
        rp_lr                 = 3e-5,
        rp_chunks_per_agent   = args.rp_chunks_per_agent,
        rp_entropy_coef       = 0.01,
    )

    from agent_system.multi_turn_rollout.models import ModelWrapper
    from agent_system.multi_turn_rollout.methods.latent_mas import LatentMASMethod

    logger.info(f"Loading model: {args.model_name}")
    model_wrapper = ModelWrapper(
        model_name=args.model_name,
        device=torch.device(args.device),
        use_vllm=False,
        args=latent_args,
    )
    for param in model_wrapper.model.parameters():
        param.requires_grad_(False)

    method = LatentMASMethod(
        model=model_wrapper,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.judger_max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        do_sample=False,
        generate_bs=args.batch_size,
        args=latent_args,
    )
    assert method.reading_policy is not None

    logger.info(f"Loading data: {args.train_file}")
    train_data = load_dataset(args.train_file)
    logger.info(f"Loaded {len(train_data)} examples")

    os.makedirs(args.output_dir, exist_ok=True)

    _default_exp_names = {
        "full":   "exp1_full_latent",
        "bypass": "exp2_bypass_gates",
        "none":   "exp3_no_latent",
    }
    _default_titles = {
        "full":   "Full-latent (all gates=1)",
        "bypass": "Bypass gating (raw KV)",
        "none":   "No-latent (all gates=0)",
    }

    if args.gate_mode is not None:
        fname = args.exp_name or _default_exp_names[args.gate_mode]
        title = f"{fname} — {_default_titles[args.gate_mode]}"
        experiments = [(fname, args.gate_mode, title)]
    else:
        experiments = [
            ("exp1_full_latent",  "full",   "Exp 1 — Full-latent (all gates=1)"),
            ("exp2_bypass_gates", "bypass", "Exp 2 — Bypass gating (raw KV)"),
            ("exp3_no_latent",    "none",   "Exp 3 — No-latent (all gates=0)"),
        ]

    summary_lines = []
    for fname, mode, title in experiments:
        log_path = os.path.join(args.output_dir, f"{fname}.log")
        logger.info(f"\n{'='*60}\n{title}\n{'='*60}")
        avg_r, hit = run_experiment(
            method=method,
            train_data=train_data,
            exp_name=fname,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gate_mode=mode,
            log_path=log_path,
            seed=args.seed,
        )
        summary_lines.append(f"  {title:40s}  avg_reward={avg_r:.4f}  hit_rate={hit:.1%}")

    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    for line in summary_lines:
        logger.info(line)
    logger.info("="*60)

if __name__ == "__main__":
    main()
