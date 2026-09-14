#!/usr/bin/env python3
"""
Gate ablation: fixed and random chunk-gate patterns on Qwen3-4B / GSM8K.

Experiments (run one at a time; confirm before proceeding to the next):
  Exp 1 – full_latent   gate_pattern=[1]*12  keep=12/12  100%
  Exp 2 – no_latent     gate_pattern=[0]*12  keep= 0/12    0%
  Exp 3 – random_75     k=9/12  seed=42  random_fixed_k
  Exp 4 – random_50     k=6/12  seed=42  random_fixed_k
  Exp 5 – random_25     k=3/12  seed=42  random_fixed_k

Common parameters:
  model               Qwen3-4B
  task                gsm8k
  latent_steps        16
  chunks_per_agent    4
  chunk_size          4
  total_gates         12
  use_reading_policy  True  (RP infrastructure only; gates are forced, not learned)
  train_rp            False
  do_sample           False
  temperature         1.0
  top_p               1.0
  judger_max_new_tokens 512
  batch_size          2
  n_steps             25   (25 batches × 2 = 50 samples total)
  seed                42   (data + gate patterns)
  shuffle             False  (take first 50 rows in dataset order)
  same_eval_indices   True   (same 50 samples for all experiments)
  latent_space_realign       False
  realign_matrix_is_identity False

Logs → /tmp/rp_test_run/gate_ablation/<exp_name>.log
"""

import argparse as _ap
import logging
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH   = "/root/Agentic-Communication-single_model-main/models/Qwen3-4B"
TRAIN_FILE   = os.path.join(ROOT, "data/gsm8k/train.parquet")
OUTPUT_DIR   = "/tmp/rp_test_run/gate_ablation"

LATENT_STEPS      = 16
CHUNKS_PER_AGENT  = 4
CHUNK_SIZE        = LATENT_STEPS // CHUNKS_PER_AGENT   # 4
N_AGENTS          = 3
N_GATES           = N_AGENTS * CHUNKS_PER_AGENT        # 12

BATCH_SIZE  = 2
N_STEPS     = 25       # default: 25 batches × 2 = 50 samples (overridden by --num_samples)
SEED        = 42
GATE_SEED   = 42

# Experiment table:  (name, keep_chunks, gate_mode, gate_pattern_or_None)
#   gate_mode = "fixed"          → use gate_pattern directly
#   gate_mode = "random_fixed_k" → pick keep_chunks gates at random (GATE_SEED)
EXPERIMENTS: List[Tuple[str, int, str, Optional[List[int]]]] = [
    ("full_latent", 12, "fixed",          [1]*N_GATES),
    ("no_latent",    0, "fixed",          [0]*N_GATES),
    ("random_75",    9, "random_fixed_k", None),
    ("random_50",    6, "random_fixed_k", None),
    ("random_25",    3, "random_fixed_k", None),
    # -----------------------------------------------------------------------
    # Best-Fixed-50 ablation: structured keep patterns (keep=6/12, ratio=0.50)
    # -----------------------------------------------------------------------
    # F1: first two chunks of every agent
    ("fixed_early_half",      6, "fixed", [1,1,0,0, 1,1,0,0, 1,1,0,0]),
    # F2: last two chunks of every agent
    ("fixed_late_half",       6, "fixed", [0,0,1,1, 0,0,1,1, 0,0,1,1]),
    # F3: middle two chunks of every agent
    ("fixed_middle",          6, "fixed", [0,1,1,0, 0,1,1,0, 0,1,1,0]),
    # F4: first and last chunk of every agent
    ("fixed_first_last",      6, "fixed", [1,0,0,1, 1,0,0,1, 1,0,0,1]),
    # F5: top-6 pattern extracted from the step-200 RP checkpoint logits
    ("fixed_rp200_top_pattern", 6, "fixed", [0,0,1,0, 0,0,1,1, 1,0,1,1]),
    # -----------------------------------------------------------------------
    # Held-out ablation (eval_indices = 100:200)
    # Run with: --offset 100 --num_samples 100 --tag n100_heldout
    # -----------------------------------------------------------------------
    ("heldout_full_latent",           12, "fixed",          [1]*N_GATES),
    ("heldout_no_latent",              0, "fixed",          [0]*N_GATES),
    ("heldout_random_50",              6, "random_fixed_k", None),
    ("heldout_fixed_late_half",        6, "fixed",          [0,0,1,1, 0,0,1,1, 0,0,1,1]),
    ("heldout_fixed_rp200_top_pattern",6, "fixed",          [0,0,1,0, 0,0,1,1, 1,0,1,1]),
]

# ---------------------------------------------------------------------------
# GSM8K scoring helpers
# ---------------------------------------------------------------------------
def _extract_boxed(text: str) -> Optional[str]:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    return m[-1].strip() if m else None

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
    gv = _normalize(m.group(1) if m else gold.strip())
    pred = _extract_boxed(text)
    if pred is None:
        m2 = re.search(r"####\s*(\S+)", text)
        if m2:
            pred = m2.group(1)
    return 1.0 if (_normalize(pred) is not None and _normalize(pred) == gv) else 0.0

# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_dataset(path: str, n: int = N_STEPS * BATCH_SIZE, offset: int = 0) -> List[dict]:
    import pandas as pd
    df = pd.read_parquet(path)
    slice_ = df.iloc[offset: offset + n]
    records = []
    for _, row in slice_.iterrows():
        records.append({
            "question": row["problem"],
            "gold":     row["answer"],
            "solution": row.get("solution", row.get("answer", "")),
        })
    return records

# ---------------------------------------------------------------------------
# Build the gate tensor for this experiment
# ---------------------------------------------------------------------------
def build_gate_pattern(keep_chunks: int, gate_mode: str,
                       gate_pattern_or_none: Optional[List[int]],
                       gate_seed: int = GATE_SEED) -> List[int]:
    if gate_mode == "fixed":
        return gate_pattern_or_none
    # random_fixed_k: choose keep_chunks indices, rest 0
    rng = np.random.default_rng(gate_seed)
    indices = rng.choice(N_GATES, size=keep_chunks, replace=False)
    pattern = [0] * N_GATES
    for idx in indices:
        pattern[idx] = 1
    return pattern

# ---------------------------------------------------------------------------
# Build args Namespace for LatentMASMethod
# ---------------------------------------------------------------------------
def make_args(use_reading_policy: bool = True) -> _ap.Namespace:
    return _ap.Namespace(
        model_name                 = MODEL_PATH,
        device                     = "cuda",
        device2                    = "cuda",
        latent_steps               = LATENT_STEPS,
        max_new_tokens             = 512,
        temperature                = 1.0,
        top_p                      = 1.0,
        prompt                     = "sequential",
        think                      = False,
        latent_space_realign       = False,
        realign_matrix_is_identity = False,
        use_vllm                   = False,
        use_second_HF_model        = False,
        tensor_parallel_size       = 1,
        gpu_memory_utilization     = 0.9,
        enable_prefix_caching      = False,
        method                     = "latent_mas",
        task                       = "gsm8k",
        text_mas_context_length    = -1,
        latent_only                = False,
        sequential_info_only       = False,
        use_reading_policy         = use_reading_policy,
        # ReadingPolicy hyper-params (values don't matter; RP is not trained)
        rp_chunks_per_agent        = CHUNKS_PER_AGENT,
        rp_entropy_coef            = 0.01,
        rp_temperature             = 1.0,
        rp_hard_threshold          = 0.5,
        rp_sparsity_lambda         = 0.05,
        rp_ema_momentum            = 0.99,
        rp_lr                      = 3e-5,
    )

# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    exp_name: str,
    keep_chunks: int,
    gate_mode: str,
    gate_pattern_or_none: Optional[List[int]],
    eval_batches: List[List[dict]],
    log: logging.Logger,
    n_steps: int = N_STEPS,
    gate_seed: int = GATE_SEED,
) -> dict:

    gate_pattern = build_gate_pattern(keep_chunks, gate_mode, gate_pattern_or_none,
                                      gate_seed=gate_seed)
    keep_ratio   = sum(gate_pattern) / N_GATES

    log.info(f"=== Experiment: {exp_name}  keep_chunks={keep_chunks}/{N_GATES}  "
             f"keep_ratio={keep_ratio:.2f}  gate_mode={gate_mode} ===")
    log.info(f"    gate_pattern={gate_pattern}")
    log.info(f"    latent_steps={LATENT_STEPS}  chunks_per_agent={CHUNKS_PER_AGENT}  "
             f"chunk_size={CHUNK_SIZE}  n_gates={N_GATES}")

    from agent_system.multi_turn_rollout.models import ModelWrapper
    from agent_system.multi_turn_rollout.methods.latent_mas import LatentMASMethod

    args = make_args(use_reading_policy=True)

    mw = ModelWrapper(
        model_name=MODEL_PATH,
        device=torch.device("cuda"),
        use_vllm=False,
        args=args,
    )
    for p in mw.model.parameters():
        p.requires_grad_(False)

    tied = getattr(mw, "input_output_embeddings_tied", False)
    log.info(f"    input_output_embeddings_tied={tied}")

    method = LatentMASMethod(
        model=mw,
        latent_steps=LATENT_STEPS,
        judger_max_new_tokens=512,
        temperature=1.0,
        top_p=1.0,
        do_sample=False,
        generate_bs=BATCH_SIZE,
        args=args,
    )

    total_reward  = 0.0
    nonzero_steps = 0

    for step, batch in enumerate(eval_batches):
        # Fixed gate tensor [B, N_GATES] — same pattern for every item in batch
        g_1d = torch.tensor(gate_pattern, dtype=torch.float32)   # [12]
        forced_gates = g_1d.unsqueeze(0).expand(len(batch), -1)  # [B, 12]

        state = method.collect_pre_judger_state(batch)
        texts, gates_out, _ = method.run_judger_with_new_gates(
            state,
            training=False,
            forced_gates=forced_gates,
        )

        rewards = [
            check_gsm8k(text, item["gold"])
            for text, item in zip(texts, batch)
        ]
        mean_r = float(np.mean(rewards))
        total_reward  += mean_r
        if mean_r > 0:
            nonzero_steps += 1

        log.info(
            f"[step {step+1:>3}]  reward={mean_r:.3f}  "
            f"keep_ratio={keep_ratio:.3f}  "
            f"rewards={[round(r, 2) for r in rewards]}"
        )

        if step == 0:
            for i, (text, item) in enumerate(zip(texts, batch)):
                log.info(f"  Q[{i}]: {item['question'][:80]}...")
                log.info(f"  gold:  {item['gold'][-60:].strip()}")
                log.info(f"  pred:  {text[:120].strip()}")

    avg_reward = total_reward / n_steps
    hit_rate   = nonzero_steps / n_steps

    log.info(
        f"=== DONE {exp_name}: avg_reward={avg_reward:.4f}  "
        f"steps_with_reward={nonzero_steps}/{n_steps}  hit_rate={hit_rate:.1%}  "
        f"keep_ratio={keep_ratio:.2f}  gate_pattern={gate_pattern} ==="
    )

    del method
    del mw
    torch.cuda.empty_cache()

    return {
        "exp_name":    exp_name,
        "keep_chunks": keep_chunks,
        "keep_ratio":  keep_ratio,
        "gate_pattern": gate_pattern,
        "avg_reward":  avg_reward,
        "hit_rate":    hit_rate,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None,
                        help="Run only this experiment name (default: all in order)")
    parser.add_argument("--list", action="store_true",
                        help="List experiment names and exit")
    parser.add_argument("--num_samples", type=int, default=N_STEPS * BATCH_SIZE,
                        help=f"Total evaluation samples (default: {N_STEPS * BATCH_SIZE}). "
                             f"Must be divisible by batch_size={BATCH_SIZE}.")
    parser.add_argument("--tag", type=str, default="",
                        help="Sub-directory tag under OUTPUT_DIR (e.g. 'n100'). "
                             "Keeps different runs separate.")
    parser.add_argument("--gate_seed", type=int, default=None,
                        help="Override the gate seed for random_fixed_k experiments "
                             "(default: uses GATE_SEED=42 from the EXPERIMENTS table). "
                             "Also appends _s<seed> to the experiment name in the log.")
    parser.add_argument("--keep_chunks", type=int, default=None,
                        help="Override keep_chunks when combined with --exp and --gate_seed.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N dataset rows (e.g. 100 for held-out eval on 100:200).")
    cli = parser.parse_args()

    if cli.list:
        for name, k, mode, _ in EXPERIMENTS:
            print(f"  {name}  keep={k}/{N_GATES}  mode={mode}")
        return

    assert cli.num_samples % BATCH_SIZE == 0, \
        f"--num_samples must be divisible by batch_size={BATCH_SIZE}"
    n_steps = cli.num_samples // BATCH_SIZE

    out_dir = os.path.join(OUTPUT_DIR, cli.tag) if cli.tag else OUTPUT_DIR

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    main_log = logging.getLogger("gate_ablation.main")

    os.makedirs(out_dir, exist_ok=True)

    # Pre-select the same eval samples for all experiments (shuffle=False)
    main_log.info(f"Loading dataset: {TRAIN_FILE}")
    flat_samples = load_dataset(TRAIN_FILE, n=cli.num_samples, offset=cli.offset)
    idx_start = cli.offset
    idx_end   = cli.offset + cli.num_samples
    main_log.info(f"Loaded {len(flat_samples)} samples "
                  f"(indices {idx_start}:{idx_end}, n_steps={n_steps})")

    # Group into batches of BATCH_SIZE
    eval_batches = [
        flat_samples[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        for i in range(n_steps)
    ]

    # Determine which experiments to run
    if cli.exp:
        names_to_run = {cli.exp}
        exps = [e for e in EXPERIMENTS if e[0] in names_to_run]
        if not exps:
            main_log.error(f"Unknown experiment '{cli.exp}'. "
                           f"Valid: {[e[0] for e in EXPERIMENTS]}")
            sys.exit(1)
    else:
        exps = list(EXPERIMENTS)

    # Apply CLI overrides (gate_seed / keep_chunks)
    effective_gate_seed = cli.gate_seed if cli.gate_seed is not None else GATE_SEED
    seed_suffix = f"_s{cli.gate_seed}" if cli.gate_seed is not None else ""

    summary = []
    for exp_name, keep_chunks, gate_mode, gate_pattern_or_none in exps:
        # Allow --keep_chunks to override the table value (only for random_fixed_k)
        if cli.keep_chunks is not None and gate_mode == "random_fixed_k":
            keep_chunks = cli.keep_chunks
        # Append seed suffix when a custom gate_seed is requested
        log_exp_name = exp_name + seed_suffix
        log_path = os.path.join(out_dir, f"{log_exp_name}.log")
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
        exp_log = logging.getLogger(
            f"gate_ablation.{log_exp_name}.{cli.tag or 'default'}"
        )
        exp_log.setLevel(logging.INFO)
        exp_log.propagate = False
        exp_log.addHandler(fh)
        exp_log.addHandler(logging.StreamHandler())

        result = run_experiment(
            exp_name=log_exp_name,
            keep_chunks=keep_chunks,
            gate_mode=gate_mode,
            gate_pattern_or_none=gate_pattern_or_none,
            eval_batches=eval_batches,
            log=exp_log,
            n_steps=n_steps,
            gate_seed=effective_gate_seed,
        )
        summary.append(result)

    # Summary table
    main_log.info("\n" + "=" * 70)
    main_log.info("GATE ABLATION SUMMARY")
    main_log.info(f"  {'Experiment':<20} {'keep':>6} {'ratio':>7}  {'avg_r':>7}  {'hit%':>7}  gate_pattern")
    main_log.info("  " + "-" * 68)
    for r in summary:
        main_log.info(
            f"  {r['exp_name']:<20} {r['keep_chunks']:>4}/{N_GATES}  "
            f"{r['keep_ratio']:>5.2f}   {r['avg_reward']:>7.4f}  "
            f"{r['hit_rate']*100:>6.1f}%  {r['gate_pattern']}"
        )
    main_log.info("=" * 70)

    # Write summary file
    summary_path = os.path.join(out_dir, "summary.log")
    with open(summary_path, "w") as f:
        f.write("Gate Ablation Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"  {'Experiment':<20} {'keep':>6} {'ratio':>7}  {'avg_r':>7}  {'hit%':>7}  gate_pattern\n")
        f.write("  " + "-" * 68 + "\n")
        for r in summary:
            f.write(
                f"  {r['exp_name']:<20} {r['keep_chunks']:>4}/{N_GATES}  "
                f"{r['keep_ratio']:>5.2f}   {r['avg_reward']:>7.4f}  "
                f"{r['hit_rate']*100:>6.1f}%  {r['gate_pattern']}\n"
            )
        f.write("=" * 70 + "\n")
    main_log.info(f"Summary written to {summary_path}")

if __name__ == "__main__":
    main()
