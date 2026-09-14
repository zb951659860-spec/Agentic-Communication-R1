#!/usr/bin/env python3
"""
Realignment compute ablation: identity vs. ridge-regression W_a on Qwen3-4B.

Group A — Identity mapping
    latent_space_realign=False, realign_matrix_is_identity=False  →  use_identity=True
    W_a = I  (no rotation, norm-rescale still applied)

Group B — Computed realignment
    latent_space_realign=True,  realign_matrix_is_identity=False  →  use_identity=False
    W_a = (W_out^T W_out + λI)^{-1} W_out^T W_in  (ridge regression)

Common settings:
    model=Qwen3-4B, task=GSM8K, latent_steps=16, gate_mode=bypass,
    use_reading_policy=False, batch_size=2, do_sample=False,
    temperature=1.0, top_p=1.0, judger_max_new_tokens=512, num_samples=50

Logs: /tmp/rp_test_run/realign_ablation/groupA_identity.log
      /tmp/rp_test_run/realign_ablation/groupB_computed.log
"""

import argparse as _ap
import logging
import math
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
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH  = "/root/Agentic-Communication-single_model-main/models/Qwen3-4B"
TRAIN_FILE  = os.path.join(ROOT, "data/gsm8k/train.parquet")
OUTPUT_DIR  = "/tmp/rp_test_run/realign_ablation"
LATENT_STEPS = 16
BATCH_SIZE   = 2
N_STEPS      = 50
SEED         = 42

# ---------------------------------------------------------------------------
# Reward helper
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

def load_dataset(path):
    import pandas as pd
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        records.append({
            "question": row["problem"],
            "gold":     row["answer"],
            "solution": row.get("solution", row.get("answer", "")),
        })
    return records

# ---------------------------------------------------------------------------
# Build latent_args Namespace
# ---------------------------------------------------------------------------
def make_args(latent_space_realign: bool, realign_matrix_is_identity: bool) -> _ap.Namespace:
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
        latent_space_realign       = latent_space_realign,
        realign_matrix_is_identity = realign_matrix_is_identity,
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
        use_reading_policy         = False,
    )

# ---------------------------------------------------------------------------
# Single-group runner
# ---------------------------------------------------------------------------
def run_group(
    group_name: str,
    latent_space_realign: bool,
    realign_matrix_is_identity: bool,
    train_data: list,
    log_path: str,
    seed: int = SEED,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    log = logging.getLogger(f"ablation.{group_name}")
    log.setLevel(logging.INFO)
    log.propagate = False
    log.addHandler(fh)
    log.addHandler(logging.StreamHandler())

    args = make_args(latent_space_realign, realign_matrix_is_identity)

    # use_identity mirrors _build_latent_realign_matrix logic
    use_identity = (not latent_space_realign) or realign_matrix_is_identity

    from agent_system.multi_turn_rollout.models import ModelWrapper
    from agent_system.multi_turn_rollout.methods.latent_mas import LatentMASMethod

    log.info(f"=== Group: {group_name}  latent_steps={LATENT_STEPS}  B={BATCH_SIZE}  n_steps={N_STEPS} ===")
    log.info(f"Loading model: {MODEL_PATH}")

    mw = ModelWrapper(
        model_name=MODEL_PATH,
        device=torch.device("cuda"),
        use_vllm=False,
        args=args,
    )
    for p in mw.model.parameters():
        p.requires_grad_(False)

    # Config header
    tied = getattr(mw, "input_output_embeddings_tied", False)
    log.info(
        f"    realign_enabled={latent_space_realign}  "
        f"matrix_is_identity={use_identity}  "
        f"realign_matrix_is_identity_flag={realign_matrix_is_identity}  "
        f"input_output_embeddings_tied={tied}"
    )

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

    mw.relative_change_history.clear()

    total_reward = 0.0
    nonzero_steps = 0
    step_dt_list: List[float] = []

    for step in range(N_STEPS):
        batch = random.sample(train_data, BATCH_SIZE)

        hist_before = len(mw.relative_change_history)

        results = method.run_batch(batch)

        # Per-step d_t (from latent calls during this step)
        hist_after = mw.relative_change_history[hist_before:]
        step_dt = float(np.mean(hist_after)) if hist_after else 0.0
        step_dt_list.append(step_dt)

        # Re-score against raw parquet gold (run_batch compares extracted pred to
        # the full solution string, which never matches; check_gsm8k handles it)
        rewards = [
            check_gsm8k(r.get("raw_prediction", ""), item["gold"])
            for r, item in zip(results, batch)
        ]
        mean_r  = float(np.mean(rewards))
        total_reward += mean_r
        if mean_r > 0:
            nonzero_steps += 1

        log.info(
            f"[step {step+1:>3}]  reward={mean_r:.3f}  "
            f"d_t={step_dt:.6f}  "
            f"rewards={[round(r,2) for r in rewards]}"
        )

        # Sample output on first step
        if step == 0:
            for i, (res, item) in enumerate(zip(results, batch)):
                log.info(f"  Q[{i}]: {item['question'][:80]}...")
                log.info(f"  gold:  {item['gold'][-60:].strip()}")
                log.info(f"  pred:  {res.get('raw_prediction','')[:120].strip()}")

    avg_reward = total_reward / N_STEPS
    hit_rate   = nonzero_steps / N_STEPS
    mean_dt    = float(np.mean(step_dt_list)) if step_dt_list else 0.0
    log.info(
        f"=== DONE {group_name}: avg_reward={avg_reward:.4f}  "
        f"steps_with_reward={nonzero_steps}/{N_STEPS}  hit_rate={hit_rate:.1%}  "
        f"delta_from_identity={mean_dt:.6f} ==="
    )

    del method
    del mw
    torch.cuda.empty_cache()

    return avg_reward, hit_rate, mean_dt

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    logger = logging.getLogger("ablation.main")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading dataset …")
    train_data = load_dataset(TRAIN_FILE)
    logger.info(f"Loaded {len(train_data)} examples")

    groups = [
        # (name,              realign, identity_flag)
        ("groupA_identity",   False,   False),   # use_identity=True  (W_a = I)
        ("groupB_computed",   True,    False),   # use_identity=False (W_a = ridge)
    ]

    summary = []
    for name, realign, id_flag in groups:
        log_path = os.path.join(OUTPUT_DIR, f"{name}.log")
        logger.info(f"\n{'='*65}\n  {name}  realign={realign}  id_flag={id_flag}\n{'='*65}")
        avg_r, hit, dt = run_group(
            group_name=name,
            latent_space_realign=realign,
            realign_matrix_is_identity=id_flag,
            train_data=train_data,
            log_path=log_path,
        )
        summary.append((name, realign, id_flag, avg_r, hit, dt))

    logger.info("\n" + "="*65)
    logger.info("SUMMARY")
    logger.info(f"  {'Group':<28} {'realign':>8} {'identity':>9}  {'avg_r':>7}  {'hit%':>7}  {'delta':>10}")
    logger.info("  " + "-"*68)
    for name, realign, id_flag, avg_r, hit, dt in summary:
        use_id = (not realign) or id_flag
        logger.info(
            f"  {name:<28} {str(realign):>8} {str(use_id):>9}  "
            f"{avg_r:>7.4f}  {hit*100:>6.1f}%  {dt:>10.6f}"
        )
    logger.info("="*65)

if __name__ == "__main__":
    main()
