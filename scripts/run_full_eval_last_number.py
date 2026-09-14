"""
Re-evaluate all gate-ablation methods on 500 samples using the
last_number grader so every method is on the same scale.

Methods run in a single model load. Each batch shares one P/C/R
forward pass; run_judger_with_new_gates is called once per method.

Methods evaluated:
  full_latent           [1]*12
  no_latent             [0]*12
  fixed_late_half       [0,0,1,1, 0,0,1,1, 0,0,1,1]
  fixed_rp200_top       [0,0,1,0, 0,0,1,1, 1,0,1,1]   <- already in JSONL
  random_50_s{42,43,44,45}  6 random gates per sample, seeded
  rp_step200_top6       dynamic GateNet top-k=6         <- already in JSONL

Methods already evaluated with last_number (skipped here, loaded from JSONL):
  rp_step200_top6, fixed_rp200_top_pattern

Outputs (OUT_DIR):
  <method>.jsonl          per-sample records
  all_methods_summary.txt final accuracy table
  all_methods.csv         merged CSV (one row per sample, one col per method)
"""
import argparse
import copy
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH    = "/root/Agentic-Communication-single_model-main/models/Qwen3-4B"
TRAIN_FILE    = os.path.join(ROOT, "data/gsm8k/train.parquet")
N_GATES       = 12
GATE_NAMES    = ["P1","P2","P3","P4","C1","C2","C3","C4","R1","R2","R3","R4"]

# Precomputed predictions — skip inference, re-grade from JSONL
PRECOMPUTED = {
    "rp_step200_top6":
        "/tmp/rp_test_run/paired_comparison/paired_rp_step200_top6_0_500.jsonl",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Grader ──────────────────────────────────────────────────────────────
def _last_num(text: str) -> Optional[str]:
    nums = re.findall(r"-?\d+(?:[\.,]\d+)*", text.replace(",", ""))
    return nums[-1].replace(",", "") if nums else None

def grade(prediction: str, gold: str) -> float:
    pred = _last_num(prediction)
    ref  = _last_num(str(gold))
    if pred is None or ref is None:
        return 0.0
    try:
        return 1.0 if float(pred) == float(ref) else 0.0
    except ValueError:
        return 0.0

def qhash(q: str) -> str:
    return hashlib.md5(q.encode()).hexdigest()[:12]


# ── Gate builders ────────────────────────────────────────────────────────
def make_fixed(B: int, pattern: List[int], device) -> torch.Tensor:
    t = torch.tensor(pattern, dtype=torch.float32, device=device)
    return t.unsqueeze(0).expand(B, -1).clone()

def make_random_k(B: int, k: int, n: int, rng: np.random.Generator,
                  device) -> torch.Tensor:
    gates = torch.zeros(B, n, dtype=torch.float32, device=device)
    for b in range(B):
        idx = rng.choice(n, size=k, replace=False)
        gates[b, idx] = 1.0
    return gates

def topk_gates(logits: torch.Tensor, k: int) -> torch.Tensor:
    B, N = logits.shape
    g = torch.zeros(B, N, dtype=torch.float32, device=logits.device)
    g.scatter_(-1, logits.topk(k, dim=-1).indices, 1.0)
    return g


# ── CLI ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",            required=True)
    p.add_argument("--train_file",            default=TRAIN_FILE)
    p.add_argument("--num_samples",           type=int, default=500)
    p.add_argument("--offset",                type=int, default=0)
    p.add_argument("--batch_size",            type=int, default=2)
    p.add_argument("--judger_max_new_tokens", type=int, default=512)
    p.add_argument("--top_k",                type=int, default=6)
    p.add_argument("--random_seeds",          type=int, nargs="+",
                   default=[42, 43, 44, 45])
    p.add_argument("--out_dir",
                   default="/tmp/rp_test_run/full_eval_last_number")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.offset}_{args.offset + args.num_samples}"

    # ── checkpoint ──────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt             = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args        = ckpt["args"]
    ckpt_step        = ckpt.get("step", "?")
    latent_steps     = ckpt_args["latent_steps"]
    chunks_per_agent = ckpt_args["rp_chunks_per_agent"]
    logger.info(f"  step={ckpt_step}  latent_steps={latent_steps}  "
                f"chunks_per_agent={chunks_per_agent}")

    # ── dataset ─────────────────────────────────────────────────────
    df      = pd.read_parquet(args.train_file)
    rows    = df.iloc[args.offset:args.offset + args.num_samples].to_dict("records")
    samples = [
        {
            "sample_id":     args.offset + i,
            "question":      r["problem"],
            "gold":          str(r["answer"]),
            "question_hash": qhash(r["problem"]),
        }
        for i, r in enumerate(rows)
    ]
    batches = [samples[i:i+args.batch_size]
               for i in range(0, len(samples), args.batch_size)]
    n_steps = len(batches)
    logger.info(f"Samples: {len(samples)}  Batches: {n_steps}")

    # ── define methods ───────────────────────────────────────────────
    # Each entry: (name, kind, extra)
    # kind in: fixed | random_k | rp_topk
    FIXED_LATE  = [0,0,1,1, 0,0,1,1, 0,0,1,1]
    FIXED_RP200 = [0,0,1,0, 0,0,1,1, 1,0,1,1]

    methods_to_run = [
        ("full_latent",         "fixed",   [1]*N_GATES),
        ("no_latent",           "fixed",   [0]*N_GATES),
        ("fixed_late_half",     "fixed",   FIXED_LATE),
        ("fixed_rp200_top",     "fixed",   FIXED_RP200),
    ] + [
        (f"random_50_s{s}",     "random_k", s)
        for s in args.random_seeds
    ] + [
        ("rp_step200_top6",     "rp_topk", args.top_k),
    ]

    # ── model ────────────────────────────────────────────────────────
    import argparse as _ap
    from agent_system.multi_turn_rollout.models import ModelWrapper
    from agent_system.multi_turn_rollout.methods.latent_mas import LatentMASMethod
    from agent_system.multi_turn_rollout.reading_policy import ReadingPolicy

    latent_args = _ap.Namespace(
        model_name                 = MODEL_PATH,
        device                     = "cuda",
        device2                    = "cuda",
        latent_steps               = latent_steps,
        max_new_tokens             = args.judger_max_new_tokens,
        temperature                = 0.7,
        top_p                      = 0.95,
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
        task                       = ckpt_args.get("task", "gsm8k"),
        text_mas_context_length    = -1,
        latent_only                = False,
        sequential_info_only       = False,
        use_reading_policy         = True,
        rp_temperature             = ckpt_args.get("rp_temperature", 1.0),
        rp_hard_threshold          = ckpt_args.get("rp_hard_threshold", 0.5),
        rp_sparsity_lambda         = ckpt_args.get("rp_sparsity_lambda", 0.05),
        rp_ema_momentum            = ckpt_args.get("rp_ema_momentum", 0.99),
        rp_lr                      = ckpt_args.get("rp_lr", 3e-5),
        rp_chunks_per_agent        = chunks_per_agent,
        rp_entropy_coef            = ckpt_args.get("rp_entropy_coef", 0.01),
    )

    logger.info("Loading model...")
    t0 = time.time()
    mw = ModelWrapper(model_name=MODEL_PATH, device=torch.device("cuda"),
                      use_vllm=False, args=latent_args)
    for param in mw.model.parameters():
        param.requires_grad_(False)

    method = LatentMASMethod(
        model=mw, latent_steps=latent_steps,
        judger_max_new_tokens=args.judger_max_new_tokens,
        temperature=0.7, top_p=0.95, do_sample=False,
        generate_bs=args.batch_size, args=latent_args,
    )
    rp: ReadingPolicy = method.reading_policy
    rp.build_gate_nets_from_state_dict(ckpt["reading_policy"])
    rp.load_state_dict(ckpt["reading_policy"])
    rp.eval()
    logger.info(f"Model ready in {time.time()-t0:.1f}s")
    device = next(mw.model.parameters()).device

    # ── open output files ────────────────────────────────────────────
    file_handles = {}
    all_rewards  = {name: [] for name, *_ in methods_to_run}
    for name, *_ in methods_to_run:
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        file_handles[name] = open(path, "w")

    # ── per-method RNG state (seeded once, advanced per batch) ──────
    rngs = {}
    for name, kind, extra in methods_to_run:
        if kind == "random_k":
            rngs[name] = np.random.default_rng(extra)

    # ── evaluation loop ───────────────────────────────────────────────
    logger.info(f"Starting evaluation: {n_steps} steps × {len(methods_to_run)} methods")
    t_loop = time.time()

    for step, batch in enumerate(batches, start=1):
        batch_input = [{"question": s["question"], "gold": s["gold"]}
                       for s in batch]
        B = len(batch)

        # shared P/C/R forward pass
        with torch.no_grad():
            state = method.collect_pre_judger_state(batch_input)

        # RP logits (needed only for rp_topk, but cheap)
        with torch.no_grad():
            _, logits = rp.compute_gates(
                state.query, state.block_summaries, training=False
            )

        step_log_parts = []
        for name, kind, extra in methods_to_run:
            # build gate tensor
            if kind == "fixed":
                gates = make_fixed(B, extra, device)
            elif kind == "random_k":
                gates = make_random_k(B, args.top_k, N_GATES,
                                      rngs[name], device)
            elif kind == "rp_topk":
                gates = topk_gates(logits, extra)
            else:
                raise ValueError(kind)

            state_copy = copy.deepcopy(state)
            with torch.no_grad():
                texts, _, _ = method.run_judger_with_new_gates(
                    state_copy, training=False, forced_gates=gates
                )

            for b, s in enumerate(batch):
                pred = texts[b] if b < len(texts) else ""
                r    = grade(pred, s["gold"])
                all_rewards[name].append(r)

                file_handles[name].write(json.dumps({
                    "sample_id":     s["sample_id"],
                    "question_hash": s["question_hash"],
                    "method":        name,
                    "grader":        "last_number",
                    "reward":        r,
                    "prediction":    pred,
                    "gold_answer":   s["gold"],
                    "gate_pattern":  [int(g) for g in gates[b].tolist()],
                }) + "\n")

            step_r  = all_rewards[name]
            run_avg = sum(step_r) / len(step_r)
            step_log_parts.append(f"{name[:16]}={run_avg:.3f}")

        # flush every 10 steps
        if step % 10 == 0:
            for fh in file_handles.values():
                fh.flush()

        elapsed = time.time() - t_loop
        eta     = elapsed / step * (n_steps - step)
        logger.info(f"[step {step:>4}/{n_steps}]  "
                    + "  ".join(step_log_parts)
                    + f"  eta={eta/60:.0f}m")

    for fh in file_handles.values():
        fh.close()

    # ── load precomputed methods (re-grade with last_number) ─────────
    precomp_rewards = {}
    for name, jsonl_path in PRECOMPUTED.items():
        if not os.path.exists(jsonl_path):
            logger.warning(f"Precomputed JSONL not found: {jsonl_path}")
            continue
        recs = [json.loads(l) for l in open(jsonl_path)]
        rewards = [grade(r["prediction"], r["gold_answer"]) for r in recs]
        precomp_rewards[name] = rewards
        # write to out_dir as well
        out_path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(out_path, "w") as f:
            for r, rec in zip(rewards, recs):
                f.write(json.dumps({
                    "sample_id":     rec["sample_id"],
                    "question_hash": rec["question_hash"],
                    "method":        name,
                    "grader":        "last_number",
                    "reward":        r,
                    "prediction":    rec["prediction"],
                    "gold_answer":   rec["gold_answer"],
                    "gate_pattern":  rec["gate_pattern"],
                }) + "\n")
        logger.info(f"Re-graded {name}: "
                    f"{sum(rewards)/len(rewards):.3f} "
                    f"({int(sum(rewards))}/{len(rewards)})")

    # ── summary ──────────────────────────────────────────────────────
    all_results = {}
    # newly run methods
    for name, *_ in methods_to_run:
        r = all_rewards[name]
        all_results[name] = (sum(r) / len(r), int(sum(r)), len(r))
    # precomputed
    for name, r in precomp_rewards.items():
        all_results[name] = (sum(r) / len(r), int(sum(r)), len(r))

    # ordered for display
    display_order = [
        "full_latent",
        "rp_step200_top6",
        "fixed_rp200_top",
        "fixed_late_half",
        "random_50_s42", "random_50_s43", "random_50_s44", "random_50_s45",
        "no_latent",
    ]

    lines = [
        "",
        "=" * 65,
        "ALL-METHODS EVALUATION — last_number grader",
        f"  checkpoint: step {ckpt_step}   indices: {args.offset}:{args.offset+args.num_samples}",
        "=" * 65,
        f"  {'Method':<28} {'Accuracy':>9}  {'Correct':>10}",
        f"  {'-'*52}",
    ]
    for name in display_order:
        if name not in all_results:
            continue
        acc, correct, n = all_results[name]
        lines.append(f"  {name:<28} {acc:>9.3f}  {correct:>6}/{n}")

    # random_50 mean
    rand_keys = [k for k in all_results if k.startswith("random_50_s")]
    if rand_keys:
        rand_accs = [all_results[k][0] for k in rand_keys]
        rand_mean = sum(rand_accs) / len(rand_accs)
        rand_std  = float(np.std(rand_accs))
        lines.append(f"  {'random_50 (mean±std)':<28} "
                     f"{rand_mean:>9.3f}  ±{rand_std:.3f}")

    lines.append("=" * 65)
    summary = "\n".join(lines)
    logger.info(summary)

    summ_path = os.path.join(args.out_dir, "all_methods_summary.txt")
    with open(summ_path, "w") as f:
        f.write(summary.strip() + "\n")
    logger.info(f"Summary → {summ_path}")

    # ── merged CSV ────────────────────────────────────────────────────
    all_records_by_id = {}
    for name in list(all_results.keys()):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        for rec in (json.loads(l) for l in open(path)):
            sid = rec["sample_id"]
            if sid not in all_records_by_id:
                all_records_by_id[sid] = {
                    "sample_id": sid,
                    "question_hash": rec["question_hash"],
                    "gold_answer": rec["gold_answer"],
                }
            all_records_by_id[sid][f"reward_{name}"] = rec["reward"]
            all_records_by_id[sid][f"gate_{name}"]   = str(rec["gate_pattern"])

    csv_path = os.path.join(args.out_dir, "all_methods.csv")
    method_names = [n for n in display_order if n in all_results]
    fieldnames   = (["sample_id", "question_hash", "gold_answer"]
                    + [f"reward_{n}" for n in method_names]
                    + [f"gate_{n}"   for n in method_names])
    with open(csv_path, "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for sid in sorted(all_records_by_id):
            w.writerow(all_records_by_id[sid])
    logger.info(f"CSV → {csv_path}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
