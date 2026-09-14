"""
Paired comparison: RP dynamic top-k=6 vs fixed_rp200_top_pattern
on the SAME 500 samples (indices 0:500 of GSM8K train).

Both methods share one P/C/R forward pass per batch
(collect_pre_judger_state once), then run_judger_with_new_gates is
called twice with different gate masks — halving backbone compute vs
two independent eval runs.

Outputs (--out_dir):
  paired_rp_step200_top6_0_500.jsonl
  paired_fixed_rp200_top_pattern_0_500.jsonl
  paired_comparison.csv          (merged, one row per sample)
  paired_summary.txt
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

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH    = "/root/Agentic-Communication-single_model-main/models/Qwen3-4B"
TRAIN_FILE    = os.path.join(ROOT, "data/gsm8k/train.parquet")
N_GATES       = 12
FIXED_PATTERN = [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1]
GATE_NAMES    = ["P1","P2","P3","P4","C1","C2","C3","C4","R1","R2","R3","R4"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",              required=True)
    p.add_argument("--train_file",              default=TRAIN_FILE)
    p.add_argument("--num_samples",             type=int, default=500)
    p.add_argument("--offset",                  type=int, default=0)
    p.add_argument("--top_k",                   type=int, default=6)
    p.add_argument("--batch_size",              type=int, default=2)
    p.add_argument("--judger_max_new_tokens",   type=int, default=512)
    p.add_argument("--out_dir",
                   default="/tmp/rp_test_run/paired_comparison")
    p.add_argument("--fixed_pattern", type=int, nargs="+",
                   default=FIXED_PATTERN,
                   help="12-element binary gate mask for the fixed experiment")
    return p.parse_args()


# ── Grader ──────────────────────────────────────────────────────────────
def _last_number(text: str):
    nums = re.findall(r"-?\d+(?:[\.,]\d+)*", text.replace(",", ""))
    return nums[-1].replace(",", "") if nums else None

def check_gsm8k(prediction: str, gold: str) -> float:
    pred = _last_number(prediction)
    ref  = _last_number(str(gold))
    if pred is None or ref is None:
        return 0.0
    try:
        return 1.0 if float(pred) == float(ref) else 0.0
    except ValueError:
        return 0.0

def qhash(q: str) -> str:
    return hashlib.md5(q.encode()).hexdigest()[:12]


# ── Gate helpers ────────────────────────────────────────────────────────
def topk_gates(logits: torch.Tensor, k: int) -> torch.Tensor:
    B, N = logits.shape
    gates = torch.zeros(B, N, dtype=torch.float32, device=logits.device)
    gates.scatter_(-1, logits.topk(k, dim=-1).indices, 1.0)
    return gates

def make_fixed_gates(B: int, pattern: list, device) -> torch.Tensor:
    t = torch.tensor(pattern, dtype=torch.float32, device=device)
    return t.unsqueeze(0).expand(B, -1).clone()


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tag = f"{args.offset}_{args.offset + args.num_samples}"

    # ── checkpoint ──────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt      = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]
    ckpt_step = ckpt.get("step", "?")
    latent_steps     = ckpt_args["latent_steps"]
    chunks_per_agent = ckpt_args["rp_chunks_per_agent"]
    logger.info(f"  step={ckpt_step}  latent_steps={latent_steps}  "
                f"chunks_per_agent={chunks_per_agent}")

    # ── dataset ─────────────────────────────────────────────────────
    df   = pd.read_parquet(args.train_file)
    rows = df.iloc[args.offset : args.offset + args.num_samples].to_dict("records")
    logger.info(f"Samples: {len(rows)} "
                f"(indices {args.offset}:{args.offset + args.num_samples})")

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
    logger.info(f"Batches: {n_steps} × {args.batch_size}")

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
        model=mw,
        latent_steps=latent_steps,
        judger_max_new_tokens=args.judger_max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        do_sample=False,
        generate_bs=args.batch_size,
        args=latent_args,
    )

    rp: ReadingPolicy = method.reading_policy
    rp.build_gate_nets_from_state_dict(ckpt["reading_policy"])
    rp.load_state_dict(ckpt["reading_policy"])
    rp.eval()
    logger.info(f"Model ready in {time.time() - t0:.1f}s")

    device = next(mw.model.parameters()).device

    # ── output files ─────────────────────────────────────────────────
    exp1_name = f"paired_rp_step200_top6_{tag}"
    exp2_name = f"paired_fixed_rp200_top_pattern_{tag}"
    path_exp1 = os.path.join(args.out_dir, exp1_name + ".jsonl")
    path_exp2 = os.path.join(args.out_dir, exp2_name + ".jsonl")

    fh1 = open(path_exp1, "w")
    fh2 = open(path_exp2, "w")

    logger.info(f"Writing exp1 → {path_exp1}")
    logger.info(f"Writing exp2 → {path_exp2}")
    logger.info(f"Fixed pattern: {args.fixed_pattern}")
    logger.info("-" * 70)

    # ── eval loop ────────────────────────────────────────────────────
    rewards_exp1 = []
    rewards_exp2 = []
    t_loop = time.time()

    for step, batch in enumerate(batches, start=1):
        batch_input = [{"question": s["question"], "gold": s["gold"]}
                       for s in batch]

        # ── shared P/C/R forward pass ──────────────────────────────
        with torch.no_grad():
            state = method.collect_pre_judger_state(batch_input)

        # ── Exp 1: RP dynamic top-k ─────────────────────────────────
        with torch.no_grad():
            _, logits = rp.compute_gates(
                state.query, state.block_summaries, training=False
            )
        gates_exp1 = topk_gates(logits, args.top_k)   # [B, 12]

        state_for_exp1 = copy.deepcopy(state)
        with torch.no_grad():
            texts1, _, _ = method.run_judger_with_new_gates(
                state_for_exp1, training=False, forced_gates=gates_exp1
            )

        # ── Exp 2: fixed pattern ─────────────────────────────────────
        B = len(batch)
        gates_exp2 = make_fixed_gates(B, args.fixed_pattern, device)

        state_for_exp2 = copy.deepcopy(state)
        with torch.no_grad():
            texts2, _, _ = method.run_judger_with_new_gates(
                state_for_exp2, training=False, forced_gates=gates_exp2
            )

        # ── grade + write ────────────────────────────────────────────
        step_r1, step_r2 = [], []
        for b, s in enumerate(batch):
            pred1 = texts1[b] if b < len(texts1) else ""
            pred2 = texts2[b] if b < len(texts2) else ""

            r1 = check_gsm8k(pred1, s["gold"])
            r2 = check_gsm8k(pred2, s["gold"])
            step_r1.append(r1)
            step_r2.append(r2)

            fh1.write(json.dumps({
                "sample_id":     s["sample_id"],
                "question_hash": s["question_hash"],
                "method":        "rp_step200_top6",
                "reward":        r1,
                "prediction":    pred1,
                "gold_answer":   s["gold"],
                "gate_pattern":  [int(g) for g in gates_exp1[b].tolist()],
            }) + "\n")

            fh2.write(json.dumps({
                "sample_id":     s["sample_id"],
                "question_hash": s["question_hash"],
                "method":        "fixed_rp200_top_pattern",
                "reward":        r2,
                "prediction":    pred2,
                "gold_answer":   s["gold"],
                "gate_pattern":  args.fixed_pattern[:],
            }) + "\n")

        rewards_exp1.extend(step_r1)
        rewards_exp2.extend(step_r2)

        # running stats
        n_done    = len(rewards_exp1)
        avg1_run  = sum(rewards_exp1) / n_done
        avg2_run  = sum(rewards_exp2) / n_done
        elapsed   = time.time() - t_loop
        eta       = elapsed / step * (n_steps - step)
        pat0      = [int(g) for g in gates_exp1[0].tolist()]

        logger.info(
            f"[step {step:>4}/{n_steps}]  "
            f"rp={sum(step_r1)/len(step_r1):.2f}  "
            f"fix={sum(step_r2)/len(step_r2):.2f}  "
            f"| run_avg: rp={avg1_run:.3f} fix={avg2_run:.3f}  "
            f"eta={eta/60:.0f}m  "
            f"rp_pat={pat0}"
        )
        fh1.flush()
        fh2.flush()

    fh1.close()
    fh2.close()

    # ── paired analysis ───────────────────────────────────────────────
    N = len(rewards_exp1)
    assert N == len(rewards_exp2) == args.num_samples, \
        f"Sample count mismatch: {N} vs {len(rewards_exp2)} vs {args.num_samples}"

    acc1 = sum(rewards_exp1) / N
    acc2 = sum(rewards_exp2) / N

    both_right  = sum(r == 1 and f == 1 for r,f in zip(rewards_exp1, rewards_exp2))
    both_wrong  = sum(r == 0 and f == 0 for r,f in zip(rewards_exp1, rewards_exp2))
    rp_only     = sum(r == 1 and f == 0 for r,f in zip(rewards_exp1, rewards_exp2))
    fixed_only  = sum(r == 0 and f == 1 for r,f in zip(rewards_exp1, rewards_exp2))
    agree_rate  = (both_right + both_wrong) / N

    summary = "\n".join([
        "",
        "=" * 65,
        "PAIRED COMPARISON SUMMARY",
        "=" * 65,
        f"  samples        : {N}  "
        f"(indices {args.offset}:{args.offset + args.num_samples})",
        f"  checkpoint     : step {ckpt_step}",
        f"  grader         : last-number-in-text",
        f"  fixed_pattern  : {args.fixed_pattern}",
        "",
        f"  {'Method':<32} {'Accuracy':>9}  {'Correct':>8}",
        f"  {'-'*52}",
        f"  {'rp_step200_top6':<32} {acc1:>9.3f}  "
        f"{int(sum(rewards_exp1)):>6}/{N}",
        f"  {'fixed_rp200_top_pattern':<32} {acc2:>9.3f}  "
        f"{int(sum(rewards_exp2)):>6}/{N}",
        f"  {'delta (RP - fixed)':<32} {acc1 - acc2:>+9.3f}",
        "",
        f"  PAIRED BREAKDOWN",
        f"  both correct   : {both_right:>4}  ({both_right/N*100:.1f}%)",
        f"  both wrong     : {both_wrong:>4}  ({both_wrong/N*100:.1f}%)",
        f"  RP only right  : {rp_only:>4}  ({rp_only/N*100:.1f}%)",
        f"  fixed only     : {fixed_only:>4}  ({fixed_only/N*100:.1f}%)",
        f"  agreement rate : {agree_rate:.3f}",
        "=" * 65,
        f"  Files:",
        f"    {path_exp1}",
        f"    {path_exp2}",
    ])
    logger.info(summary)

    summary_path = os.path.join(args.out_dir, "paired_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary.strip() + "\n")
    logger.info(f"Summary → {summary_path}")

    # ── merged CSV ────────────────────────────────────────────────────
    recs1 = [json.loads(l) for l in open(path_exp1)]
    recs2 = [json.loads(l) for l in open(path_exp2)]
    csv_path = os.path.join(args.out_dir, "paired_comparison.csv")
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow([
            "sample_id", "question_hash", "gold_answer",
            "rp_reward",    "rp_gate_pattern",    "rp_prediction",
            "fixed_reward", "fixed_gate_pattern", "fixed_prediction",
            "delta",
        ])
        for r1, r2 in zip(recs1, recs2):
            assert r1["sample_id"] == r2["sample_id"], \
                f"ID mismatch at {r1['sample_id']} vs {r2['sample_id']}"
            w.writerow([
                r1["sample_id"],
                r1["question_hash"],
                r1["gold_answer"],
                r1["reward"],  str(r1["gate_pattern"]), r1["prediction"],
                r2["reward"],  str(r2["gate_pattern"]), r2["prediction"],
                r1["reward"] - r2["reward"],
            ])
    logger.info(f"CSV → {csv_path}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
