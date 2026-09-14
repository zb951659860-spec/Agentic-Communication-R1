"""
Collect per-sample top-k gate selections from the RP checkpoint across N samples
and report per-chunk selection frequency.

Only runs the P/C/R forward pass (collect_pre_judger_state) + GateNet logits.
Skips the Judger entirely — much faster than full eval.

Usage:
    python3 scripts/collect_gate_freq.py \
        --checkpoint /tmp/rp_test_run/rp_training_cont/checkpoint_step200.pt \
        --num_samples 500 \
        --top_k 6
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH  = "/root/Agentic-Communication-single_model-main/models/Qwen3-4B"
TRAIN_FILE  = os.path.join(ROOT, "data/gsm8k/train.parquet")
N_GATES     = 12
BATCH_SIZE  = 2

GATE_NAMES = [
    "P1","P2","P3","P4",
    "C1","C2","C3","C4",
    "R1","R2","R3","R4",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--train_file",  default=TRAIN_FILE)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--offset",      type=int, default=0)
    p.add_argument("--top_k",       type=int, default=6)
    p.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    return p.parse_args()


def topk_gates(logits: torch.Tensor, k: int) -> torch.Tensor:
    B, N = logits.shape
    gates = torch.zeros(B, N, dtype=torch.float32, device=logits.device)
    gates.scatter_(-1, logits.topk(k, dim=-1).indices, 1.0)
    return gates


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt      = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]
    ckpt_step = ckpt.get("step", "?")
    latent_steps     = ckpt_args["latent_steps"]
    chunks_per_agent = ckpt_args["rp_chunks_per_agent"]
    print(f"Checkpoint: step={ckpt_step}  latent_steps={latent_steps}  "
          f"chunks_per_agent={chunks_per_agent}  top_k={args.top_k}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    df   = pd.read_parquet(args.train_file)
    rows = df.iloc[args.offset : args.offset + args.num_samples].to_dict("records")
    print(f"Samples: {len(rows)}  (indices {args.offset}:{args.offset + args.num_samples})")

    batches = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        batches.append([{"question": r["problem"], "gold": str(r["answer"])} for r in chunk])

    # ------------------------------------------------------------------
    # Build LatentMASMethod
    # ------------------------------------------------------------------
    import argparse as _ap
    from agent_system.multi_turn_rollout.models import ModelWrapper
    from agent_system.multi_turn_rollout.methods.latent_mas import LatentMASMethod
    from agent_system.multi_turn_rollout.reading_policy import ReadingPolicy

    latent_args = _ap.Namespace(
        model_name              = MODEL_PATH,
        device                  = "cuda",
        device2                 = "cuda",
        latent_steps            = latent_steps,
        max_new_tokens          = 512,
        temperature             = 0.7,
        top_p                   = 0.95,
        prompt                  = "sequential",
        think                   = False,
        latent_space_realign    = False,
        realign_matrix_is_identity = False,
        use_vllm                = False,
        use_second_HF_model     = False,
        tensor_parallel_size    = 1,
        gpu_memory_utilization  = 0.9,
        enable_prefix_caching   = False,
        method                  = "latent_mas",
        task                    = ckpt_args.get("task", "gsm8k"),
        text_mas_context_length = -1,
        latent_only             = False,
        sequential_info_only    = False,
        use_reading_policy      = True,
        rp_temperature          = ckpt_args.get("rp_temperature", 1.0),
        rp_hard_threshold       = ckpt_args.get("rp_hard_threshold", 0.5),
        rp_sparsity_lambda      = ckpt_args.get("rp_sparsity_lambda", 0.05),
        rp_ema_momentum         = ckpt_args.get("rp_ema_momentum", 0.99),
        rp_lr                   = ckpt_args.get("rp_lr", 3e-5),
        rp_chunks_per_agent     = chunks_per_agent,
        rp_entropy_coef         = ckpt_args.get("rp_entropy_coef", 0.01),
    )

    print("Loading model...")
    mw = ModelWrapper(model_name=MODEL_PATH, device=torch.device("cuda"),
                      use_vllm=False, args=latent_args)
    for p_ in mw.model.parameters():
        p_.requires_grad_(False)

    method = LatentMASMethod(
        model=mw, latent_steps=latent_steps, judger_max_new_tokens=512,
        temperature=0.7, top_p=0.95, do_sample=False,
        generate_bs=args.batch_size, args=latent_args,
    )

    rp: ReadingPolicy = method.reading_policy
    rp.build_gate_nets_from_state_dict(ckpt["reading_policy"])
    rp.load_state_dict(ckpt["reading_policy"])
    rp.eval()
    print("ReadingPolicy weights restored.")

    # ------------------------------------------------------------------
    # Collection loop — no Judger, only forward pass + gate computation
    # ------------------------------------------------------------------
    gate_counts  = np.zeros(N_GATES, dtype=np.int64)   # selected count per gate
    logit_sums   = np.zeros(N_GATES, dtype=np.float64) # raw logit accumulator
    n_samples    = 0
    all_patterns = []   # list of [N_GATES] arrays, one per sample

    for step, batch in enumerate(batches, start=1):
        with torch.no_grad():
            state = method.collect_pre_judger_state(batch)

        with torch.no_grad():
            _, logits = rp.compute_gates(
                state.query, state.block_summaries, training=False
            )   # [B, N_GATES]

        gates = topk_gates(logits, args.top_k)   # [B, N_GATES] binary

        B = gates.shape[0]
        gate_counts  += gates.cpu().numpy().sum(axis=0).astype(np.int64)
        logit_sums   += logits.cpu().numpy().sum(axis=0)
        n_samples    += B

        for b in range(B):
            all_patterns.append(gates[b].cpu().numpy().astype(int).tolist())

        if step % 10 == 0:
            print(f"  step {step}/{len(batches)}  samples={n_samples}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    freq = gate_counts / n_samples
    avg_logit = logit_sums / n_samples

    print(f"\n{'='*60}")
    print(f"GATE SELECTION FREQUENCY  (top_k={args.top_k}, N={n_samples} samples)")
    print(f"  checkpoint: step {ckpt_step}")
    print(f"  indices: {args.offset}:{args.offset + args.num_samples}")
    print(f"{'='*60}")
    print(f"\n  {'Gate':<6} {'Agent':<8} {'Chunk':<8} {'Selected':>10} {'Freq':>8} {'AvgLogit':>10}")
    print(f"  {'-'*52}")
    for i, name in enumerate(GATE_NAMES):
        agent = ["P","C","R"][i // chunks_per_agent]
        chunk = (i % chunks_per_agent) + 1
        print(f"  {name:<6} {agent:<8} {chunk:<8} "
              f"{gate_counts[i]:>10d} {freq[i]:>8.3f} {avg_logit[i]:>10.4f}")

    print(f"\n  Total gates selected per sample: {freq.sum():.2f}  (target={args.top_k})")

    # Per-agent summary
    print(f"\n  {'Agent':<8} {'Chunks':<30} {'Freq (mean)':>12}")
    print(f"  {'-'*52}")
    agent_labels = ["P (Planner)", "C (Critic)", "R (Refiner)"]
    for a, label in enumerate(agent_labels):
        idx = slice(a * chunks_per_agent, (a + 1) * chunks_per_agent)
        chunk_freqs = freq[idx]
        print(f"  {label:<12} {str([f'{x:.3f}' for x in chunk_freqs]):<32} "
              f"{chunk_freqs.mean():>10.3f}")

    print(f"\n  Global mean freq per gate: {freq.mean():.3f}  "
          f"(expected={args.top_k/N_GATES:.3f})")
    print(f"{'='*60}")

    # Write results to file
    out_path = os.path.join(os.path.dirname(args.checkpoint), "gate_freq_analysis.txt")
    with open(out_path, "w") as f:
        f.write(f"GATE SELECTION FREQUENCY  top_k={args.top_k}  N={n_samples}\n")
        f.write(f"checkpoint step {ckpt_step}  indices {args.offset}:{args.offset+args.num_samples}\n\n")
        f.write(f"{'Gate':<6} {'Freq':>8} {'AvgLogit':>10}\n")
        for i, name in enumerate(GATE_NAMES):
            f.write(f"{name:<6} {freq[i]:>8.3f} {avg_logit[i]:>10.4f}\n")
    print(f"\nResults written → {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
