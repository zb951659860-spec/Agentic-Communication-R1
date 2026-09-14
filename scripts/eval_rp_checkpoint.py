"""Evaluate a ReadingPolicy checkpoint using top-k gate inference.

Usage:
    python3 scripts/eval_rp_checkpoint.py \
        --checkpoint /tmp/rp_test_run/rp_training/checkpoint_step50.pt \
        --num_samples 100 \
        --gate_inference topk \
        --top_k 6 \
        --judger_max_new_tokens 512

Gate inference modes:
    topk        : keep the top-k gates by logit value (hard binary mask)
    threshold   : σ(logit) > 0.5  (default eval behaviour of ReadingPolicy)
    full_latent : forced all-ones  (ceiling baseline)
    no_latent   : forced all-zeros (floor baseline)
"""
import argparse
import logging
import os
import sys

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_rp")

MODEL_PATH  = "/root/Agentic-Communication-single_model-main/models/Qwen3-4B"
TRAIN_FILE  = os.path.join(ROOT, "data/gsm8k/train.parquet")
N_GATES     = 12
BATCH_SIZE  = 2


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _check_gsm8k_last_number(pred: str, gold: str) -> float:
    import re
    nums = re.findall(r"[-+]?\d*\.?\d+", gold.replace(",", ""))
    gold_num = nums[-1] if nums else None
    pred_nums = re.findall(r"[-+]?\d*\.?\d+", pred.replace(",", ""))
    pred_num  = pred_nums[-1] if pred_nums else None
    return 1.0 if (gold_num is not None and pred_num == gold_num) else 0.0


def _extract_boxed(text: str):
    import re
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    return m[-1].strip() if m else None

def _normalize(s) -> str:
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace(" ", "")
    try:
        return str(float(s))
    except ValueError:
        return s.lower()

def _check_gsm8k_boxed(pred: str, gold: str) -> float:
    import re
    m = re.search(r"####\s*(\S+)", gold)
    gv = _normalize(m.group(1) if m else gold.strip())
    extracted = _extract_boxed(pred)
    if extracted is None:
        m2 = re.search(r"####\s*(\S+)", pred)
        if m2:
            extracted = m2.group(1)
    return 1.0 if (_normalize(extracted) is not None and _normalize(extracted) == gv) else 0.0


def make_grader(name: str):
    if name == "boxed":
        return _check_gsm8k_boxed
    elif name == "last_number":
        return _check_gsm8k_last_number
    else:
        raise ValueError(f"Unknown grader: {name!r}. Choose 'boxed' or 'last_number'.")


# keep legacy name so any callers don't break
def check_gsm8k(pred: str, gold: str) -> float:
    return _check_gsm8k_last_number(pred, gold)


def topk_gates(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Return a binary [B, N_gates] mask keeping the top-k gates per row."""
    B, N = logits.shape
    gates = torch.zeros(B, N, dtype=torch.float32, device=logits.device)
    topk_idx = logits.topk(k, dim=-1).indices   # [B, k]
    gates.scatter_(-1, topk_idx, 1.0)
    return gates


def build_forced_gates(mode: str, logits: torch.Tensor, k: int) -> torch.Tensor:
    B = logits.shape[0]
    if mode == "topk":
        return topk_gates(logits, k)
    elif mode == "threshold":
        return (torch.sigmoid(logits) > 0.5).float()
    elif mode == "full_latent":
        return torch.ones(B, N_GATES, dtype=torch.float32)
    elif mode == "no_latent":
        return torch.zeros(B, N_GATES, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown gate_inference mode: {mode!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",           required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--train_file",           default=TRAIN_FILE)
    p.add_argument("--num_samples",          type=int, default=100,
                   help="Number of dataset samples to evaluate")
    p.add_argument("--offset",              type=int, default=0,
                   help="Skip the first N dataset rows (e.g. 100 for held-out eval on 100:200).")
    p.add_argument("--gate_inference",       default="topk",
                   choices=["topk", "threshold", "full_latent", "no_latent"],
                   help="Gate selection strategy at evaluation")
    p.add_argument("--top_k",               type=int, default=6,
                   help="k for topk mode (number of gates to keep open)")
    p.add_argument("--judger_max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size",           type=int, default=BATCH_SIZE)
    p.add_argument("--output_dir",           default=None,
                   help="Write result log here (defaults to checkpoint dir)")
    p.add_argument("--grader",               default="last_number",
                   choices=["last_number", "boxed"],
                   help="Grader used to score predictions. "
                        "'last_number': last number in text (permissive). "
                        "'boxed': \\boxed{} then #### fallback (strict, matches training reward).")
    return p.parse_args()


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Load checkpoint metadata
    # -----------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]
    ckpt_step = ckpt.get("step", "?")
    logger.info(f"Loaded checkpoint: step={ckpt_step}  path={args.checkpoint}")

    latent_steps      = ckpt_args["latent_steps"]
    chunks_per_agent  = ckpt_args["rp_chunks_per_agent"]
    chunk_size        = latent_steps // chunks_per_agent
    logger.info(f"Config: latent_steps={latent_steps}  chunks_per_agent={chunks_per_agent}  "
                f"chunk_size={chunk_size}  n_gates={N_GATES}")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    df   = pd.read_parquet(args.train_file)
    rows = df.iloc[args.offset : args.offset + args.num_samples].to_dict("records")
    logger.info(f"Evaluating on {len(rows)} samples "
                f"(indices {args.offset}:{args.offset + args.num_samples})")

    # Build batches
    batches = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        batches.append([
            {"question": r["problem"], "gold": str(r["answer"])} for r in chunk
        ])

    # -----------------------------------------------------------------------
    # Build LatentMASMethod  (same init path as training script)
    # -----------------------------------------------------------------------
    import argparse as _ap
    from agent_system.multi_turn_rollout.models import ModelWrapper
    from agent_system.multi_turn_rollout.methods.latent_mas import LatentMASMethod
    from agent_system.multi_turn_rollout.reading_policy import ReadingPolicy

    latent_args = _ap.Namespace(
        model_name              = MODEL_PATH,
        device                  = "cuda",
        device2                 = "cuda",
        latent_steps            = latent_steps,
        max_new_tokens          = args.judger_max_new_tokens,
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

    logger.info(f"Loading model: {MODEL_PATH}")
    model_wrapper = ModelWrapper(
        model_name=MODEL_PATH,
        device=torch.device("cuda"),
        use_vllm=False,
        args=latent_args,
    )
    for p_ in model_wrapper.model.parameters():
        p_.requires_grad_(False)

    method = LatentMASMethod(
        model=model_wrapper,
        latent_steps=latent_steps,
        judger_max_new_tokens=args.judger_max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        do_sample=False,
        generate_bs=args.batch_size,
        args=latent_args,
    )
    assert method.reading_policy is not None

    rp: ReadingPolicy = method.reading_policy

    # -----------------------------------------------------------------------
    # Restore checkpoint weights
    # -----------------------------------------------------------------------
    rp.build_gate_nets_from_state_dict(ckpt["reading_policy"])
    rp.load_state_dict(ckpt["reading_policy"])
    rp.eval()
    logger.info("ReadingPolicy weights restored from checkpoint")
    grader = make_grader(args.grader)
    logger.info(f"Grader: {args.grader}")

    # -----------------------------------------------------------------------
    # Print gate net final biases (quick sanity check)
    # -----------------------------------------------------------------------
    biases = [rp.gate_nets[i].layers[-1].bias.item() for i in range(N_GATES)]
    logger.info(f"Gate biases: {[round(b,5) for b in biases]}")

    # -----------------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------------
    all_rewards  = []
    all_gates    = []
    n_steps_done = 0

    for step, batch in enumerate(batches, start=1):
        with torch.no_grad():
            state = method.collect_pre_judger_state(batch)

        # Compute GateNet logits (no grad needed for eval)
        with torch.no_grad():
            _, logits = rp.compute_gates(
                state.query, state.block_summaries, training=False
            )  # logits: [B, N_gates]

        # Apply gate selection strategy
        gates = build_forced_gates(args.gate_inference, logits, args.top_k)  # [B, N_gates]
        gate_mean = gates.mean().item()

        texts, gates_out, _ = method.run_judger_with_new_gates(
            state, training=False, forced_gates=gates
        )

        rewards = [
            grader(text, item["gold"])
            for text, item in zip(texts, batch)
        ]

        batch_reward = sum(rewards) / len(rewards)
        all_rewards.extend(rewards)
        all_gates.append(gate_mean)
        n_steps_done += 1

        gate_pattern = [int(g) for g in gates[0].tolist()]
        logger.info(
            f"[step {step:>4}]  reward={batch_reward:.3f}  gate={gate_mean:.3f}  "
            f"rewards={rewards}  pattern={gate_pattern}"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    avg_reward  = sum(all_rewards) / len(all_rewards)
    hit_rate    = sum(1 for r in all_rewards if r > 0.0) / len(all_rewards) * 100.0
    avg_gate    = sum(all_gates) / len(all_gates)
    active_avg  = avg_gate * N_GATES

    summary = (
        f"\n{'='*72}\n"
        f"EVAL SUMMARY\n"
        f"  checkpoint     : {args.checkpoint}\n"
        f"  step           : {ckpt_step}\n"
        f"  gate_inference : {args.gate_inference}"
        + (f"  (top_k={args.top_k})" if args.gate_inference == "topk" else "") + "\n"
        f"  eval_indices   : {args.offset}:{args.offset + args.num_samples}\n"
        f"  n_samples      : {len(all_rewards)}\n"
        f"  grader         : {args.grader}\n"
        f"  avg_reward     : {avg_reward:.4f}\n"
        f"  hit_rate       : {hit_rate:.1f}%\n"
        f"  avg_gate_ratio : {avg_gate:.3f}  ({active_avg:.1f}/{N_GATES} active)\n"
        f"{'='*72}"
    )
    logger.info(summary)

    # -----------------------------------------------------------------------
    # Write log file
    # -----------------------------------------------------------------------
    out_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"step{ckpt_step}_{args.gate_inference}"
    if args.gate_inference == "topk":
        tag += f"_k{args.top_k}"
    if args.grader != "last_number":
        tag += f"_{args.grader}"
    if args.offset:
        tag += f"_off{args.offset}"
    log_path = os.path.join(out_dir, f"eval_{tag}.log")
    with open(log_path, "w") as f:
        f.write(summary + "\n")
    logger.info(f"Log written → {log_path}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
