"""
Quick verification: measure truncation rate and accuracy for
judger_max_new_tokens = 512 vs 1024 on N GSM8K samples.

Usage:
    python scripts/verify_token_limits.py --n_samples 150
"""
import argparse
import re
import random
import sys
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_boxed(text: str):
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return m.group(1).strip() if m else None

def _normalize(s):
    if s is None:
        return None
    s = s.replace(",", "").strip()
    try:
        return str(float(s))
    except ValueError:
        return s.lower()

def _extract_gold(gold: str):
    m = re.search(r"####\s*(\S+)", gold)
    return _normalize(m.group(1) if m else gold.strip())

def is_correct(pred_text: str, gold: str) -> bool:
    pred = _normalize(_extract_boxed(pred_text))
    return pred is not None and pred == _extract_gold(gold)

def is_truncated(pred_text: str, n_generated: int, max_new_tokens: int) -> bool:
    """Truncated = hit token limit AND no boxed answer found."""
    return n_generated >= max_new_tokens and _extract_boxed(pred_text) is None

JUDGER_PROMPT = (
    "You are a helpful assistant. You are provided with latent information for reference "
    "and a target question to solve.\n\n"
    "The latent information might contain irrelevant contents. Ignore it if it is not "
    "helpful for solving the target question.\n\n"
    "You must reason step-by-step to solve the provided Target Question without outputting "
    "other irrelevant information.\n\n"
    "Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}."
)

def build_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user",   "content": f"Target Question: {question}\n\n{JUDGER_PROMPT}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ── generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_batch(model, tokenizer, prompts, max_new_tokens, device, batch_size=8):
    results = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024).to(device)
        input_len = enc["input_ids"].shape[1]
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
        for i, seq in enumerate(out):
            gen_ids   = seq[input_len:]
            n_gen     = int((gen_ids != tokenizer.pad_token_id).sum())
            gen_text  = tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append((gen_text, n_gen))
        done = min(start + batch_size, len(prompts))
        print(f"  generated {done}/{len(prompts)} ...", flush=True)
    return results

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default="/root/Agentic-Communication-single_model-main/models/Qwen2.5-3B-Instruct")
    ap.add_argument("--val_file",   default="data/gsm8k/test.parquet")
    ap.add_argument("--n_samples",  type=int, default=150)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df = pd.read_parquet(args.val_file)
    q_col = next(c for c in df.columns if "question" in c.lower() or "problem" in c.lower())
    a_col = next(c for c in df.columns if "answer"   in c.lower() or "solution" in c.lower())
    samples = df[[q_col, a_col]].sample(args.n_samples, random_state=args.seed).to_dict("records")
    questions = [s[q_col] for s in samples]
    golds     = [s[a_col] for s in samples]
    print(f"Loaded {len(samples)} samples from {args.val_file}")

    # Load model
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print("Model loaded.")

    prompts = [build_prompt(q, tokenizer) for q in questions]

    for max_tok in (512,):
        print(f"\n{'='*60}")
        print(f" max_new_tokens = {max_tok}")
        print(f"{'='*60}")
        results = run_batch(model, tokenizer, prompts, max_tok, device, args.batch_size)

        n_correct   = 0
        n_truncated = 0
        n_no_box    = 0
        for (text, n_gen), gold in zip(results, golds):
            if is_correct(text, gold):
                n_correct += 1
            if is_truncated(text, n_gen, max_tok):
                n_truncated += 1
            if _extract_boxed(text) is None:
                n_no_box += 1

        N = len(samples)
        print(f"  Accuracy      : {n_correct}/{N} = {n_correct/N*100:.1f}%")
        print(f"  Truncated     : {n_truncated}/{N} = {n_truncated/N*100:.1f}%  (hit limit + no \\boxed{{}})")
        print(f"  No \\boxed{{}}  : {n_no_box}/{N} = {n_no_box/N*100:.1f}%")
        print(f"  Has \\boxed{{}} : {N-n_no_box}/{N} = {(N-n_no_box)/N*100:.1f}%")

if __name__ == "__main__":
    main()
