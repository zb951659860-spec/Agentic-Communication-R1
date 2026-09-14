"""
Measure KV-cache token counts per agent segment (Planner, Critic, Refiner)
and Judger prompt length on a small GSM8K sample.

Usage:
    python scripts/measure_agent_tokens.py --n_samples 10
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd, torch, random
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default="/root/Agentic-Communication-single_model-main/models/Qwen2.5-3B-Instruct")
    ap.add_argument("--val_file",   default="data/gsm8k/test.parquet")
    ap.add_argument("--n_samples",  type=int, default=10)
    ap.add_argument("--latent_steps", type=int, default=16)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Load model & tokenizer (just for token counting, no generation needed)
    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    # Load data
    df = pd.read_parquet(args.val_file)
    q_col = next(c for c in df.columns if "question" in c.lower() or "problem" in c.lower())
    samples = df[q_col].sample(args.n_samples, random_state=args.seed).tolist()
    print(f"Loaded {len(samples)} samples.\n")

    # Build agent prompts using the same structure as latent_mas.py
    # We import the prompt builder directly
    from agent_system.multi_turn_rollout.prompts import build_agent_message_sequential_latent_mas

    # Minimal args namespace for prompt builder
    class MinArgs:
        task = "gsm8k"
        prompt_style = "sequential"
        think = False
        latent_only = False
        sequential_info_only = False
        use_reading_policy = True
        device = "cpu"
        device2 = "cpu"
        model_name = args.model
        latent_steps = args.latent_steps

    min_args = MinArgs()

    agent_roles = ["planner", "critic", "refiner", "judger"]

    # Collect token counts per sample
    records = []
    for i, question in enumerate(samples):
        item = {"question": question, "gold": "0"}
        row = {"sample": i}

        cumulative_context = ""
        for role in agent_roles:
            messages = build_agent_message_sequential_latent_mas(
                role=role, question=question, context=cumulative_context,
                method="latent_mas", args=min_args,
            )
            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_tokens = len(ids)

            if role != "judger":
                # Agent generates latent_steps tokens → those enter the KV cache too
                total_kv_tokens = prompt_tokens + args.latent_steps
                row[f"{role}_prompt_tokens"] = prompt_tokens
                row[f"{role}_generated_tokens"] = args.latent_steps
                row[f"{role}_kv_tokens"] = total_kv_tokens
                # Simulate context accumulation (each subsequent agent sees prior agents' text)
                cumulative_context += f"\n[{role.capitalize()} output tokens: {args.latent_steps}]"
            else:
                row["judger_prompt_tokens"] = prompt_tokens

        records.append(row)

    # Compute stats
    import statistics
    roles_to_report = ["planner", "critic", "refiner"]
    print("=" * 65)
    print(f"{'Segment':<12} {'Prompt':>8} {'Generated':>10} {'KV total':>10}")
    print("-" * 65)
    for role in roles_to_report:
        p = [r[f"{role}_prompt_tokens"] for r in records]
        g = [r[f"{role}_generated_tokens"] for r in records]
        k = [r[f"{role}_kv_tokens"] for r in records]
        print(f"{role.capitalize():<12} "
              f"{statistics.mean(p):>6.0f}±{statistics.stdev(p) if len(p)>1 else 0:.0f}  "
              f"{statistics.mean(g):>8.1f}      "
              f"{statistics.mean(k):>8.0f}±{statistics.stdev(k) if len(k)>1 else 0:.0f}")

    # Judger prompt
    jp = [r["judger_prompt_tokens"] for r in records]
    print(f"{'Judger (prompt)':<12} {statistics.mean(jp):>6.0f}±{statistics.stdev(jp) if len(jp)>1 else 0:.0f}  {'—':>8}      {'—':>8}")

    total_kv = [
        sum(r[f"{role}_kv_tokens"] for role in roles_to_report) for r in records
    ]
    print("-" * 65)
    print(f"{'Total KV':<12} {'—':>8}   {'—':>10}  {statistics.mean(total_kv):>8.0f}±{statistics.stdev(total_kv) if len(total_kv)>1 else 0:.0f}")
    print("=" * 65)

    print("\nPer-agent share of total KV cache:")
    for role in roles_to_report:
        k = statistics.mean([r[f"{role}_kv_tokens"] for r in records])
        share = k / statistics.mean(total_kv) * 100
        print(f"  {role.capitalize():<10}: {k:>6.0f} tokens  ({share:.1f}%)")

    print(f"\nLatent steps per agent: {args.latent_steps}")
    print(f"Note: each agent contributes prompt_tokens + {args.latent_steps} generated tokens to the KV cache.")
    print("The gate for each agent scales ALL tokens in its range (prompt + generated).")

if __name__ == "__main__":
    main()
