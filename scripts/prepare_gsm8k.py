#!/usr/bin/env python3
"""
Download GSM8K from HuggingFace and save train / test splits as Parquet.

Output (default --output_dir data/gsm8k):
    data/gsm8k/train.parquet   (7,473 rows)
    data/gsm8k/test.parquet    (1,319 rows)

Columns written:
    problem  — the question text   (maps to --question_col default "problem")
    answer   — the solution text   (maps to --answer_col   default "answer")

The answer field retains the full GSM8K format that ends with  "#### <number>",
which is what train_reading_policy.py's _check_gsm8k() extractor expects.

Usage:
    python scripts/prepare_gsm8k.py
    python scripts/prepare_gsm8k.py --output_dir /some/other/path
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading GSM8K from HuggingFace (main config)…")
    ds = load_dataset("gsm8k", "main", trust_remote_code=True)

    for split_name, hf_key in [("train", "train"), ("test", "test")]:
        split = ds[hf_key]
        df = pd.DataFrame({
            "problem": split["question"],   # renamed to match --question_col default
            "answer":  split["answer"],
        })
        out_path = out_dir / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {split_name}: {len(df):,} rows  →  {out_path}")

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare GSM8K dataset for SELMA RP training")
    p.add_argument("--output_dir", default="data/gsm8k",
                   help="Directory to write train.parquet and test.parquet")
    main(p.parse_args())
