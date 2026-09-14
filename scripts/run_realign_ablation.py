#!/usr/bin/env python3
"""
Realignment ablation: 4 × 50-step bypass experiments with Qwen3-4B.

  A  latent_steps=0,  realign=False   — no latent at all (prompt-KV only)
  B  latent_steps=1,  realign=False   — 1 latent step, no norm-rescaling
  C  latent_steps=1,  realign=True    — 1 latent step, norm-rescaling (identity+norm)
  D  latent_steps=16, realign=True    — 16 latent steps, norm-rescaling

All: gate_mode=bypass, batch_size=2, steps=50, greedy decoding.
Logs written to /tmp/rp_test_run/realign_ablation/.
"""

import argparse
import os
import sys
import logging

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Re-use helpers from diagnostic_experiments
from scripts.diagnostic_experiments import run_experiment, load_dataset

OUTPUT_DIR = "/tmp/rp_test_run/realign_ablation"
MODEL_PATH = "/root/Agentic-Communication-single_model-main/models/Qwen3-4B"
TRAIN_FILE = "data/gsm8k/train.parquet"
N_STEPS    = 50
BATCH_SIZE = 2
SEED       = 42

# Experiment table: (name, latent_steps, chunks_per_agent, realign, realign_is_identity)
EXPERIMENTS = [
    ("expA_latent0_norealign",   0,  4, False, False),
    ("expB_latent4_norealign",   4,  4, False, False),
    ("expC_latent4_realign",     4,  4, True,  True),
    ("expD_latent16_realign",   16,  4, True,  True),
    ("expE_latent16_norealign", 16,  4, False, False),
]


def build_latent_args(model_path, latent_steps, chunks_per_agent,
                      latent_space_realign, realign_matrix_is_identity):
    import argparse as _ap
    return _ap.Namespace(
        model_name                 = model_path,
        device                     = "cuda",
        device2                    = "cuda",
        latent_steps               = latent_steps,
        max_new_tokens             = 512,
        temperature                = 0.7,
        top_p                      = 0.95,
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
        use_reading_policy         = True,
        rp_temperature             = 1.0,
        rp_hard_threshold          = 0.5,
        rp_sparsity_lambda         = 0.05,
        rp_ema_momentum            = 0.99,
        rp_lr                      = 3e-5,
        rp_chunks_per_agent        = chunks_per_agent,
        rp_entropy_coef            = 0.01,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    logger = logging.getLogger("realign_ablation")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from agent_system.multi_turn_rollout.models import ModelWrapper

    logger.info(f"Loading model: {MODEL_PATH}")
    logger.info(f"Output dir:    {OUTPUT_DIR}")

    train_data = load_dataset(
        os.path.join(ROOT, TRAIN_FILE)
        if not os.path.isabs(TRAIN_FILE) else TRAIN_FILE
    )
    logger.info(f"Loaded {len(train_data)} train examples")

    summary = []

    for exp_name, latent_steps, chunks_per_agent, realign, realign_identity in EXPERIMENTS:
        logger.info(f"\n{'='*65}")
        logger.info(f"  {exp_name}  latent_steps={latent_steps}  "
                    f"realign={realign}  identity={realign_identity}  "
                    f"chunks_per_agent={chunks_per_agent}")
        logger.info(f"{'='*65}")

        latent_args = build_latent_args(
            MODEL_PATH, latent_steps, chunks_per_agent, realign, realign_identity
        )

        # Fresh ModelWrapper per experiment — ensures realign matrix is built
        # with the correct args for this experiment
        model_wrapper = ModelWrapper(
            model_name=MODEL_PATH,
            device=torch.device("cuda"),
            use_vllm=False,
            args=latent_args,
        )
        for param in model_wrapper.model.parameters():
            param.requires_grad_(False)

        from agent_system.multi_turn_rollout.methods.latent_mas import LatentMASMethod
        method = LatentMASMethod(
            model=model_wrapper,
            latent_steps=latent_steps,
            judger_max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=False,
            generate_bs=BATCH_SIZE,
            args=latent_args,
        )
        assert method.reading_policy is not None

        log_path = os.path.join(OUTPUT_DIR, f"{exp_name}.log")
        avg_r, hit = run_experiment(
            method=method,
            train_data=train_data,
            exp_name=exp_name,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            gate_mode="bypass",
            log_path=log_path,
            seed=SEED,
        )
        summary.append((exp_name, latent_steps, realign, avg_r, hit))

        # Release GPU memory before next experiment
        del method
        del model_wrapper
        torch.cuda.empty_cache()

    logger.info("\n" + "="*65)
    logger.info("REALIGN ABLATION SUMMARY")
    logger.info(f"  {'Experiment':<32} {'latent':>7} {'realign':>8}  {'avg_r':>7}  {'hit%':>7}")
    logger.info("  " + "-"*62)
    for exp_name, ls, r, avg_r, hit in summary:
        logger.info(f"  {exp_name:<32} {ls:>7} {str(r):>8}  {avg_r:>7.4f}  {hit*100:>6.1f}%")
    logger.info("="*65)


if __name__ == "__main__":
    main()
