"""
RMSNorm ablation sweep.

Runs two experiments:
  1. No RMSNorm at the previously-optimal LR (default 1e-3)
  2. No RMSNorm at lower LRs to find a stable setting

Usage:
    python scripts/norm_ablation.py \
        --train_data data/tinystories/TinyStoriesV2-GPT4-train.txt \
        --val_data data/tinystories/TinyStoriesV2-GPT4-valid.txt \
        --vocab_filepath output/tinystories_bpe/vocab.json \
        --merges_filepath output/tinystories_bpe/merges.json \
        --optimal_lr 3e-3
"""

import argparse
import subprocess


# LRs to try for the no-norm ablation: optimal + lower candidates
ABLATION_LRS_LOWER = [3e-4, 1e-4, 3e-5]

SWEEP_ITERS = 2000
SWEEP_VAL_INTERVAL = 200
SWEEP_LOG_INTERVAL = 50
SWEEP_CKPT_INTERVAL = 1000

MODEL_DEFAULTS = [
    "--vocab_size", "10000",
    "--context_length", "256",
    "--d_model", "512",
    "--num_layers", "4",
    "--num_heads", "16",
    "--d_ff", "1344",
    "--batch_size", "64",
    "--weight_decay", "0.1",
    "--grad_clip", "1.0",
]


def build_cmd(extra_args):
    return ["uv", "run", "python", "-m", "cs336_basics.training_together"] + extra_args


def run_experiment(lr, tag, args, use_rmsnorm=False):
    lr_min = lr / 10
    run_name = f"norm_ablation_{tag}_lr{lr:.0e}"
    checkpoint_dir = f"output/checkpoints/norm_ablation/{tag}_lr{lr:.0e}"
    extra = [
        "--train_data", args.train_data,
        "--val_data", args.val_data,
        "--vocab_filepath", args.vocab_filepath,
        "--merges_filepath", args.merges_filepath,
        "--data_cache_dir", args.data_cache_dir,
        "--lr_max", str(lr),
        "--lr_min", str(lr_min),
        "--max_iters", str(SWEEP_ITERS),
        "--warmup_iters", "200",
        "--val_interval", str(SWEEP_VAL_INTERVAL),
        "--log_interval", str(SWEEP_LOG_INTERVAL),
        "--checkpoint_interval", str(SWEEP_CKPT_INTERVAL),
        "--checkpoint_dir", checkpoint_dir,
        "--wandb_project", args.wandb_project,
        "--wandb_run_name", run_name,
    ] + MODEL_DEFAULTS

    if not use_rmsnorm:
        extra.append("--no_rmsnorm")

    cmd = build_cmd(extra)
    print(f"\n[{tag}] lr={lr:.0e}  cmd:\n  {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="RMSNorm ablation sweep")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--vocab_filepath", required=True)
    parser.add_argument("--merges_filepath", required=True)
    parser.add_argument("--data_cache_dir", default="/tmp/ts_cache")
    parser.add_argument("--optimal_lr", type=float, default=1e-3,
                        help="Previously optimal LR (with RMSNorm)")
    parser.add_argument("--wandb_project", default="cs336-norm-ablation")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # 1. No-norm at the optimal LR — expect instability
    print("=== Run 1: No RMSNorm at optimal LR ===")
    run_experiment(args.optimal_lr, "no_norm_optimal", args, use_rmsnorm=False)

    # 2. No-norm at lower LRs — search for stability
    print("\n=== Run 2: No RMSNorm at lower LRs ===")
    for lr in ABLATION_LRS_LOWER:
        run_experiment(lr, "no_norm_lower", args, use_rmsnorm=False)

    print("\nAblation complete. Compare runs in W&B under project:", args.wandb_project)


if __name__ == "__main__":
    main()
