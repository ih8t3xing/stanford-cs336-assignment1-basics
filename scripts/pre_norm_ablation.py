"""
Pre-norm vs Post-norm ablation.

Runs:
  1. Pre-norm  at the optimal LR (baseline)
  2. Post-norm at the optimal LR
  3. Post-norm at lower LRs to find a stable setting (post-norm is known to be
     harder to train at high LRs)

Usage:
    python scripts/pre_norm_ablation.py \
        --train_data data/tinystories/TinyStoriesV2-GPT4-train.txt \
        --val_data data/tinystories/TinyStoriesV2-GPT4-valid.txt \
        --vocab_filepath output/tinystories_bpe/vocab.json \
        --merges_filepath output/tinystories_bpe/merges.json \
        --optimal_lr 3e-3
"""

import argparse
import subprocess


# Lower LRs to try for post-norm if the optimal LR diverges
POST_NORM_LOWER_LRS = [1e-3, 3e-4, 1e-4]

SWEEP_ITERS = 5000
SWEEP_VAL_INTERVAL = 500
SWEEP_LOG_INTERVAL = 100

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


def run_experiment(lr, tag, args, post_norm=False):
    lr_min = lr / 10
    run_name = f"pre_norm_ablation_{tag}_lr{lr:.0e}"
    checkpoint_dir = f"output/checkpoints/pre_norm_ablation/{tag}_lr{lr:.0e}"
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
        "--checkpoint_interval", str(SWEEP_ITERS),
        "--checkpoint_dir", checkpoint_dir,
        "--wandb_project", args.wandb_project,
        "--wandb_run_name", run_name,
    ] + MODEL_DEFAULTS

    if post_norm:
        extra.append("--post_norm")

    cmd = build_cmd(extra)
    print(f"\n[{tag}] lr={lr:.0e}  cmd:\n  {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Pre-norm vs Post-norm ablation")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--vocab_filepath", required=True)
    parser.add_argument("--merges_filepath", required=True)
    parser.add_argument("--data_cache_dir", default="/tmp/ts_cache")
    parser.add_argument("--optimal_lr", type=float, default=3e-3,
                        help="Previously optimal LR (with pre-norm)")
    parser.add_argument("--wandb_project", default="cs336-pre-norm-ablation")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # 1. Pre-norm baseline at optimal LR
    print("=== Run 1: Pre-norm (baseline) at optimal LR ===")
    run_experiment(args.optimal_lr, "pre_norm", args, post_norm=False)

    # 2. Post-norm at the same optimal LR — may diverge
    print("\n=== Run 2: Post-norm at optimal LR ===")
    run_experiment(args.optimal_lr, "post_norm_optimal", args, post_norm=True)

    # 3. Post-norm at lower LRs — search for a stable setting
    print("\n=== Run 3: Post-norm at lower LRs ===")
    for lr in POST_NORM_LOWER_LRS:
        run_experiment(lr, "post_norm_lower", args, post_norm=True)

    print("\nAblation complete. Compare runs in W&B under project:", args.wandb_project)


if __name__ == "__main__":
    main()
