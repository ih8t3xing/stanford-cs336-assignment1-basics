"""
Batch size experiment for TinyStories TransformerLM.

Varies batch size from 1 to GPU memory limit. LR is scaled linearly with batch
size (linear scaling rule). Total tokens seen is held constant across all runs
so learning curves are on a fair compute budget.

Usage:
    python scripts/batch_size_sweep.py \
        --train_data data/tinystories/TinyStoriesV2-GPT4-train.txt \
        --val_data data/tinystories/TinyStoriesV2-GPT4-valid.txt \
        --vocab_filepath output/tinystories_bpe/vocab.json \
        --merges_filepath output/tinystories_bpe/merges.json \
        --best_lr 1e-3 \
        --data_cache_dir /tmp/ts_cache

Dry run (print commands only):
    python scripts/batch_size_sweep.py ... --dry_run
"""

import argparse
import subprocess

# Batch sizes to sweep: from 1 up to H100 memory limit (~1024 fits for this model)
# BATCH_SIZES = [1, 4, 16, 64, 128]
BATCH_SIZES = [128, 64, 16, 4, 1]

# Base config that the best_lr was tuned for
BASE_BATCH_SIZE = 64
CONTEXT_LENGTH = 256

# Total tokens seen by the base run: 64 * 256 * 5000 = ~82M tokens
# All runs will see the same total tokens for a fair comparison.
TOTAL_TOKENS = BASE_BATCH_SIZE * CONTEXT_LENGTH * 5000

VAL_INTERVAL_TOKENS = BASE_BATCH_SIZE * CONTEXT_LENGTH * 500  # eval every ~8M tokens
LOG_INTERVAL_TOKENS = BASE_BATCH_SIZE * CONTEXT_LENGTH * 100

MODEL_DEFAULTS = [
    "--vocab_size", "10000",
    "--context_length", str(CONTEXT_LENGTH),
    "--d_model", "512",
    "--num_layers", "4",
    "--num_heads", "16",
    "--d_ff", "1344",
    "--weight_decay", "0.1",
    "--grad_clip", "1.0",
]


def build_cmd(extra_args: list[str]) -> list[str]:
    return ["uv", "run", "python", "-m", "cs336_basics.training_together"] + extra_args


def sweep(args):
    for i, bs in enumerate(BATCH_SIZES):
        # Linear LR scaling rule: lr ∝ batch_size
        lr = args.best_lr * (bs / BASE_BATCH_SIZE)
        lr_min = lr / 10

        # Fix total tokens seen across all runs for fair comparison.
        # Clamp so small batches don't run forever and large batches get enough steps.
        max_iters = TOTAL_TOKENS // (bs * CONTEXT_LENGTH)
        max_iters = min(max_iters, 10_000)   # bs=1 would be 320k steps — too slow
        max_iters = max(max_iters, 1_000)    # bs=1024 would be 312 steps — too few
        val_interval = max(10, max_iters // 10)
        log_interval = max(5, max_iters // 50)
        warmup_iters = max(50, min(200, max_iters // 20))

        run_name = f"bs_sweep_bs{bs}"
        checkpoint_dir = f"output/checkpoints/bs_sweep/bs{bs}"

        extra = [
            "--train_data", args.train_data,
            "--val_data", args.val_data,
            "--vocab_filepath", args.vocab_filepath,
            "--merges_filepath", args.merges_filepath,
            "--data_cache_dir", args.data_cache_dir,
            "--batch_size", str(bs),
            "--lr_max", str(lr),
            "--lr_min", str(lr_min),
            "--max_iters", str(max_iters),
            "--warmup_iters", str(warmup_iters),
            "--val_interval", str(val_interval),
            "--log_interval", str(log_interval),
            "--checkpoint_interval", str(max_iters),  # save only at end
            "--checkpoint_dir", checkpoint_dir,
            "--wandb_project", args.wandb_project,
            "--wandb_run_name", run_name,
        ] + MODEL_DEFAULTS

        cmd = build_cmd(extra)
        tokens_M = TOTAL_TOKENS / 1e6
        print(
            f"\n[{i+1}/{len(BATCH_SIZES)}] bs={bs:4d}  lr={lr:.2e}  "
            f"iters={max_iters:6d}  tokens={tokens_M:.1f}M"
        )
        print("  cmd:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    if not args.dry_run:
        print("\nBatch size sweep complete.")


def main():
    parser = argparse.ArgumentParser(description="Batch size sweep for TinyStories TransformerLM")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--vocab_filepath", required=True)
    parser.add_argument("--merges_filepath", required=True)
    parser.add_argument("--data_cache_dir", default="/tmp/ts_cache")
    parser.add_argument("--best_lr", type=float, required=True,
                        help="Best LR found for base_batch_size=64; scaled linearly for other sizes")
    parser.add_argument("--wandb_project", type=str, default="cs336-lm-bs-sweep")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    print(f"Sweeping batch sizes: {BATCH_SIZES}")
    print(f"Base LR: {args.best_lr:.2e} at batch_size={BASE_BATCH_SIZE}")
    print(f"Total tokens per run: {TOTAL_TOKENS/1e6:.1f}M (fixed)")
    sweep(args)


if __name__ == "__main__":
    main()
