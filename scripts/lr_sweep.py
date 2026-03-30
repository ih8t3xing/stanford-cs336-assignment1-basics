"""
Learning rate hyperparameter sweep for TinyStories TransformerLM.

Strategy:
  Phase 1 - short runs (5k steps) over a log-scale LR grid to identify best region.
  Phase 2 - full run (100k steps) with the best LR to hit val loss <= 1.45.

Usage (Phase 1 sweep):
    python scripts/lr_sweep.py \
        --train_data data/tinystories/TinyStoriesV2-GPT4-train.txt \
        --val_data data/tinystories/TinyStoriesV2-GPT4-valid.txt \
        --vocab_filepath output/tinystories_bpe/vocab.json \
        --merges_filepath output/tinystories_bpe/merges.json \
        --data_cache_dir /tmp/ts_cache \
        --phase sweep

Usage (Phase 2 full run with best LR):
    python scripts/lr_sweep.py ... --phase full --best_lr 1e-3
"""

import argparse
import subprocess


# Log-scale LR grid: 1e-4 to 1e-2
# SWEEP_LRS = [1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 1e-2]
SWEEP_LRS = [3e-2, 1e-1, 3e-1, 1.0]  # Testing higher LRs to see divergence behavior

# Short run to compare learning curves quickly
SWEEP_ITERS = 5000
SWEEP_VAL_INTERVAL = 500
SWEEP_LOG_INTERVAL = 100

# Full run config to hit val loss <= 1.45
FULL_ITERS = 100_000
FULL_VAL_INTERVAL = 1000
FULL_LOG_INTERVAL = 200

# Model config (~25M param model)
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


def build_cmd(extra_args: list[str]) -> list[str]:
    return ["uv", "run", "python", "-m", "cs336_basics.training_together"] + extra_args


def sweep(args):
    """Run all LR sweeps sequentially on a single GPU."""
    for i, lr in enumerate(SWEEP_LRS):
        lr_min = lr / 10
        run_name = f"lr_sweep_lr{lr:.0e}"
        checkpoint_dir = f"output/checkpoints/lr_sweep/lr{lr:.0e}"
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

        cmd = build_cmd(extra)
        print(f"\n[{i+1}/{len(SWEEP_LRS)}] lr={lr:.0e}  cmd: {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    if not args.dry_run:
        print("Sweep complete.")


def full(args):
    assert args.best_lr is not None, "Provide --best_lr for full run"
    lr = args.best_lr
    lr_min = lr / 10
    run_name = f"lr_full_lr{lr:.0e}"
    checkpoint_dir = f"output/checkpoints/lr_full/lr{lr:.0e}"
    extra = [
        "--train_data", args.train_data,
        "--val_data", args.val_data,
        "--vocab_filepath", args.vocab_filepath,
        "--merges_filepath", args.merges_filepath,
        "--data_cache_dir", args.data_cache_dir,
        "--lr_max", str(lr),
        "--lr_min", str(lr_min),
        "--max_iters", str(FULL_ITERS),
        "--warmup_iters", "500",
        "--val_interval", str(FULL_VAL_INTERVAL),
        "--log_interval", str(FULL_LOG_INTERVAL),
        "--checkpoint_interval", "10000",
        "--checkpoint_dir", checkpoint_dir,
        "--wandb_project", args.wandb_project,
        "--wandb_run_name", run_name,
    ] + MODEL_DEFAULTS
    cmd = build_cmd(extra)
    print("Running:", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="LR sweep for TinyStories TransformerLM")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--vocab_filepath", required=True)
    parser.add_argument("--merges_filepath", required=True)
    parser.add_argument("--data_cache_dir", default="/tmp/ts_cache")
    parser.add_argument("--phase", choices=["sweep", "full"], default="sweep",
                        help="'sweep': short multi-LR runs; 'full': one long run with --best_lr")
    parser.add_argument("--best_lr", type=float, default=None,
                        help="LR to use for the full run (required when --phase full)")
    parser.add_argument("--wandb_project", type=str, default="cs336-lm-lr-sweep")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    if args.phase == "sweep":
        print(f"Phase 1: sweeping LRs {SWEEP_LRS} for {SWEEP_ITERS} steps each")
        sweep(args)
    else:
        print(f"Phase 2: full run ({FULL_ITERS} steps) with lr={args.best_lr}")
        full(args)


if __name__ == "__main__":
    main()
