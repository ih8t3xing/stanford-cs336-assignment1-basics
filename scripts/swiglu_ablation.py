"""
SwiGLU vs. SiLU feed-forward ablation.

Runs:
  1. SwiGLU (baseline): FFN(x) = W2(SiLU(W1 x) * W3 x), d_ff = (8/3)*d_model ~ 1344
  2. SiLU (no gate):    FFN(x) = W2(SiLU(W1 x)),         d_ff = 4*d_model = 2048

Both runs use approximately matched parameter counts.

Usage:
    python scripts/swiglu_ablation.py \
        --train_data data/tinystories/TinyStoriesV2-GPT4-train.txt \
        --val_data data/tinystories/TinyStoriesV2-GPT4-valid.txt \
        --vocab_filepath output/tinystories_bpe/vocab.json \
        --merges_filepath output/tinystories_bpe/merges.json \
        --optimal_lr 3e-3
"""

import argparse
import subprocess


SWEEP_ITERS = 2000
SWEEP_VAL_INTERVAL = 200
SWEEP_LOG_INTERVAL = 50
SWEEP_CKPT_INTERVAL = 1000

D_MODEL = 512

# SwiGLU: d_ff = floor((8/3 * 512) / 64) * 64 = floor(1365.3 / 64) * 64 = 1344
SWIGLU_D_FF = (int((8 / 3) * D_MODEL) // 64) * 64  # 1344

# SiLU: d_ff = 4 * d_model to match param count (2 matrices vs 3)
SILU_D_FF = 4 * D_MODEL  # 2048

MODEL_DEFAULTS = [
    "--vocab_size", "10000",
    "--context_length", "256",
    "--d_model", str(D_MODEL),
    "--num_layers", "4",
    "--num_heads", "16",
    "--batch_size", "64",
    "--weight_decay", "0.1",
    "--grad_clip", "1.0",
]


def build_cmd(extra_args):
    return ["uv", "run", "python", "-m", "cs336_basics.training_together"] + extra_args


def run_experiment(lr, tag, d_ff, args, no_swiglu=False):
    lr_min = lr / 10
    run_name = f"swiglu_ablation_{tag}_lr{lr:.0e}"
    checkpoint_dir = f"output/checkpoints/swiglu_ablation/{tag}_lr{lr:.0e}"
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
        "--d_ff", str(d_ff),
    ] + MODEL_DEFAULTS

    if no_swiglu:
        extra.append("--no_swiglu")

    cmd = build_cmd(extra)
    print(f"\n[{tag}] lr={lr:.0e}  d_ff={d_ff}  cmd:\n  {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="SwiGLU vs SiLU FFN ablation")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--vocab_filepath", required=True)
    parser.add_argument("--merges_filepath", required=True)
    parser.add_argument("--data_cache_dir", default="/tmp/ts_cache")
    parser.add_argument("--optimal_lr", type=float, default=3e-3,
                        help="Previously optimal LR")
    parser.add_argument("--wandb_project", default="cs336-swiglu-ablation")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"SwiGLU d_ff={SWIGLU_D_FF}, SiLU d_ff={SILU_D_FF}")

    print("\n=== Run 1: SwiGLU (baseline) at optimal LR ===")
    run_experiment(args.optimal_lr, "swiglu", SWIGLU_D_FF, args, no_swiglu=False)

    print("\n=== Run 2: SiLU (no gate) at optimal LR ===")
    run_experiment(args.optimal_lr, "silu", SILU_D_FF, args, no_swiglu=True)

    print("\nAblation complete. Compare runs in W&B under project:", args.wandb_project)


if __name__ == "__main__":
    main()
