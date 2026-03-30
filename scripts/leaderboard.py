"""
Leaderboard submission: minimize OWT val loss within 1.5 H100-hours.

Key optimizations over the baseline 25M TinyStories model:
  - Larger model: d_model=768, 12 layers → ~110M params
  - Weight tying (input embedding = LM head)
  - FlashAttention via F.scaled_dot_product_attention
  - bfloat16 autocast
  - torch.compile
  - Longer context (512) and larger batch (128)
  - Chinchilla-informed scale: ~2B tokens budget at this throughput

Usage:
    python scripts/leaderboard.py
    python scripts/leaderboard.py --max_iters 30000  # shorter test run
    python scripts/leaderboard.py --dry_run
"""

import argparse
import os
import subprocess

# ── Model config ─────────────────────────────────────────────────────────────
D_MODEL    = 768
NUM_LAYERS = 12
NUM_HEADS  = 12                                     # d_k = 768/12 = 64
# SwiGLU d_ff: floor(8/3 * 768 / 64) * 64 = 2048
D_FF       = (int((8 / 3) * D_MODEL) // 64) * 64  # 2048

MODEL_ARGS = [
    "--vocab_size",     "32000",
    "--context_length", "512",
    "--d_model",        str(D_MODEL),
    "--num_layers",     str(NUM_LAYERS),
    "--num_heads",      str(NUM_HEADS),
    "--d_ff",           str(D_FF),
    "--batch_size",     "128",
    "--weight_decay",   "0.1",
    "--grad_clip",      "1.0",
    "--use_flash",
    "--tie_weights",
    "--bf16",
    "--compile",
]

# ── Training schedule ─────────────────────────────────────────────────────────
# Chinchilla: ~20 tokens/param → 110M × 20 = 2.2B tokens
# tokens/iter = 128 × 512 = 65,536  →  2.2B / 65k ≈ 33,500 iters
# With compile+bf16+flash on H100, ~500k tok/s → 1.5h = 2.7B tokens → ~41k iters
# Set conservatively to 40k; increase if H100 is faster.
MAX_ITERS   = 40_000
LR          = 3e-4       # lower LR for larger model
LR_MIN      = 3e-5
WARMUP      = 1_000
VAL_INT     = 500
LOG_INT     = 100
CKPT_INT    = 5_000


def tokenize_owt(args):
    out_dir   = args.data_cache_dir
    train_npy = os.path.join(out_dir, "train.npy")
    val_npy   = os.path.join(out_dir, "val.npy")
    if os.path.exists(train_npy) and os.path.exists(val_npy):
        print(f"[tokenize] Found {out_dir}/{{train,val}}.npy — skipping.")
        return
    cmd = [
        "uv", "run", "python", "scripts/prepare_data.py",
        "--vocab",      args.vocab_filepath,
        "--merges",     args.merges_filepath,
        "--train_text", args.train_data,
        "--val_text",   args.val_data,
        "--out_dir",    out_dir,
        "--dtype",      "uint16",
    ]
    print("[tokenize]", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def train(args):
    cmd = [
        "uv", "run", "python", "-m", "cs336_basics.training_together",
        "--train_data",          args.train_data,
        "--val_data",            args.val_data,
        "--vocab_filepath",      args.vocab_filepath,
        "--merges_filepath",     args.merges_filepath,
        "--data_cache_dir",      args.data_cache_dir,
        "--lr_max",              str(LR),
        "--lr_min",              str(LR_MIN),
        "--max_iters",           str(args.max_iters),
        "--warmup_iters",        str(WARMUP),
        "--val_interval",        str(VAL_INT),
        "--log_interval",        str(LOG_INT),
        "--checkpoint_interval", str(CKPT_INT),
        "--checkpoint_dir",      args.checkpoint_dir,
        "--wandb_project",       args.wandb_project,
        "--wandb_run_name",      "leaderboard",
    ] + MODEL_ARGS
    print("[train]", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="OWT leaderboard run")
    parser.add_argument("--train_data",      default="data/owt_train.txt")
    parser.add_argument("--val_data",        default="data/owt_valid.txt")
    parser.add_argument("--vocab_filepath",  default="output/owt_bpe/vocab.json")
    parser.add_argument("--merges_filepath", default="output/owt_bpe/merges.json")
    parser.add_argument("--data_cache_dir",  default="data/owt")
    parser.add_argument("--checkpoint_dir",  default="output/checkpoints/leaderboard")
    parser.add_argument("--max_iters",       type=int, default=MAX_ITERS)
    parser.add_argument("--wandb_project",   default="cs336-leaderboard")
    parser.add_argument("--dry_run",         action="store_true")
    args = parser.parse_args()

    tokens_per_iter = 128 * 512
    total_tokens = args.max_iters * tokens_per_iter
    print(f"Model: d_model={D_MODEL}, {NUM_LAYERS} layers, {NUM_HEADS} heads, d_ff={D_FF}")
    print(f"Optimizations: FlashAttention + bf16 + torch.compile + weight tying")
    print(f"Training: {args.max_iters:,} iters × {tokens_per_iter:,} tok/iter = {total_tokens/1e9:.2f}B tokens")
    print()

    os.makedirs(args.data_cache_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=== Step 1: Tokenize OWT ===")
    tokenize_owt(args)

    print("\n=== Step 2: Train ===")
    train(args)

    ckpt = os.path.join(args.checkpoint_dir, f"ckpt_{args.max_iters:07d}_final.pt")
    print(f"\nDone. Generate text with:")
    print(f"  uv run python scripts/generate.py \\")
    print(f"    --checkpoint {ckpt} \\")
    print(f"    --vocab_filepath {args.vocab_filepath} \\")
    print(f"    --merges_filepath {args.merges_filepath} \\")
    print(f"    --vocab_size 32000 --d_model {D_MODEL} --num_layers {NUM_LAYERS} --num_heads {NUM_HEADS} --d_ff {D_FF} --context_length 512 \\")
    print(f'    --prompt "The latest research shows" --max_new_tokens 300')


if __name__ == "__main__":
    main()
