"""
OWT experiment: train the same model on OpenWebText.

Steps:
  1. Tokenize OWT train/val text → data/owt/train.npy, val.npy  (skipped if already done)
  2. Train for the same 100k iterations as TinyStories

Usage:
    # Full run (100k iters, same as TinyStories)
    python scripts/owt_experiment.py

    # Shorter run (e.g. 50k iters for ~1.5h)
    python scripts/owt_experiment.py --max_iters 50000

    # Dry run (print commands only)
    python scripts/owt_experiment.py --dry_run
"""

import argparse
import os
import subprocess


# ── Model config – identical to TinyStories full run ────────────────────────
D_MODEL = 512
D_FF = (int((8 / 3) * D_MODEL) // 64) * 64  # SwiGLU: 1344
MODEL_DEFAULTS = [
    "--vocab_size",     "32000",   # OWT BPE uses 32k vocab
    "--context_length", "256",
    "--d_model",        str(D_MODEL),
    "--num_layers",     "4",
    "--num_heads",      "16",
    "--d_ff",           str(D_FF),
    "--batch_size",     "64",
    "--weight_decay",   "0.1",
    "--grad_clip",      "1.0",
]

LR         = 3e-3
LR_MIN     = LR / 10
WARMUP     = 500
MAX_ITERS  = 100_000
VAL_INT    = 1_000
LOG_INT    = 200
CKPT_INT   = 10_000


def tokenize_owt(args):
    out_dir = args.data_cache_dir
    train_npy = os.path.join(out_dir, "train.npy")
    val_npy   = os.path.join(out_dir, "val.npy")
    if os.path.exists(train_npy) and os.path.exists(val_npy):
        print(f"[tokenize] Already done: {out_dir}/{{train,val}}.npy — skipping.")
        return

    cmd = [
        "uv", "run", "python", "scripts/prepare_data.py",
        "--vocab",       args.vocab_filepath,
        "--merges",      args.merges_filepath,
        "--train_text",  args.train_data,
        "--val_text",    args.val_data,
        "--out_dir",     out_dir,
        "--dtype",       "uint16",
    ]
    print("[tokenize] cmd:", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def train_owt(args):
    checkpoint_dir = args.checkpoint_dir
    run_name       = "owt_main"
    cmd = [
        "uv", "run", "python", "-m", "cs336_basics.training_together",
        "--train_data",         args.train_data,
        "--val_data",           args.val_data,
        "--vocab_filepath",     args.vocab_filepath,
        "--merges_filepath",    args.merges_filepath,
        "--data_cache_dir",     args.data_cache_dir,
        "--lr_max",             str(LR),
        "--lr_min",             str(LR_MIN),
        "--max_iters",          str(args.max_iters),
        "--warmup_iters",       str(WARMUP),
        "--val_interval",       str(VAL_INT),
        "--log_interval",       str(LOG_INT),
        "--checkpoint_interval",str(CKPT_INT),
        "--checkpoint_dir",     checkpoint_dir,
        "--wandb_project",      args.wandb_project,
        "--wandb_run_name",     run_name,
    ] + MODEL_DEFAULTS
    print("[train] cmd:", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="OWT main experiment")
    parser.add_argument("--train_data",      default="data/owt_train.txt")
    parser.add_argument("--val_data",        default="data/owt_valid.txt")
    parser.add_argument("--vocab_filepath",  default="output/owt_bpe/vocab.json")
    parser.add_argument("--merges_filepath", default="output/owt_bpe/merges.json")
    parser.add_argument("--data_cache_dir",  default="data/owt",
                        help="Where to store/find tokenized .npy files")
    parser.add_argument("--checkpoint_dir",  default="output/checkpoints/owt_main")
    parser.add_argument("--max_iters",       type=int, default=MAX_ITERS)
    parser.add_argument("--wandb_project",   default="cs336-owt-main")
    parser.add_argument("--dry_run",         action="store_true")
    args = parser.parse_args()

    os.makedirs(args.data_cache_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=== Train on OWT (training_together tokenizes automatically with multiprocessing) ===")
    train_owt(args)

    print("\nDone. Checkpoint dir:", args.checkpoint_dir)
    print("Generate text with:")
    ckpt = os.path.join(args.checkpoint_dir, f"ckpt_{args.max_iters:07d}_final.pt")
    print(f"  uv run python scripts/generate.py \\")
    print(f"    --checkpoint {ckpt} \\")
    print(f"    --vocab_filepath {args.vocab_filepath} \\")
    print(f"    --merges_filepath {args.merges_filepath} \\")
    print(f"    --vocab_size 32000 \\")
    print(f'    --prompt "The quick brown fox" \\')
    print(f"    --max_new_tokens 300")


if __name__ == "__main__":
    main()
