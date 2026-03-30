"""
Training script that puts together all components for training a TransformerLM.

Usage:
    python -m cs336_basics.training_together \
        --train_data data/tinystories_valid/train.txt \
        --val_data data/tinystories_valid/val.txt \
        --vocab_filepath output/tinystories_bpe/vocab.txt \
        --merges_filepath output/tinystories_bpe/merges.txt \
        --vocab_size 10000 \
        --max_iters 200 \
        --checkpoint_dir output/checkpoints/tinystories_valid/
"""

import argparse
import os
import time

import numpy as np
import torch
import wandb

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM


def tokenize_to_npy(txt_path: str, tokenizer: Tokenizer, npy_path: str) -> np.memmap:
    """Tokenize a txt file and save as uint16 npy, return memmap."""
    print(f"Tokenizing {txt_path} -> {npy_path} ...")
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    ids = list(tokenizer.encode_iterable(lines))
    arr = np.array(ids, dtype=np.uint16)
    np.save(npy_path, arr)
    print(f"  Saved {len(arr):,} tokens to {npy_path}")


def estimate_val_loss(model, val_data, batch_size, context_length, device, num_batches=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(val_data, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )

    # Tokenize txt files to npy if needed, then load
    tokenizer = Tokenizer.from_files(args.vocab_filepath, args.merges_filepath)

    def _load_data(txt_path: str) -> np.memmap:
        npy_path = os.path.splitext(txt_path)[0] + ".npy"
        if not os.path.exists(npy_path):
            tokenize_to_npy(txt_path, tokenizer, npy_path)
        else:
            print(f"Found cached {npy_path}, skipping tokenization.")
        return np.memmap(npy_path, dtype=np.uint16, mode="r")

    train_data = _load_data(args.train_data)
    val_data = _load_data(args.val_data)
    print(f"Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")

    # Build model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Resume from checkpoint if provided
    start_iter = 0
    if args.resume_checkpoint:
        start_iter = load_checkpoint(args.resume_checkpoint, model, optimizer)
        print(f"Resumed from checkpoint at iteration {start_iter}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model.train()
    t0 = time.time()
    train_start_time = time.time()

    for step in range(start_iter, args.max_iters):
        # Set learning rate
        lr = get_lr_cosine_schedule(
            t=step,
            alpha_max=args.lr_max,
            alpha_min=args.lr_min,
            T_w=args.warmup_iters,
            T_c=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            wallclock = time.time() - train_start_time
            tokens_per_sec = args.log_interval * args.batch_size * args.context_length / max(elapsed, 1e-9)
            print(
                f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                f"{tokens_per_sec:.0f} tok/s"
            )
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/tokens_per_sec": tokens_per_sec, "wallclock": wallclock}, step=step)
            t0 = time.time()

        # Validation
        if step % args.val_interval == 0:
            val_loss = estimate_val_loss(model, val_data, args.batch_size, args.context_length, device)
            wallclock = time.time() - train_start_time
            print(f"  --> val loss {val_loss:.4f} at step {step}")
            wandb.log({"val/loss": val_loss, "wallclock": wallclock}, step=step)

        # Checkpointing
        if step > 0 and step % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{step:07d}.pt")
            save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"  --> saved checkpoint: {ckpt_path}")

    # Final checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{args.max_iters:07d}_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, ckpt_path)
    print(f"Training complete. Final checkpoint: {ckpt_path}")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train a TransformerLM")

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Path to raw train text (.txt)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to raw val text (.txt)")
    parser.add_argument("--vocab_filepath", type=str, required=True, help="Path to BPE vocab JSON")
    parser.add_argument("--merges_filepath", type=str, required=True, help="Path to BPE merges JSON")

    # Model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training
    parser.add_argument("--max_iters", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")

    # Optimizer
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Checkpointing & logging
    parser.add_argument("--checkpoint_dir", type=str, default="output/checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="cs336-lm")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
