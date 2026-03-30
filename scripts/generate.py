"""
Generate text from a trained TransformerLM checkpoint.

Usage:
    uv run python scripts/generate.py \
        --checkpoint output/checkpoints/lr_sweep/lr3e-03//ckpt_0005000_final.pt \
        --vocab_filepath output/tinystories_bpe/vocab.json \
        --merges_filepath output/tinystories_bpe/merges.json \
        --prompt "Once upon a time" \
        --max_new_tokens 300 \
        --temperature 0.8 \
        --top_p 0.9
"""

import argparse
import torch

from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.decoding import decode
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained TransformerLM checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--vocab_filepath", type=str, required=True)
    parser.add_argument("--merges_filepath", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    # Model config — must match the checkpoint
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"Using device: {device}")

    tokenizer = Tokenizer.from_files(args.vocab_filepath, args.merges_filepath)

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

    # Load checkpoint (load_checkpoint needs an optimizer too)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    iteration = load_checkpoint(args.checkpoint, model, optimizer)
    print(f"Loaded checkpoint at iteration {iteration}")

    print(f"\nPrompt: {args.prompt!r}")
    print(f"temperature={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}\n")
    print("=" * 60)

    generated = decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=torch.device(device),
    )

    full_text = args.prompt + generated
    print(full_text)
    print("=" * 60)
    tokens = tokenizer.encode(full_text)
    print(f"\nGenerated {len(tokens)} tokens total.")


if __name__ == "__main__":
    main()
