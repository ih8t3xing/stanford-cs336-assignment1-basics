"""
Tokenize text files and save as numpy arrays for training.

Usage:
    python scripts/prepare_data.py \
        --vocab output/tinystories_bpe/vocab.json \
        --merges output/tinystories_bpe/merges.json \
        --train_text data/TinyStoriesV2-GPT4-train.txt \
        --val_text data/TinyStoriesV2-GPT4-valid.txt \
        --out_dir data/tinystories/ \
        --dtype uint16
        
    python scripts/prepare_data.py \
        --vocab output/tinystories_bpe/vocab.json \
        --merges output/tinystories_bpe/merges.json \
        --train_text data/TinyStoriesV2-GPT4-valid.txt \
        --val_text data/TinyStoriesV2-GPT4-valid.txt \
        --out_dir data/tinystories_valid/ \
        --dtype uint16

    python scripts/prepare_data.py \
        --vocab output/owt_bpe/vocab.json \
        --merges output/owt_bpe/merges.json \
        --train_text data/owt_train.txt \
        --val_text data/owt_valid.txt \
        --out_dir data/owt/ \
        --dtype uint16

"""

import argparse
import os

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def tokenize_file(tokenizer: Tokenizer, txt_path: str, out_path: str, dtype: str):
    print(f"Tokenizing {txt_path} ...")
    with open(txt_path, "r", encoding="utf-8") as f:
        ids = list(tokenizer.encode_iterable(f))
    arr = np.array(ids, dtype=dtype)
    np.save(out_path, arr)
    print(f"  {len(arr):,} tokens → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Tokenize text data into .npy files")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.train.json")
    parser.add_argument("--merges", type=str, default="bpe_merges.train.json")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--train_text", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--val_text", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--dtype", type=str, default="uint16", choices=["uint16", "uint32"])
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=args.special_tokens,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    tokenize_file(tokenizer, args.train_text, os.path.join(args.out_dir, "train.npy"), args.dtype)
    tokenize_file(tokenizer, args.val_text,   os.path.join(args.out_dir, "val.npy"),   args.dtype)


if __name__ == "__main__":
    main()
