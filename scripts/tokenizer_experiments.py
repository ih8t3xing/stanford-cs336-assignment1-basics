"""
Tokenizer experiments: compression ratio (bytes/token) for TinyStories and OWT.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cs336_basics.tokenizer import Tokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]
EOT_STR = "<|endoftext|>"

TS_TRAIN = "data/TinyStoriesV2-GPT4-train.txt"
OWT_TRAIN = "data/owt_train.txt"
TS_VOCAB = "output/tinystories_bpe/vocab.json"
TS_MERGES = "output/tinystories_bpe/merges.json"
OWT_VOCAB = "output/owt_bpe/vocab.json"
OWT_MERGES = "output/owt_bpe/merges.json"

N_DOCS = 10
SEED = 42


def load_documents(path: str, n: int, seed: int) -> list[str]:
    """Sample n documents by first randomly seeking into the file, then reading forward to EOT boundaries."""
    rng = random.Random(seed)
    file_size = os.path.getsize(path)
    eot = EOT_STR.encode("utf-8")
    sampled = []
    seen_offsets = set()

    with open(path, "rb") as f:
        attempts = 0
        while len(sampled) < n and attempts < n * 20:
            attempts += 1
            # Jump to a random position
            pos = rng.randint(0, file_size - 1)
            if pos in seen_offsets:
                continue

            # Seek forward to the next EOT to find the start of a document
            f.seek(pos)
            ahead = f.read(65536)
            eot_pos = ahead.find(eot)
            if eot_pos == -1:
                continue
            doc_start = pos + eot_pos + len(eot)

            # Read forward from doc_start to the next EOT (end of document)
            f.seek(doc_start)
            content = f.read(65536)
            eot_end = content.find(eot)
            if eot_end == -1:
                doc_text = content.decode("utf-8", errors="ignore").strip()
            else:
                doc_text = content[:eot_end].decode("utf-8", errors="ignore").strip()

            if doc_text and doc_start not in seen_offsets:
                seen_offsets.add(doc_start)
                sampled.append(doc_text)

    return sampled


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    """Bytes per token across all docs."""
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
    return total_bytes / total_tokens


def load_tokenizer(vocab_path: str, merges_path: str) -> Tokenizer:
    """Load tokenizer from JSON files saved with latin-1 encoding."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    vocab = {int(k): v.encode("latin-1") for k, v in raw_vocab.items()}

    with open(merges_path, "r", encoding="utf-8") as f:
        raw_merges = json.load(f)
    merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in raw_merges]

    return Tokenizer(vocab, merges, special_tokens=SPECIAL_TOKENS)


import numpy as np

SAMPLE_BYTES = 1_000_000  # 1 MB for throughput benchmark
PILE_GB = 825


def encode_dataset(tokenizer: Tokenizer, data_path: str, out_path: str) -> None:
    """Encode an entire text file to a uint16 NumPy array and save it."""
    with open(data_path, "r", errors="ignore") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    arr = np.array(ids, dtype=np.uint16)
    np.save(out_path, arr)
    print(f"  {os.path.basename(data_path)} -> {out_path} ({len(arr):,} tokens, {arr.nbytes / 1e6:.1f} MB)")


def throughput_benchmark(tokenizer: Tokenizer, data_path: str, label: str) -> float:
    """Measure tokenizer throughput in MB/s on the first SAMPLE_BYTES of data_path."""
    import time

    with open(data_path, "r", errors="ignore") as f:
        sample = f.read(SAMPLE_BYTES)

    n_bytes = len(sample.encode("utf-8"))

    # Warmup
    tokenizer.encode(sample[:10_000])

    start = time.perf_counter()
    ids = tokenizer.encode(sample)
    elapsed = time.perf_counter() - start

    mb_per_sec = n_bytes / elapsed / 1e6
    pile_hours = (PILE_GB * 1e9) / (mb_per_sec * 1e6) / 3600
    print(
        f"{label}: {mb_per_sec:.2f} MB/s over {n_bytes/1e6:.1f} MB "
        f"({len(ids):,} tokens) → Pile would take {pile_hours:.1f} hours"
    )
    return mb_per_sec


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ts_tokenizer = load_tokenizer(
        os.path.join(base, TS_VOCAB),
        os.path.join(base, TS_MERGES),
    )
    owt_tokenizer = load_tokenizer(
        os.path.join(base, OWT_VOCAB),
        os.path.join(base, OWT_MERGES),
    )

    print(f"Sampling {N_DOCS} documents from each corpus...")
    ts_docs = load_documents(os.path.join(base, TS_TRAIN), N_DOCS, SEED)
    owt_docs = load_documents(os.path.join(base, OWT_TRAIN), N_DOCS, SEED)

    ts_ratio = compression_ratio(ts_tokenizer, ts_docs)
    owt_ratio = compression_ratio(owt_tokenizer, owt_docs)
    ts_on_owt_ratio = compression_ratio(ts_tokenizer, owt_docs)
    owt_on_ts_ratio = compression_ratio(owt_tokenizer, ts_docs)

    print(f"TinyStories tokenizer (10K vocab) on TinyStories docs: {ts_ratio:.4f} bytes/token")
    print(f"OpenWebText tokenizer (32K vocab) on OWT docs:          {owt_ratio:.4f} bytes/token")
    print(f"TinyStories tokenizer (10K vocab) on OWT docs:          {ts_on_owt_ratio:.4f} bytes/token")
    print(f"OpenWebText tokenizer (32K vocab) on TinyStories docs:  {owt_on_ts_ratio:.4f} bytes/token")

    print("\nThroughput benchmark (1 MB sample):")
    throughput_benchmark(ts_tokenizer, os.path.join(base, TS_TRAIN), "TinyStories tokenizer on TinyStories")
    throughput_benchmark(owt_tokenizer, os.path.join(base, OWT_TRAIN), "OWT tokenizer on OWT")

    print("\nEncoding datasets to uint16 NumPy arrays...")
    encode_dataset(ts_tokenizer, os.path.join(base, TS_TRAIN), os.path.join(base, "output/tinystories_train.npy"))
    encode_dataset(ts_tokenizer, os.path.join(base, "data/TinyStoriesV2-GPT4-valid.txt"), os.path.join(base, "output/tinystories_valid.npy"))
    encode_dataset(owt_tokenizer, os.path.join(base, OWT_TRAIN), os.path.join(base, "output/owt_train.npy"))
    encode_dataset(owt_tokenizer, os.path.join(base, "data/owt_valid.txt"), os.path.join(base, "output/owt_valid.npy"))


if __name__ == "__main__":
    main()
