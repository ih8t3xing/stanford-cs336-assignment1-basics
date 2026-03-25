import json
import multiprocessing as mp
import os
import time

import psutil

from cs336_basics.train_bpe import train_bpe


def main():
    input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    output_dir = "../output/tinystories_bpe"
    os.makedirs(output_dir, exist_ok=True)

    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    # num_processes = mp.cpu_count()
    num_processes = 8

    proc = psutil.Process(os.getpid())

    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes,
        verbose=True,
    )
    t1 = time.perf_counter()

    peak_rss = proc.memory_info().rss
    elapsed_s = t1 - t0

    print(f"Elapsed: {elapsed_s:.2f}s ({elapsed_s/60:.2f} min)")
    print(f"RSS: {peak_rss / (1024**3):.2f} GB")
    print(f"Vocab size: {len(vocab)}")
    print(f"Num merges: {len(merges)}")

    def safe_decode(b: bytes) -> str:
        return b.decode('latin-1')

    # Longest token
    top10 = sorted(vocab.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
    print("\nTop 10 longest tokens:")
    for rank, (tid, bts) in enumerate(top10, 1):
        print(f"  {rank:2d}. id={tid:5d} len={len(bts):3d}  {repr(bts.decode('utf-8', errors='replace'))}")

    # Serialize vocab: {str(id): latin1_str}
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): safe_decode(v) for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)

    # Serialize merges: [[latin1_str, latin1_str], ...]
    merges_path = os.path.join(output_dir, "merges.json")
    with open(merges_path, 'w', encoding='utf-8') as f:
        json.dump([[safe_decode(a), safe_decode(b)] for a, b in merges], f, ensure_ascii=False, indent=2)

    print(f"\nSaved vocab to:  {vocab_path}")
    print(f"Saved merges to: {merges_path}")


if __name__ == "__main__":
    main()
