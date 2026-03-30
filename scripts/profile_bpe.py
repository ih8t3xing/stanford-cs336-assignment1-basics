import io
import sys
import os
import cProfile
import pstats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    from cs336_basics.train_bpe import train_bpe

    pr = cProfile.Profile()
    pr.enable()
    train_bpe("tests/fixtures/corpus.en", vocab_size=500, special_tokens=["<|endoftext|>"])
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    print(s.getvalue())
