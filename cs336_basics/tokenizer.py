from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

from tqdm import tqdm

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Reverse mapping: bytes -> token ID
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        # Merge priority: pair -> rank (lower = applied first)
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # Regex to split on special tokens (longest match first to handle overlaps)
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pat = re.compile(
                "(" + "|".join(re.escape(s) for s in sorted_specials) + ")"
            )
            self._special_set = set(self.special_tokens)
        else:
            self._special_pat = None
            self._special_set = set()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        # JSON keys are strings; values are lists of ints (byte arrays)
        vocab = {
            int(k): v.encode("latin-1") if isinstance(v, str) else bytes(v)
            for k, v in raw_vocab.items()
        }

        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)
        merges = [
            (
                a.encode("latin-1") if isinstance(a, str) else bytes(a),
                b.encode("latin-1") if isinstance(b, str) else bytes(b),
            )
            for a, b in raw_merges
        ]

        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        """Apply BPE merges to a list of byte tokens (single pre-token)."""
        while len(tokens) >= 2:
            best_rank = None
            best_i = -1
            for i in range(len(tokens) - 1):
                rank = self.merge_rank.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i == -1:
                break
            merged = tokens[best_i] + tokens[best_i + 1]
            tokens = tokens[:best_i] + [merged] + tokens[best_i + 2 :]
        return tokens

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a plain text chunk (no special tokens) to token IDs."""
        ids: list[int] = []
        for match in re.finditer(PAT, text):
            word = match.group().encode("utf-8")
            byte_tokens = [bytes([b]) for b in word]
            merged = self._apply_merges(byte_tokens)
            for tok in merged:
                ids.append(self.bytes_to_id[tok])
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self._special_pat:
            parts = self._special_pat.split(text)
            for part in parts:
                if not part:
                    continue
                if part in self._special_set:
                    ids.append(self.bytes_to_id[part.encode("utf-8")])
                else:
                    ids.extend(self._encode_chunk(part))
        else:
            ids.extend(self._encode_chunk(text))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings, yielding token IDs one at a time.

        Each string in the iterable is encoded independently so that no token
        crosses item boundaries, keeping memory usage constant in the number
        of items processed at once.
        """
        for text in tqdm(iterable, desc="Tokenizing"):
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")
