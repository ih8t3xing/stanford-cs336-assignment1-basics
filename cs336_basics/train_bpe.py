import multiprocessing as mp
import os
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _find_chunk_boundaries(
    file_path: str | os.PathLike,
    num_chunks: int,
    split_special: str,
) -> list[int]:
    """Return byte offsets splitting file_path into num_chunks pieces aligned to split_special."""
    split_bytes = split_special.encode('utf-8')
    file_size = os.path.getsize(file_path)
    boundaries = [0]
    with open(file_path, 'rb') as f:
        for i in range(1, num_chunks):
            target = i * file_size // num_chunks
            f.seek(target)
            # Advance to next split_special boundary
            remaining = f.read()
            idx = remaining.find(split_bytes)
            if idx == -1:
                break
            boundaries.append(target + idx + len(split_bytes))
    boundaries.append(file_size)
    return boundaries


def _count_words_in_file_range(
    args: tuple,
) -> dict[tuple[bytes, ...], int]:
    """Read [start, end) bytes from file_path, strip special tokens, and count pre-token freqs."""
    file_path, start, end, special_tokens = args
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')

    freq_map: dict[tuple[bytes, ...], int] = {}
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        split_pattern = '(' + '|'.join(re.escape(s) for s in sorted_specials) + ')'
        special_set = set(special_tokens)
        for part in re.split(split_pattern, chunk):
            if part in special_set or not part:
                continue
            for match in re.finditer(PAT, part):
                word_tuple = tuple(bytes([b]) for b in match.group().encode('utf-8'))
                freq_map[word_tuple] = freq_map.get(word_tuple, 0) + 1
    else:
        for match in re.finditer(PAT, chunk):
            word_tuple = tuple(bytes([b]) for b in match.group().encode('utf-8'))
            freq_map[word_tuple] = freq_map.get(word_tuple, 0) + 1
    return freq_map

def pretokenize(input_path: str) -> list[str]:
    """Pretokenize the input text by splitting on whitespace and punctuation.

    Args:
        text (str): The input text to pretokenize.  
    """
        
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = [match.group() for match in re.finditer(PAT, text)]
    return tokens

def convert_words_to_bytes(tokens: list[str]) -> list[bytes]:
    """Convert a list of string tokens to a list of byte tokens.

    Args:
        tokens (list[str]): A list of string tokens to convert.

    Returns:
        list[bytes]: A list of byte tokens.
    """
    return [token.encode('utf-8') for token in tokens]

def initialize_vocab(tokens: list[bytes], special_tokens: list[str]) -> dict[int, bytes]:
    """Initialize the BPE tokenizer vocabulary with the given tokens and special tokens.

    Args:
        tokens (list[bytes]): A list of byte tokens to initialize the vocabulary with.
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        dict[int, bytes]: The initialized BPE tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
        to bytes (token bytes).
    """
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode('utf-8')
    for i, token in enumerate(tokens):
        vocab[i + len(special_tokens)] = token
    return vocab

def get_token_frequencies(tokens: list[bytes]) -> dict[bytes, int]:
    """Get the frequency of each token in the list of tokens.

    Args:
        tokens (list[bytes]): A list of byte tokens to count frequencies for.
    """
    token_freqs = {}
    for token in tokens:
        if token in token_freqs:
            token_freqs[token] += 1
        else:
            token_freqs[token] = 1
    return token_freqs

def get_pair_frequencies(tokens: list[bytes]) -> dict[tuple[bytes, bytes], int]:
    """Get the frequency of each adjacent token pair in the list of tokens.

    Args:
        tokens (list[bytes]): A list of byte tokens to count adjacent pair frequencies for.
    """
    pair_freqs = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair in pair_freqs:
            pair_freqs[pair] += 1
        else:
            pair_freqs[pair] = 1
    return pair_freqs

def get_most_frequent_pair(pair_freqs: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    """Get the most frequent adjacent token pair from the given pair frequencies.

    Args:
        pair_freqs (dict[tuple[bytes, bytes], int]): A dictionary mapping adjacent token pairs to their frequencies.    
    """
    most_frequent_pair = max(pair_freqs, key=pair_freqs.get)
    return most_frequent_pair   

def merge_pair(tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    """Merge the given adjacent token pair in the list of tokens.

    Args:
        tokens (list[bytes]): A list of byte tokens to merge the pair in.
        pair (tuple[bytes, bytes]): The adjacent token pair to merge, represented as a tuple of bytes (<token1>, <token2>).
    """
    merged_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            merged_tokens.append(pair[0] + pair[1])
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens


def update_vocab(vocab: dict[int, bytes], pair: tuple[bytes, bytes]) -> dict[int, bytes]:
    """Update the BPE tokenizer vocabulary by adding the merged token for the given pair.

    Args:
        vocab (dict[int, bytes]): The current BPE tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
    """
    new_token_id = max(vocab.keys()) + 1
    new_token = pair[0] + pair[1]
    vocab[new_token_id] = new_token
    return vocab


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    import time
    _t = time.perf_counter

    # Step 1: Find file chunk boundaries aligned to the first special token,
    # then dispatch workers to read and pretokenize each chunk from disk in parallel.
    # This avoids reading or regex-splitting the full file in the main process.
    num_processes = kwargs.get('num_processes', mp.cpu_count())
    verbose = kwargs.get('verbose', False)

    t0 = _t()

    split_token = special_tokens[0] if special_tokens else None
    if split_token and num_processes > 1:
        boundaries = _find_chunk_boundaries(input_path, num_processes, split_token)
    else:
        boundaries = [0, os.path.getsize(input_path)]

    worker_args = [
        (input_path, boundaries[i], boundaries[i + 1], special_tokens)
        for i in range(len(boundaries) - 1)
    ]

    if num_processes > 1 and len(worker_args) > 1:
        with mp.Pool(num_processes) as pool:
            freq_maps = pool.map(_count_words_in_file_range, worker_args)
    else:
        freq_maps = [_count_words_in_file_range(worker_args[0])]

    if verbose:
        print(f"[timing] pretokenize:    {_t() - t0:.3f}s"); t0 = _t()

    # Step 2: Initialize vocab: special tokens first, then all 256 individual bytes
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode('utf-8')
    for b in range(256):
        vocab[len(special_tokens) + b] = bytes([b])

    # Step 3: Merge frequency maps from all workers.
    word_freq_map: dict[tuple[bytes, ...], int] = {}
    for fm in freq_maps:
        for word_tuple, freq in fm.items():
            word_freq_map[word_tuple] = word_freq_map.get(word_tuple, 0) + freq
    del freq_maps

    word_list: list[list[bytes]] = []
    freq_list: list[int] = []
    for word_tuple, freq in word_freq_map.items():
        word_list.append(list(word_tuple))
        freq_list.append(freq)
    del word_freq_map

    # Build pair_freqs and reverse index pair_to_word_ids once.
    # pair_to_word_ids[pair] = set of word indices that currently contain pair.
    pair_freqs: dict[tuple[bytes, bytes], int] = {}
    pair_to_word_ids: dict[tuple[bytes, bytes], set[int]] = {}
    for idx, word in enumerate(word_list):
        freq = freq_list[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
            if pair not in pair_to_word_ids:
                pair_to_word_ids[pair] = set()
            pair_to_word_ids[pair].add(idx)

    if verbose:
        print(f"[timing] build indexes:  {_t() - t0:.3f}s"); t0 = _t()

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)
    next_vocab_id = len(vocab)

    for _ in range(num_merges):
        if not pair_freqs:
            break

        # Find the most frequent pair; break ties lexicographically
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
        new_token = best_pair[0] + best_pair[1]

        # Only process words that actually contain best_pair (from the reverse index).
        # Copy to a list since we mutate pair_to_word_ids inside the loop.
        for word_idx in list(pair_to_word_ids.get(best_pair, set())):
            word = word_list[word_idx]
            freq = freq_list[word_idx]

            # Remove this word's contribution to all pair counts.
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] -= freq
                if pair_freqs[pair] <= 0:
                    del pair_freqs[pair]
                    pair_to_word_ids.pop(pair, None)
                else:
                    pair_to_word_ids[pair].discard(word_idx)

            # Merge best_pair in this word.
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word_list[word_idx] = new_word

            # Add the merged word's contribution back to pair counts.
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
                if pair not in pair_to_word_ids:
                    pair_to_word_ids[pair] = set()
                pair_to_word_ids[pair].add(word_idx)

        vocab[next_vocab_id] = new_token
        next_vocab_id += 1
        merges.append(best_pair)

    if verbose:
        print(f"[timing] merge loop:     {_t() - t0:.3f}s")

    return vocab, merges

