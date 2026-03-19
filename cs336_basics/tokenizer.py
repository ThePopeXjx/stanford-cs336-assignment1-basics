#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import os

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()
    special_token_bytes = [s.encode("utf-8") for s in special_tokens]
    ordinary_token_bytes = [bytes([i]) for i in range(256)]
    vocab = { idx: token_bytes for idx, token_bytes in enumerate(special_token_bytes + ordinary_token_bytes) }
    nxt_idx = len(vocab)
    merges = list()
    
    # Pre-tokenize
    words_freq = dict(Counter(re.findall(PAT, corpus)))
    bytes_seq_freq = {tuple(word.encode("utf-8")): count for word, count in words_freq.items()}
    idx_seq_freq = {tuple(token_bytes + len(special_tokens) for token_bytes in bytes_seq): count
                    for bytes_seq, count in bytes_seq_freq.items()}
    
    overall_pair_counts = {}
    for idx_seq, count in idx_seq_freq.items():
        idx_seq_pair_counts = get_idx_seq_pair_counts(idx_seq)
        for pair, pair_count in idx_seq_pair_counts.items():
            overall_pair_counts[pair] = overall_pair_counts.get(pair, 0) + count * pair_count

    # Compute merges
    while nxt_idx < vocab_size:
        if len(overall_pair_counts) == 0:
            break
        # be careful with the tie breaker
        max_pair = max(
            overall_pair_counts,
            key=lambda pair: (
                overall_pair_counts[pair],
                vocab[pair[0]],
                vocab[pair[1]],
            )
        )
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
        vocab[nxt_idx] = vocab[max_pair[0]] + vocab[max_pair[1]]
        # maintain overall_pair_counts
        new_idx_seq_freq = {}
        for idx_seq, count in idx_seq_freq.items():
            new_idx_seq = []
            i = 0
            while i < len(idx_seq):
                if i < len(idx_seq) - 1 and (idx_seq[i], idx_seq[i + 1]) == max_pair:
                    new_idx_seq.append(nxt_idx)
                    i += 2
                else:
                    new_idx_seq.append(idx_seq[i])
                    i += 1
            new_idx_seq = tuple(new_idx_seq)
            new_idx_seq_freq[new_idx_seq] = count
            if new_idx_seq != idx_seq:
                # remove old counts
                idx_seq_pair_counts = get_idx_seq_pair_counts(idx_seq)
                for pair, pair_count in idx_seq_pair_counts.items():
                    overall_pair_counts[pair] = overall_pair_counts[pair] - count * pair_count
                    if overall_pair_counts[pair] == 0:
                        overall_pair_counts.pop(pair)
                # add new counts
                new_idx_seq_pair_counts = get_idx_seq_pair_counts(new_idx_seq)
                for pair, pair_count in new_idx_seq_pair_counts.items():
                    overall_pair_counts[pair] = overall_pair_counts.get(pair, 0) + count * pair_count
        idx_seq_freq = new_idx_seq_freq
        # increment vocab size
        nxt_idx += 1
    
    return vocab, merges


def get_idx_seq_pair_counts(
    idx_seq: tuple[int],
):
    pair_counts = {}
    for i in range(len(idx_seq) - 1):
        idx_pair = (idx_seq[i], idx_seq[i + 1])
        pair_counts[idx_pair] = pair_counts.get(idx_pair, 0) + 1
    return pair_counts
