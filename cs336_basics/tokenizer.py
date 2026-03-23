#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import resource
import sys
import time

import regex as re
from tqdm import tqdm

from tests.common import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    enable_progress_bar: bool = False,
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
    with tqdm(
        total=vocab_size,
        initial=len(vocab),
        disable=not enable_progress_bar,
        desc="BPE vocab",
    ) as pbar:
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
            pbar.update(1)
    
    return vocab, merges


def get_idx_seq_pair_counts(
    idx_seq: tuple[int],
):
    pair_counts = {}
    for i in range(len(idx_seq) - 1):
        idx_pair = (idx_seq[i], idx_seq[i + 1])
        pair_counts[idx_pair] = pair_counts.get(idx_pair, 0) + 1
    return pair_counts


def train_bpe_tinystories(
    input_path: str | os.PathLike = "/mnt/data1/jiaxingxu/TinyStoriesV2-GPT4-train.txt",
    vocab_size: int = 10000,
    special_tokens: list[str] = ["<|endoftext|>"],
    output_dir: str | os.PathLike = "/home/jiaxingxu/spring2024-assignment1-basics/output/tinystories",
    vocab_filename: str = "bpe_vocab.json",
    merges_filename: str = "bpe_merges.txt",
):
    start = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        enable_progress_bar=True,
    )
    elapsed_seconds = time.perf_counter() - start

    # On Linux ru_maxrss is in KB; on macOS it is in bytes.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_kb = peak_rss / 1024 if sys.platform == "darwin" else peak_rss
    peak_rss_gb = peak_rss_kb / (1024 * 1024)

    byte_encoder = gpt2_bytes_to_unicode()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    serialized_vocab = {
        "".join(byte_encoder[b] for b in token_bytes): token_id
        for token_id, token_bytes in vocab.items()
    }
    with open(output_path / vocab_filename, "w", encoding="utf-8") as f:
        json.dump(serialized_vocab, f, ensure_ascii=False, indent=4)

    with open(output_path / merges_filename, "w", encoding="utf-8") as f:
        for token_1, token_2 in merges:
            t1 = "".join(byte_encoder[b] for b in token_1)
            t2 = "".join(byte_encoder[b] for b in token_2)
            f.write(f"{t1} {t2}\n")

    longest_token_bytes = max(vocab.values(), key=len)
    longest_token_str = "".join(byte_encoder[b] for b in longest_token_bytes)

    stats = {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hours": elapsed_seconds / 3600.0,
        "peak_rss_kb": peak_rss_kb,
        "peak_rss_gb": peak_rss_gb,
        "longest_token_num_bytes": len(longest_token_bytes),
        "longest_token": longest_token_str,
        "vocab_path": str((output_path / vocab_filename).resolve()),
        "merges_path": str((output_path / merges_filename).resolve()),
    }

    print("=" * 72)
    print("TinyStories BPE Training Summary")
    print("- elapsed time : "
          f"{stats['elapsed_seconds']:.2f}s ({stats['elapsed_hours']:.4f}h)")
    print(f"- peak memory  : {stats['peak_rss_kb']:.0f} KB ({stats['peak_rss_gb']:.3f} GB)")
    print(f"- longest token: {stats['longest_token_num_bytes']} bytes")
    print(f"- token text   : {stats['longest_token']}")
    print(f"- vocab path   : {stats['vocab_path']}")
    print(f"- merges path  : {stats['merges_path']}")
    print("=" * 72)
    return stats


def train_bpe_openwebtext(
    input_path: str | os.PathLike = "/mnt/data1/jiaxingxu/owt_train.txt",
    vocab_size: int = 32000,
    special_tokens: list[str] = ["<|endoftext|>"],
    output_dir: str | os.PathLike = "/home/jiaxingxu/spring2024-assignment1-basics/output/openwebtext",
    vocab_filename: str = "bpe_vocab.json",
    merges_filename: str = "bpe_merges.txt",
):
    start = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        enable_progress_bar=True,
    )
    elapsed_seconds = time.perf_counter() - start

    # On Linux ru_maxrss is in KB; on macOS it is in bytes.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_kb = peak_rss / 1024 if sys.platform == "darwin" else peak_rss
    peak_rss_gb = peak_rss_kb / (1024 * 1024)

    byte_encoder = gpt2_bytes_to_unicode()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    serialized_vocab = {
        "".join(byte_encoder[b] for b in token_bytes): token_id
        for token_id, token_bytes in vocab.items()
    }
    with open(output_path / vocab_filename, "w", encoding="utf-8") as f:
        json.dump(serialized_vocab, f, ensure_ascii=False, indent=4)

    with open(output_path / merges_filename, "w", encoding="utf-8") as f:
        for token_1, token_2 in merges:
            t1 = "".join(byte_encoder[b] for b in token_1)
            t2 = "".join(byte_encoder[b] for b in token_2)
            f.write(f"{t1} {t2}\n")

    longest_token_bytes = max(vocab.values(), key=len)
    longest_token_str = "".join(byte_encoder[b] for b in longest_token_bytes)

    stats = {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hours": elapsed_seconds / 3600.0,
        "peak_rss_kb": peak_rss_kb,
        "peak_rss_gb": peak_rss_gb,
        "longest_token_num_bytes": len(longest_token_bytes),
        "longest_token": longest_token_str,
        "vocab_path": str((output_path / vocab_filename).resolve()),
        "merges_path": str((output_path / merges_filename).resolve()),
    }

    print("=" * 72)
    print("OpenWebText BPE Training Summary")
    print("- elapsed time : "
          f"{stats['elapsed_seconds']:.2f}s ({stats['elapsed_hours']:.4f}h)")
    print(f"- peak memory  : {stats['peak_rss_kb']:.0f} KB ({stats['peak_rss_gb']:.3f} GB)")
    print(f"- longest token: {stats['longest_token_num_bytes']} bytes")
    print(f"- token text   : {stats['longest_token']}")
    print(f"- vocab path   : {stats['vocab_path']}")
    print(f"- merges path  : {stats['merges_path']}")
    print("=" * 72)
    return stats


if __name__ == "__main__":
    # train_bpe_tinystories()
    train_bpe_openwebtext()
