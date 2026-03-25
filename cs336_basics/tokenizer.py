#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Iterable, Iterator

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


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.reverse_vocab = { v: k for k, v in vocab.items() }
        self.merges = { t: i for i, t in enumerate(merges) }
        self.special_tokens = special_tokens
        if special_tokens:
            escaped = [ re.escape(t) for t in sorted(special_tokens, key=len, reverse=True) ]
            self.special_pat = re.compile("(" + "|".join(escaped) + ")")

    @classmethod
    def from_files(cls, vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None) -> "Tokenizer":
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = { v: k for k, v in byte_encoder.items() }
        def decode_str_to_bytes(s: str) -> bytes:
            b = bytes()
            for c in s:
                b += bytes([byte_decoder[c]])
            return b

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            serialized_vocab = json.load(f)
        vocab = {
            idx: decode_str_to_bytes(str_repr)
            for str_repr, idx in serialized_vocab.items()
        }

        merges = list()
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                pairs = line.rstrip("\n").split()
                assert len(pairs) == 2
                merges.append((decode_str_to_bytes(pairs[0]),
                               decode_str_to_bytes(pairs[1])))
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        token_ids = list()
        pre_tokens: list[str] = list()
        if self.special_tokens:
            start = 0
            for m in self.special_pat.finditer(text):
                # First, process plain texts before special tokens
                if m.start() > start:
                    pre_tokens.extend(re.findall(PAT, text[start:m.start()]))
                # Retain special tokens
                pre_tokens.append(m.group(0))
                start = m.end()
            # Finally, add plain texts after all special tokens.
            if start < len(text):
                pre_tokens.extend(re.findall(PAT, text[start:]))
        else:
            pre_tokens = re.findall(PAT, text)
        for pre_token in pre_tokens:
            if self.special_tokens and pre_token in self.special_tokens:
                token_ids.append(self.reverse_vocab[pre_token.encode("utf-8")])
            else:
                pre_bytes = pre_token.encode("utf-8")
                pre_bytes = [bytes([b]) for b in pre_bytes]
                old_bytes = pre_bytes
                # In naive implementation, we should iterate merges from top (high priority) to bottom (low priority).
                # But consider the time complexity, it will be len(merges) * len(text) approximately, that's quite slow.
                # In this implementation, I pre-process the initial merges to convert it into a priority dict.
                # So that we don't have to iterate on the merge level.
                # Just find the pair that has the highest priority in current sequence.
                # Caution, the priority comparison is very important.
                # If we only look up the pair in merges, it will produce the wrong result.
                while True:
                    if len(old_bytes) == 1:
                        break
                    orders = [
                        self.merges.get(p, len(self.merges)) for p in zip(old_bytes, old_bytes[1:])
                    ]
                    max_order = min(orders)
                    if max_order < len(self.merges):
                        match_ids = [i for i, v in enumerate(orders) if v == max_order]
                        new_bytes = list()
                        i = 0
                        while i < len(old_bytes):
                            if i in match_ids:
                                new_bytes.append(old_bytes[i] + old_bytes[i + 1])
                                i += 2
                            else:
                                new_bytes.append(old_bytes[i])
                                i += 1
                        old_bytes = new_bytes
                    else:
                        break
                for b in old_bytes:
                    token_ids.append(self.reverse_vocab[b])
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            for id in token_ids:
                yield id
    
    def decode(self, ids: list[int]) -> str:
        b = bytes()
        for id in ids:
            b += self.vocab[id]
        return b.decode(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    # train_bpe_tinystories()
    train_bpe_openwebtext()
