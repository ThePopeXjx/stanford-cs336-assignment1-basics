#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

from .tokenizer import Tokenizer

# SAMPLE_ROOT = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/tinystories/samples")
# VOCAB_PATH = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/tinystories/bpe_vocab.json")
# MERGES_PATH = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/tinystories/bpe_merges.txt")
SAMPLE_ROOT = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/openwebtext/samples")
VOCAB_PATH = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/openwebtext/bpe_vocab.json")
MERGES_PATH = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/openwebtext/bpe_merges.txt")

# DATASET_PATH = Path("/mnt/data1/jiaxingxu/TinyStoriesV2-GPT4-train.txt")
# NPY_PATH = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/tinystories/train_tokens.npy")
# DATASET_PATH = Path("/mnt/data1/jiaxingxu/TinyStoriesV2-GPT4-valid.txt")
# NPY_PATH = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/tinystories/dev_tokens.npy")
# DATASET_PATH = Path("/mnt/data1/jiaxingxu/owt_train.txt")
# NPY_PATH = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/openwebtext/train_tokens.npy")
DATASET_PATH = Path("/mnt/data1/jiaxingxu/owt_valid.txt")
NPY_PATH = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/openwebtext/dev_tokens.npy")


def test_encoding_performance(sample_root: Path, vocab_path: Path, merges_path: Path):
    num_bytes = 0
    num_tokens = 0
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
    samples = os.listdir(sample_root)
    texts: list[str] = list()
    for sample in samples:
        with open(sample_root / sample, "r", encoding="utf-8") as f:
            texts.append(f.read())
    start = time.perf_counter()
    for text in texts:
        num_bytes += len(text.encode("utf-8"))
        ids = tokenizer.encode(text)
        num_tokens += len(ids)
    elapsed_time = time.perf_counter() - start

    stats = {
        "compression_ratio": num_bytes / num_tokens,
        "throughput": num_bytes / elapsed_time
    }

    print("=" * 72)
    print("Encoding Performance Summary")
    print(f"- compression ratio : {num_bytes / num_tokens:.2f}")
    print(f"- throughput : {num_bytes / elapsed_time:.2f}")
    print("=" * 72)
    return stats


def encode_dataset(dataset_path: Path, npy_path: Path,
                   vocab_path: Path, merges_path: Path, show_progress: bool = False):
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
    ids: list[int] = list()
    with open(dataset_path, "r", encoding="utf-8") as f:
        if show_progress:
            total_lines = sum(1 for _ in f)
            f.seek(0)
        iterator = tqdm(f, total=total_lines, desc="Encoding", disable=not show_progress)
        for id in tokenizer.encode_iterable(iterator):
            ids.append(id)
    arr = np.array(ids, dtype=np.uint16)
    np.save(npy_path, arr)


if __name__ == "__main__":
    # test_encoding_performance(SAMPLE_ROOT, VOCAB_PATH, MERGES_PATH)
    encode_dataset(DATASET_PATH, NPY_PATH, VOCAB_PATH, MERGES_PATH, show_progress=True)