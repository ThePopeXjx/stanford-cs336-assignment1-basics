#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import random
import re

# DATASET_PATH = Path("/mnt/data1/jiaxingxu/TinyStoriesV2-GPT4-train.txt")
# OUTPUT_ROOT = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/tinystories/samples")
DATASET_PATH = Path("/mnt/data1/jiaxingxu/owt_train.txt")
OUTPUT_ROOT = Path("/home/jiaxingxu/spring2024-assignment1-basics/output/openwebtext/samples")


def sample_from_dataset(dataset_path: Path, output_root: Path, num_sample: int):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    dataset_name = os.path.basename(dataset_path).split(".")[0]
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    samples = [ s.strip() for s in re.split(r"<\|endoftext\|>", text) if s.strip() ]
    selected = random.sample(samples, num_sample)
    selected = [ s + "\n<|endoftext|>\n" for s in selected ]
    for i in range(num_sample):
        output_path = output_root / f"{dataset_name}_sample_{i}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(selected[i])


if __name__ == "__main__":
    sample_from_dataset(DATASET_PATH, OUTPUT_ROOT, 10)