#!/usr/bin/env python3
"""
make_demo_data.py — Generate small synthetic FASTQ for testing
"""

import random
from pathlib import Path

def make_fastq(path: Path, n_reads: int = 500, read_len: int = 50, seed: int = 42):
    random.seed(seed)
    lines = []
    for i in range(n_reads):
        seq = "".join(random.choices("ACGT", k=read_len))
        qual = "".join(chr(random.randint(53, 73)) for _ in range(read_len))
        lines.append(f"@demo_read_{i+1}")
        lines.append(seq)
        lines.append("+")
        lines.append(qual)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {n_reads} reads → {path}")

if __name__ == "__main__":
    make_fastq(Path("demo_data/demo_reads.fastq"))