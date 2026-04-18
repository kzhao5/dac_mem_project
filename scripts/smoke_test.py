#!/usr/bin/env python3
"""Quick smoke test — runs on synthetic data, no GPU, no API, no internet."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.embedding import get_embedder
from dac_mem.controllers import StoreAllController, MemJudgeController
from dac_mem.datasets import load_synthetic
from dac_mem.pipeline import run_controller_on_examples


def main():
    embedder = get_embedder('hashing')
    data_path = Path(__file__).resolve().parents[1] / 'data' / 'synthetic' / 'longmemeval_mini.json'
    examples = load_synthetic(data_path)

    for Ctrl in [StoreAllController, MemJudgeController]:
        ctrl = Ctrl(embedder=embedder)
        res = run_controller_on_examples(
            ctrl, examples, 'synthetic', top_k=5, embedder=embedder,
        )
        print(ctrl.name, res.metrics)

    print('\nSmoke test passed.')


if __name__ == '__main__':
    main()
