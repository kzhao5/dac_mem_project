#!/usr/bin/env python3
"""Quick smoke test — runs on synthetic data with hashing embeddings (no GPU, no API)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.embedding import get_embedder
from dac_mem.controllers import StoreAllController, DACMemController
from dac_mem.datasets import load_synthetic
from dac_mem.pipeline import run_controller_on_examples


def main():
    # use hashing embedder for smoke test (no model download needed)
    embedder = get_embedder('hashing')
    data_path = Path(__file__).resolve().parents[1] / 'data' / 'synthetic' / 'longmemeval_mini.json'
    examples = load_synthetic(data_path)

    for Ctrl in [StoreAllController, DACMemController]:
        ctrl = Ctrl(embedder=embedder)
        res = run_controller_on_examples(
            ctrl, examples, 'synthetic',
            top_k=5, reader_mode='heuristic', embedder=embedder,
        )
        print(ctrl.name, res.metrics)

    print('\nSmoke test passed.')


if __name__ == '__main__':
    main()
