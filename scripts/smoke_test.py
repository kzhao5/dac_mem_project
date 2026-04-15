#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.controllers import StoreAllController, DACMemController
from dac_mem.datasets import load_synthetic
from dac_mem.pipeline import run_controller_on_examples


def main():
    data_path = Path(__file__).resolve().parents[1] / 'data' / 'synthetic' / 'longmemeval_mini.json'
    examples = load_synthetic(data_path)
    for controller in [StoreAllController(), DACMemController()]:
        res = run_controller_on_examples(controller, examples, 'synthetic', top_k=5, reader_mode='heuristic')
        print(controller.name, res.metrics)


if __name__ == '__main__':
    main()
