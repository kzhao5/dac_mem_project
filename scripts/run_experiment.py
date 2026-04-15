#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.controllers import controller_from_name
from dac_mem.datasets import load_locomo, load_longmemeval, load_synthetic, maybe_download_dataset
from dac_mem.pipeline import run_controller_on_examples
from dac_mem.utils import dump_json, ensure_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['synthetic', 'locomo', 'longmemeval'], default='synthetic')
    p.add_argument('--data_path', type=str, default='data/synthetic/longmemeval_mini.json')
    p.add_argument('--variant', type=str, default='s_cleaned', choices=['s_cleaned', 'm_cleaned', 'oracle'])
    p.add_argument('--download', action='store_true')
    p.add_argument('--controller', default='dac_mem')
    p.add_argument('--top_k', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--reader_mode', choices=['none', 'heuristic', 'hf_extractive'], default='none')
    p.add_argument('--reader_model', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='results')
    args = p.parse_args()

    data_path = Path(args.data_path)
    if args.download:
        data_path = maybe_download_dataset(args.dataset, data_path, variant=args.variant)

    if args.dataset == 'synthetic':
        examples = load_synthetic(data_path)
    elif args.dataset == 'locomo':
        examples = load_locomo(data_path)
    else:
        examples = load_longmemeval(data_path)

    controller = controller_from_name(args.controller)
    result = run_controller_on_examples(
        controller=controller,
        examples=examples,
        dataset_name=args.dataset,
        top_k=args.top_k,
        reader_mode=args.reader_mode,
        reader_model=args.reader_model,
        limit=args.limit,
    )

    out_dir = ensure_dir(args.output_dir)
    out_path = out_dir / f'{args.dataset}_{controller.name}.json'
    dump_json({
        'controller': result.controller,
        'dataset': result.dataset,
        'n_examples': result.n_examples,
        'metrics': result.metrics,
        'per_example': result.per_example,
    }, out_path)
    print(f'Saved results to {out_path}')
    print(result.metrics)


if __name__ == '__main__':
    main()
