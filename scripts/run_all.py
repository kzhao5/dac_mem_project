#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tabulate import tabulate

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.controllers import AMACLiteController, DACMemController, NoveltyRecencyController, RelevanceOnlyController, StoreAllController
from dac_mem.datasets import load_locomo, load_longmemeval, load_synthetic, maybe_download_dataset
from dac_mem.pipeline import run_controller_on_examples
from dac_mem.utils import dump_json, ensure_dir


def build_controllers():
    return [
        StoreAllController(),
        RelevanceOnlyController(),
        NoveltyRecencyController(),
        AMACLiteController(),
        DACMemController(),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['synthetic', 'locomo', 'longmemeval'], default='synthetic')
    p.add_argument('--data_path', type=str, default='data/synthetic/longmemeval_mini.json')
    p.add_argument('--variant', type=str, default='s_cleaned', choices=['s_cleaned', 'm_cleaned', 'oracle'])
    p.add_argument('--download', action='store_true')
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

    out_dir = ensure_dir(args.output_dir)
    rows = []
    all_results = []
    for controller in build_controllers():
        res = run_controller_on_examples(
            controller=controller,
            examples=examples,
            dataset_name=args.dataset,
            top_k=args.top_k,
            reader_mode=args.reader_mode,
            reader_model=args.reader_model,
            limit=args.limit,
        )
        all_results.append({
            'controller': res.controller,
            'metrics': res.metrics,
        })
        row = {'controller': res.controller}
        for k in ['turn_hit@k', 'session_hit@k', 'turn_precision@k', 'session_precision@k', 'persistent_size', 'persistent_derivable_fraction', 'persistent_stale_mean', 'qa_f1']:
            if k in res.metrics:
                row[k] = round(res.metrics[k], 4)
        rows.append(row)

    out_path = out_dir / f'{args.dataset}_all_results.json'
    dump_json(all_results, out_path)
    print(tabulate(rows, headers='keys', tablefmt='github'))
    print(f'Full results written to {out_path}')


if __name__ == '__main__':
    main()
