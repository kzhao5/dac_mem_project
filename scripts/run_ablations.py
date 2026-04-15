#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tabulate import tabulate

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.controllers import ControllerConfig, DACMemController
from dac_mem.datasets import load_locomo, load_longmemeval, load_synthetic, maybe_download_dataset
from dac_mem.pipeline import run_controller_on_examples
from dac_mem.utils import dump_json, ensure_dir


def build_ablations():
    base = ControllerConfig(name='dac_mem')
    return [
        ('full', DACMemController(base)),
        ('no_derivability', DACMemController(ControllerConfig(name='dac_no_deriv', derivability_weight=0.0, stale_weight=0.75, persist_threshold=0.55, ephemeral_threshold=0.58))),
        ('no_stale', DACMemController(ControllerConfig(name='dac_no_stale', derivability_weight=1.0, stale_weight=0.0, persist_threshold=0.55, ephemeral_threshold=0.58))),
        ('binary_no_ephemeral', DACMemController(ControllerConfig(name='dac_binary', derivability_weight=1.0, stale_weight=0.75, persist_threshold=0.55, ephemeral_threshold=9.99))),
        ('weak_probe', DACMemController(ControllerConfig(name='dac_weakprobe', derivability_weight=0.5, stale_weight=0.75, persist_threshold=0.55, ephemeral_threshold=0.58))),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['synthetic', 'locomo', 'longmemeval'], default='synthetic')
    p.add_argument('--data_path', type=str, default='data/synthetic/longmemeval_mini.json')
    p.add_argument('--variant', type=str, default='s_cleaned', choices=['s_cleaned', 'm_cleaned', 'oracle'])
    p.add_argument('--download', action='store_true')
    p.add_argument('--top_k', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
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
    for name, controller in build_ablations():
        res = run_controller_on_examples(controller, examples, args.dataset, top_k=args.top_k, reader_mode='none', limit=args.limit)
        all_results.append({'ablation': name, 'metrics': res.metrics})
        rows.append({
            'ablation': name,
            'turn_hit@k': round(res.metrics.get('turn_hit@k', 0.0), 4),
            'session_hit@k': round(res.metrics.get('session_hit@k', 0.0), 4),
            'persistent_size': round(res.metrics.get('persistent_size', 0.0), 2),
            'persist_derivable': round(res.metrics.get('persistent_derivable_fraction', 0.0), 4),
            'persist_stale': round(res.metrics.get('persistent_stale_mean', 0.0), 4),
        })
    out_path = out_dir / f'{args.dataset}_ablation_results.json'
    dump_json(all_results, out_path)
    print(tabulate(rows, headers='keys', tablefmt='github'))
    print(f'Ablation results written to {out_path}')


if __name__ == '__main__':
    main()
