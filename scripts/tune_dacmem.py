#!/usr/bin/env python3
"""Grid search for DAC-Mem hyperparameters."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabulate import tabulate

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.controllers import ControllerConfig, DACMemController
from dac_mem.datasets import load_dataset, maybe_download_dataset
from dac_mem.embedding import get_embedder
from dac_mem.pipeline import run_controller_on_examples

SEARCH_SPACE = {
    'persist_threshold': [0.45, 0.50, 0.55, 0.60],
    'ephemeral_threshold': [0.50, 0.55, 0.60, 0.65],
    'derivability_weight': [0.8, 1.0, 1.2],
    'stale_weight': [0.5, 0.75, 1.0],
}


def objective(metrics: dict) -> float:
    return (metrics.get('turn_hit@k', 0.0)
            + 0.5 * metrics.get('session_hit@k', 0.0)
            - 0.02 * metrics.get('persistent_size', 0.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='synthetic',
                   choices=['synthetic', 'locomo', 'longmemeval', 'perma', 'memorybench'])
    p.add_argument('--data_path', default='data/synthetic/longmemeval_mini.json')
    p.add_argument('--variant', default='s_cleaned')
    p.add_argument('--download', action='store_true')
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--embedding', default='hashing')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    data_path = Path(args.data_path)
    if args.download:
        data_path = maybe_download_dataset(args.dataset, str(data_path),
                                           variant=args.variant)
    examples = load_dataset(args.dataset, data_path)
    embedder = get_embedder(args.embedding, device=args.device)

    best_cfg = None
    best_score = -1e9
    rows = []

    for pt in SEARCH_SPACE['persist_threshold']:
        for et in SEARCH_SPACE['ephemeral_threshold']:
            for dw in SEARCH_SPACE['derivability_weight']:
                for sw in SEARCH_SPACE['stale_weight']:
                    cfg = ControllerConfig(
                        name='dac_mem', persist_threshold=pt,
                        ephemeral_threshold=et, derivability_weight=dw,
                        stale_weight=sw)
                    ctrl = DACMemController(cfg, embedder=embedder)
                    res = run_controller_on_examples(
                        ctrl, examples, args.dataset,
                        top_k=8, reader_mode='none',
                        embedder=embedder, limit=args.limit,
                    )
                    s = objective(res.metrics)
                    rows.append({
                        'persist_th': pt, 'ephemeral_th': et,
                        'deriv_w': dw, 'stale_w': sw,
                        'score': round(s, 4),
                        'turn_hit': round(res.metrics.get('turn_hit@k', 0), 4),
                        'memory': round(res.metrics.get('persistent_size', 0), 2),
                    })
                    if s > best_score:
                        best_score = s
                        best_cfg = cfg

    rows.sort(key=lambda x: x['score'], reverse=True)
    print(tabulate(rows[:15], headers='keys', tablefmt='github'))
    print(f'\nBest config: {best_cfg}')
    print(f'Best score: {best_score:.4f}')


if __name__ == '__main__':
    main()
