#!/usr/bin/env python3
"""
Ablation study for DAC-Mem (Section 9 of report).

Ablations:
  1. full             — all components enabled
  2. no_derivability   — derivability_weight = 0
  3. no_stale          — stale_weight = 0
  4. binary            — disable ephemeral pathway
  5. weak_probe        — derivability_weight = 0.5
  6. llm_judge_only    — use LLM judge probe instead of tool-grounded
  7. no_llm_extractor  — rule-based type classification only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabulate import tabulate

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.controllers import ControllerConfig, DACMemController
from dac_mem.datasets import load_dataset, maybe_download_dataset
from dac_mem.embedding import get_embedder
from dac_mem.llm import get_llm
from dac_mem.pipeline import run_controller_on_examples
from dac_mem.utils import dump_json, ensure_dir


def build_ablations(llm=None, embedder=None):
    kw = {}
    if embedder:
        kw['embedder'] = embedder

    return [
        ('full', DACMemController(
            ControllerConfig(name='dac_mem', probe_mode='embedding'), llm=llm, **kw)),
        ('no_derivability', DACMemController(
            ControllerConfig(name='dac_no_deriv', derivability_weight=0.0,
                             stale_weight=0.75, probe_mode='embedding'), llm=llm, **kw)),
        ('no_stale', DACMemController(
            ControllerConfig(name='dac_no_stale', derivability_weight=1.0,
                             stale_weight=0.0, probe_mode='embedding'), llm=llm, **kw)),
        ('binary_no_ephemeral', DACMemController(
            ControllerConfig(name='dac_binary', derivability_weight=1.0,
                             stale_weight=0.75, ephemeral_threshold=9.99,
                             probe_mode='embedding'), llm=llm, **kw)),
        ('weak_probe', DACMemController(
            ControllerConfig(name='dac_weakprobe', derivability_weight=0.5,
                             stale_weight=0.75, probe_mode='embedding'), llm=llm, **kw)),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='synthetic',
                   choices=['synthetic', 'locomo', 'longmemeval', 'perma', 'memorybench'])
    p.add_argument('--data_path', default='data/synthetic/longmemeval_mini.json')
    p.add_argument('--variant', default='s_cleaned')
    p.add_argument('--download', action='store_true')
    p.add_argument('--top_k', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--llm_provider', default=None)
    p.add_argument('--llm_model', default=None)
    p.add_argument('--embedding', default='hashing')
    p.add_argument('--device', default='cpu')
    p.add_argument('--reader_mode', default='heuristic',
                   choices=['none', 'heuristic', 'hf_extractive', 'llm'])
    p.add_argument('--output_dir', default='results')
    args = p.parse_args()

    data_path = Path(args.data_path)
    if args.download:
        data_path = maybe_download_dataset(args.dataset, str(data_path),
                                           variant=args.variant)
    examples = load_dataset(args.dataset, data_path)
    embedder = get_embedder(args.embedding, device=args.device)
    llm = get_llm(args.llm_provider, args.llm_model) if args.llm_provider else None

    out_dir = ensure_dir(args.output_dir)
    rows = []
    all_results = []

    for name, ctrl in build_ablations(llm=llm, embedder=embedder):
        res = run_controller_on_examples(
            ctrl, examples, args.dataset, top_k=args.top_k,
            reader_mode=args.reader_mode, llm=llm, embedder=embedder,
            limit=args.limit,
        )
        all_results.append({'ablation': name, 'metrics': res.metrics})
        rows.append({
            'ablation': name,
            'turn_hit@k': round(res.metrics.get('turn_hit@k', 0), 4),
            'session_hit@k': round(res.metrics.get('session_hit@k', 0), 4),
            'persistent_size': round(res.metrics.get('persistent_size', 0), 2),
            'derivable_frac': round(res.metrics.get('persistent_derivable_fraction', 0), 4),
            'stale_mean': round(res.metrics.get('persistent_stale_mean', 0), 4),
            'qa_f1': round(res.metrics.get('qa_f1', 0), 4),
        })

    dump_json(all_results, out_dir / f'{args.dataset}_ablation_results.json')
    print(tabulate(rows, headers='keys', tablefmt='github'))


if __name__ == '__main__':
    main()
