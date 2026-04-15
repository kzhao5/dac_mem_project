#!/usr/bin/env python3
"""Compare all controllers (baselines + DAC-Mem) on a dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabulate import tabulate

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.analysis import pareto_table, results_to_latex, save_analysis
from dac_mem.baselines import MemoryBankController, MemoryR1Controller
from dac_mem.controllers import (
    AMACLiteController, DACMemController, NoveltyRecencyController,
    RelevanceOnlyController, StoreAllController,
)
from dac_mem.datasets import load_dataset, maybe_download_dataset
from dac_mem.embedding import get_embedder
from dac_mem.llm import get_llm
from dac_mem.pipeline import run_controller_on_examples
from dac_mem.utils import dump_json, ensure_dir


def build_controllers(llm=None, embedder=None):
    kw = {'embedder': embedder} if embedder else {}
    ctrls = [
        StoreAllController(**kw),
        RelevanceOnlyController(**kw),
        NoveltyRecencyController(**kw),
        AMACLiteController(**kw),
        MemoryBankController(**kw),
        DACMemController(llm=llm, **kw),
    ]
    if llm is not None:
        ctrls.insert(-1, MemoryR1Controller(llm, **kw))
    return ctrls


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='synthetic',
                   choices=['synthetic', 'locomo', 'longmemeval', 'perma', 'memorybench'])
    p.add_argument('--data_path', default=None)
    p.add_argument('--variant', default='s_cleaned')
    p.add_argument('--download', action='store_true')
    p.add_argument('--top_k', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--llm_provider', default=None)
    p.add_argument('--llm_model', default=None)
    p.add_argument('--judge_provider', default=None)
    p.add_argument('--judge_model', default=None)
    p.add_argument('--reader_mode', default='heuristic',
                   choices=['none', 'heuristic', 'hf_extractive', 'llm'])
    p.add_argument('--embedding', default='hashing')
    p.add_argument('--device', default='cpu')
    p.add_argument('--probe_mode', default='embedding',
                   choices=['embedding', 'llm_judge', 'tool_grounded'])
    p.add_argument('--use_llm_extractor', action='store_true')
    p.add_argument('--output_dir', default='results')
    args = p.parse_args()

    data_path = Path(args.data_path) if args.data_path else Path(
        f'data/synthetic/longmemeval_mini.json')
    if args.download:
        data_path = maybe_download_dataset(args.dataset, str(data_path),
                                           variant=args.variant)
    examples = load_dataset(args.dataset, data_path)
    embedder = get_embedder(args.embedding, device=args.device)
    llm = get_llm(args.llm_provider, args.llm_model) if args.llm_provider else None
    judge_llm = get_llm(args.judge_provider, args.judge_model) if args.judge_provider else None

    out_dir = ensure_dir(args.output_dir)
    all_results = []
    rows = []

    for ctrl in build_controllers(llm=llm, embedder=embedder):
        if hasattr(ctrl, 'config') and hasattr(ctrl.config, 'probe_mode'):
            ctrl.config.probe_mode = args.probe_mode
        res = run_controller_on_examples(
            controller=ctrl, examples=examples,
            dataset_name=args.dataset, top_k=args.top_k,
            reader_mode=args.reader_mode, llm=llm,
            embedder=embedder, judge_llm=judge_llm,
            use_llm_extractor=args.use_llm_extractor,
            limit=args.limit,
        )
        all_results.append(res)
        row = {'controller': res.controller}
        for k in ['turn_hit@k', 'session_hit@k', 'turn_precision@k',
                   'persistent_size', 'persistent_derivable_fraction',
                   'persistent_stale_mean', 'qa_f1', 'judge_score']:
            if k in res.metrics:
                row[k] = round(res.metrics[k], 4)
        rows.append(row)

    dump_json([{'controller': r.controller, 'metrics': r.metrics}
               for r in all_results],
              out_dir / f'{args.dataset}_all_results.json')

    print(tabulate(rows, headers='keys', tablefmt='github'))
    print()

    # save analysis artefacts
    save_analysis(all_results, str(out_dir / 'analysis'))
    print(f'Analysis saved to {out_dir / "analysis"}')

    # LaTeX table
    print('\n--- LaTeX table ---')
    print(results_to_latex(all_results))


if __name__ == '__main__':
    main()
