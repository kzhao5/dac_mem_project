#!/usr/bin/env python3
"""
Full EMNLP benchmark run:
  - All controllers × all datasets
  - With LLM reader + LLM judge
  - Significance tests
  - Analysis artefacts (Pareto, LaTeX, error breakdown)

Usage:
  python scripts/run_full_benchmark.py \
    --llm_provider openai --llm_model gpt-4o \
    --judge_provider openai --judge_model gpt-4o \
    --embedding bge-large --device cuda \
    --probe_mode tool_grounded \
    --datasets locomo longmemeval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabulate import tabulate

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.analysis import save_analysis, results_to_latex
from dac_mem.baselines import MemoryBankController, MemoryR1Controller
from dac_mem.controllers import (
    AMACLiteController, DACMemController, ControllerConfig,
    NoveltyRecencyController, RelevanceOnlyController, StoreAllController,
)
from dac_mem.datasets import load_dataset, maybe_download_dataset
from dac_mem.embedding import get_embedder
from dac_mem.evaluation import paired_bootstrap_test, significance_matrix
from dac_mem.llm import get_llm
from dac_mem.pipeline import run_controller_on_examples
from dac_mem.utils import dump_json, ensure_dir


def build_all_controllers(llm=None, embedder=None, probe_mode='embedding'):
    kw = {'embedder': embedder} if embedder else {}
    cfg = ControllerConfig(name='dac_mem', probe_mode=probe_mode)
    ctrls = [
        StoreAllController(**kw),
        RelevanceOnlyController(**kw),
        NoveltyRecencyController(**kw),
        AMACLiteController(**kw),
        MemoryBankController(**kw),
        DACMemController(config=cfg, llm=llm, **kw),
    ]
    if llm is not None:
        ctrls.insert(-1, MemoryR1Controller(llm, **kw))
    return ctrls


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['locomo', 'longmemeval'],
                   choices=['synthetic', 'locomo', 'longmemeval', 'perma', 'memorybench'])
    p.add_argument('--data_dir', default='data')
    p.add_argument('--download', action='store_true')
    p.add_argument('--top_k', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    # LLM
    p.add_argument('--llm_provider', default='openai')
    p.add_argument('--llm_model', default='gpt-4o')
    p.add_argument('--judge_provider', default='openai')
    p.add_argument('--judge_model', default='gpt-4o')
    # Embedding
    p.add_argument('--embedding', default='bge-large')
    p.add_argument('--device', default='cpu')
    # Probe
    p.add_argument('--probe_mode', default='tool_grounded',
                   choices=['embedding', 'llm_judge', 'tool_grounded'])
    p.add_argument('--use_llm_extractor', action='store_true', default=True)
    p.add_argument('--output_dir', default='results/full_benchmark')
    args = p.parse_args()

    embedder = get_embedder(args.embedding, device=args.device)
    llm = get_llm(args.llm_provider, args.llm_model)
    judge_llm = get_llm(args.judge_provider, args.judge_model)
    out_dir = ensure_dir(args.output_dir)

    # data paths
    data_paths = {
        'synthetic': 'data/synthetic/longmemeval_mini.json',
        'locomo': f'{args.data_dir}/locomo/locomo10.json',
        'longmemeval': f'{args.data_dir}/longmemeval/longmemeval_s_cleaned.json',
        'perma': f'{args.data_dir}/perma/perma.json',
        'memorybench': f'{args.data_dir}/memorybench/memorybench.json',
    }

    all_dataset_results = {}

    for ds_name in args.datasets:
        print(f'\n{"="*60}')
        print(f'  Dataset: {ds_name}')
        print(f'{"="*60}')

        data_path = Path(data_paths[ds_name])
        if args.download and ds_name != 'synthetic':
            data_path = maybe_download_dataset(ds_name, str(data_path))
        examples = load_dataset(ds_name, data_path)

        ds_results = []
        rows = []

        for ctrl in build_all_controllers(
            llm=llm, embedder=embedder, probe_mode=args.probe_mode
        ):
            print(f'\n  Running: {ctrl.name} on {ds_name}...')
            res = run_controller_on_examples(
                controller=ctrl, examples=examples,
                dataset_name=ds_name, top_k=args.top_k,
                reader_mode='llm', llm=llm,
                embedder=embedder, judge_llm=judge_llm,
                use_llm_extractor=args.use_llm_extractor,
                limit=args.limit,
            )
            ds_results.append(res)
            row = {'controller': res.controller}
            for k in ['persistent_size', 'persistent_derivable_fraction',
                       'turn_hit@k', 'session_hit@k', 'qa_f1', 'judge_score']:
                if k in res.metrics:
                    row[k] = round(res.metrics[k], 3)
            rows.append(row)

        print(f'\n--- Results: {ds_name} ---')
        print(tabulate(rows, headers='keys', tablefmt='github'))

        # save per-dataset results
        ds_out = ensure_dir(f'{args.output_dir}/{ds_name}')
        dump_json([{'controller': r.controller, 'metrics': r.metrics}
                    for r in ds_results],
                  ds_out / 'all_results.json')
        save_analysis(ds_results, str(ds_out / 'analysis'))
        all_dataset_results[ds_name] = ds_results

        # significance tests
        print(f'\n--- Significance tests: {ds_name} ---')
        for metric in ['turn_hit@k', 'qa_f1']:
            scores = {}
            for r in ds_results:
                vals = [ex.get('retrieval', {}).get(metric,
                        ex.get('qa', {}).get(metric, 0.0))
                        for ex in r.per_example]
                scores[r.controller] = vals
            if 'dac_mem' in scores:
                for other in scores:
                    if other != 'dac_mem':
                        test = paired_bootstrap_test(
                            scores['dac_mem'], scores[other])
                        sig = '*' if test['significant_at_05'] else ''
                        print(f'  DAC-Mem vs {other} ({metric}): '
                              f'diff={test["observed_diff"]:.4f} '
                              f'p={test["p_value"]:.4f} {sig}')

    # combined LaTeX
    print('\n\n--- Combined LaTeX tables ---')
    for ds_name, results in all_dataset_results.items():
        print(f'\n% {ds_name}')
        print(results_to_latex(results, caption=f'Results on {ds_name}',
                               label=f'tab:{ds_name}'))


if __name__ == '__main__':
    main()
