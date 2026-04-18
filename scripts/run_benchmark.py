#!/usr/bin/env python3
"""
Main benchmark: compare all controllers using a local LLM.
Runs fully offline — no network requests.

Usage:
    # On GPU node (via SLURM):
    PYTHONPATH=src python scripts/run_benchmark.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset synthetic \
        --device cuda

    # Quick smoke test (CPU, no LLM):
    PYTHONPATH=src python scripts/run_benchmark.py \
        --dataset synthetic --no_llm
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tabulate import tabulate

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

# Disable network access for HF
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from dac_mem.analysis import results_to_latex, save_analysis
from dac_mem.baselines import MemoryBankController, MemoryR1Controller
from dac_mem.controllers import (
    AMACLiteController, MemJudgeController, NoveltyRecencyController,
    RelevanceOnlyController, StoreAllController,
)
from dac_mem.datasets import load_dataset
from dac_mem.embedding import get_embedder
from dac_mem.evaluation import paired_bootstrap_test
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
    ]
    if llm is not None:
        ctrls.append(MemoryR1Controller(llm, **kw))
    ctrls.append(MemJudgeController(llm=llm, **kw))
    return ctrls


def main():
    p = argparse.ArgumentParser(description='MemJudge benchmark')
    p.add_argument('--model', type=str, default=None,
                   help='HuggingFace model name (e.g. Qwen/Qwen2.5-7B-Instruct)')
    p.add_argument('--backend', default='huggingface',
                   choices=['huggingface', 'vllm'],
                   help='Inference backend')
    p.add_argument('--dataset', default='synthetic',
                   choices=['synthetic', 'locomo', 'longmemeval'])
    p.add_argument('--data_path', default=None,
                   help='Path to dataset file')
    p.add_argument('--embedding', default='bge-large',
                   help='Embedding model name')
    p.add_argument('--device', default='cuda',
                   help='Device for models (cuda/cpu)')
    p.add_argument('--top_k', type=int, default=8)
    p.add_argument('--limit', type=int, default=None,
                   help='Max examples to evaluate')
    p.add_argument('--no_llm', action='store_true',
                   help='Run without LLM (embedding-only mode)')
    p.add_argument('--output_dir', default='results')
    p.add_argument('--load_in_4bit', action='store_true',
                   help='Load LLM in 4-bit quantisation')
    args = p.parse_args()

    # ── Data ───────────────────────────────────────────────────────────────
    data_paths = {
        'synthetic': 'data/synthetic/longmemeval_mini.json',
        'locomo': 'data/locomo/locomo10.json',
        'longmemeval': 'data/longmemeval/longmemeval_s_cleaned.json',
    }
    data_path = Path(args.data_path or data_paths[args.dataset])
    if not data_path.exists():
        print(f'ERROR: {data_path} not found. Run scripts/download.py first.')
        sys.exit(1)
    examples = load_dataset(args.dataset, data_path)
    print(f'Loaded {len(examples)} examples from {args.dataset}')

    # ── Embedding ──────────────────────────────────────────────────────────
    embedder = get_embedder(args.embedding, device=args.device)

    # ── LLM ────────────────────────────────────────────────────────────────
    llm = None
    judge_llm = None
    model_short = 'no_llm'

    if not args.no_llm and args.model:
        print(f'Loading LLM: {args.model} ({args.backend})...')
        llm_kw = {}
        if args.load_in_4bit:
            llm_kw['load_in_4bit'] = True
        llm = get_llm(args.backend, args.model,
                       device=args.device, **llm_kw)
        judge_llm = llm  # use same model as judge
        model_short = args.model.split('/')[-1]
        print(f'LLM loaded: {model_short}')

    # ── Run all controllers ────────────────────────────────────────────────
    out_dir = ensure_dir(f'{args.output_dir}/{args.dataset}/{model_short}')
    all_results = []
    rows = []

    for ctrl in build_controllers(llm=llm, embedder=embedder):
        print(f'\nRunning: {ctrl.name}...')
        res = run_controller_on_examples(
            controller=ctrl, examples=examples,
            dataset_name=args.dataset, top_k=args.top_k,
            llm=llm, embedder=embedder, judge_llm=judge_llm,
            limit=args.limit,
        )
        all_results.append(res)

        row = {'controller': res.controller}
        for k in ['persistent_size', 'persistent_derivable_fraction',
                   'persistent_stale_mean', 'turn_hit@k', 'session_hit@k',
                   'turn_precision@k', 'qa_f1', 'judge_score']:
            if k in res.metrics:
                row[k] = round(res.metrics[k], 4)
        rows.append(row)

    # ── Output ─────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  Results: {args.dataset} | Model: {model_short}')
    print(f'{"="*60}')
    print(tabulate(rows, headers='keys', tablefmt='github'))

    # Save results
    dump_json([{'controller': r.controller, 'metrics': r.metrics}
               for r in all_results],
              out_dir / 'results.json')
    save_analysis(all_results, str(out_dir / 'analysis'))

    # LaTeX
    latex = results_to_latex(
        all_results,
        caption=f'Results on {args.dataset} ({model_short})',
        label=f'tab:{args.dataset}_{model_short}',
    )
    (out_dir / 'table.tex').write_text(latex)
    print(f'\nResults saved to {out_dir}/')

    # Significance tests (MemJudge vs others)
    if len(all_results) > 1:
        print(f'\n--- Significance (MemJudge vs baselines) ---')
        dac_result = next((r for r in all_results if r.controller == 'memjudge'), None)
        if dac_result:
            for metric in ['turn_hit@k', 'qa_f1']:
                dac_vals = [ex.get('retrieval', {}).get(metric,
                            ex.get('qa', {}).get(metric, 0.0))
                            for ex in dac_result.per_example]
                for other in all_results:
                    if other.controller == 'memjudge':
                        continue
                    other_vals = [ex.get('retrieval', {}).get(metric,
                                  ex.get('qa', {}).get(metric, 0.0))
                                  for ex in other.per_example]
                    if len(dac_vals) == len(other_vals) and len(dac_vals) > 1:
                        test = paired_bootstrap_test(dac_vals, other_vals)
                        sig = '**' if test['significant_at_01'] else \
                              '*' if test['significant_at_05'] else ''
                        print(f'  vs {other.controller} ({metric}): '
                              f'diff={test["observed_diff"]:.4f} '
                              f'p={test["p_value"]:.4f} {sig}')


if __name__ == '__main__':
    main()
