#!/usr/bin/env python3
"""Run a single controller on a dataset with full LLM support."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.controllers import controller_from_name
from dac_mem.datasets import load_dataset, maybe_download_dataset
from dac_mem.embedding import get_embedder
from dac_mem.llm import get_llm
from dac_mem.pipeline import run_controller_on_examples
from dac_mem.utils import dump_json, ensure_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='synthetic',
                   choices=['synthetic', 'locomo', 'longmemeval', 'perma', 'memorybench'])
    p.add_argument('--data_path', type=str, default=None)
    p.add_argument('--variant', default='s_cleaned')
    p.add_argument('--download', action='store_true')
    p.add_argument('--controller', default='dac_mem')
    p.add_argument('--top_k', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    # LLM
    p.add_argument('--llm_provider', default=None,
                   help='LLM provider: openai|anthropic|google|qwen|llama|vllm')
    p.add_argument('--llm_model', default=None)
    p.add_argument('--judge_provider', default=None)
    p.add_argument('--judge_model', default=None)
    # Reader
    p.add_argument('--reader_mode', default='heuristic',
                   choices=['none', 'heuristic', 'hf_extractive', 'llm'])
    p.add_argument('--reader_model', default=None)
    # Embedding
    p.add_argument('--embedding', default='hashing',
                   help='Embedding model: bge-large|minilm|gte-large|openai|hashing')
    p.add_argument('--device', default='cpu')
    # Probe
    p.add_argument('--probe_mode', default='embedding',
                   choices=['embedding', 'llm_judge', 'tool_grounded'])
    p.add_argument('--use_llm_extractor', action='store_true')
    p.add_argument('--output_dir', default='results')
    args = p.parse_args()

    # data
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = Path(f'data/synthetic/longmemeval_mini.json')
    if args.download:
        data_path = maybe_download_dataset(args.dataset, str(data_path),
                                           variant=args.variant)
    examples = load_dataset(args.dataset, data_path)

    # embedding
    embedder = get_embedder(args.embedding, device=args.device)

    # LLM
    llm = None
    if args.llm_provider:
        llm = get_llm(args.llm_provider, args.llm_model)

    judge_llm = None
    if args.judge_provider:
        judge_llm = get_llm(args.judge_provider, args.judge_model)

    # controller
    ctrl = controller_from_name(args.controller, llm=llm, embedder=embedder)
    if hasattr(ctrl, 'config') and hasattr(ctrl.config, 'probe_mode'):
        ctrl.config.probe_mode = args.probe_mode

    result = run_controller_on_examples(
        controller=ctrl,
        examples=examples,
        dataset_name=args.dataset,
        top_k=args.top_k,
        reader_mode=args.reader_mode,
        reader_model=args.reader_model,
        llm=llm,
        embedder=embedder,
        judge_llm=judge_llm,
        use_llm_extractor=args.use_llm_extractor,
        limit=args.limit,
    )

    out_dir = ensure_dir(args.output_dir)
    out_path = out_dir / f'{args.dataset}_{ctrl.name}.json'
    dump_json({
        'controller': result.controller,
        'dataset': result.dataset,
        'n_examples': result.n_examples,
        'metrics': result.metrics,
        'per_example': result.per_example,
    }, out_path)
    print(f'Saved: {out_path}')
    for k, v in sorted(result.metrics.items()):
        print(f'  {k}: {v:.4f}')


if __name__ == '__main__':
    main()
