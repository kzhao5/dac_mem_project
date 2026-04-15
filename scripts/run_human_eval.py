#!/usr/bin/env python3
"""Generate human evaluation annotation sheets and compute agreement."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from dac_mem.human_eval import (
    aggregate_annotations, generate_annotation_sheet,
    load_annotations, sample_for_annotation,
)
from dac_mem.schema import EvaluationResult


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command')

    # generate
    gen = sub.add_parser('generate', help='Generate annotation sheets')
    gen.add_argument('--results', required=True, help='Path to results JSON')
    gen.add_argument('--n', type=int, default=100)
    gen.add_argument('--output', default='results/human_eval/annotation_sheet.tsv')
    gen.add_argument('--format', default='tsv', choices=['tsv', 'json'])

    # aggregate
    agg = sub.add_parser('aggregate', help='Compute agreement from completed annotations')
    agg.add_argument('--annotations', required=True, help='Path to completed annotation file')

    args = p.parse_args()

    if args.command == 'generate':
        raw = json.loads(Path(args.results).read_text())
        result = EvaluationResult(
            controller=raw.get('controller', 'unknown'),
            dataset=raw.get('dataset', 'unknown'),
            n_examples=raw.get('n_examples', 0),
            metrics=raw.get('metrics', {}),
            per_example=raw.get('per_example', []),
        )
        items = sample_for_annotation(result, n=args.n)
        out = generate_annotation_sheet(items, args.output, format=args.format)
        print(f'Generated annotation sheet: {out} ({len(items)} items)')

    elif args.command == 'aggregate':
        records = load_annotations(args.annotations)
        stats = aggregate_annotations(records)
        print(json.dumps(stats, indent=2))

    else:
        p.print_help()


if __name__ == '__main__':
    main()
