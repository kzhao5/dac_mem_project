"""
Analysis utilities for the paper:
  - Pareto frontier (retrieval quality vs memory size)
  - Error analysis / breakdown by question type
  - Case study generator
  - Scalability analysis (memory growth over sessions)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import EvaluationResult


# ═══════════════════════════════════════════════════════════════════════════
# Pareto frontier
# ═══════════════════════════════════════════════════════════════════════════

def pareto_frontier(results: List[EvaluationResult],
                    x_metric: str = 'persistent_size',
                    y_metric: str = 'turn_hit@k') -> List[Dict[str, Any]]:
    """Compute the Pareto frontier: controllers that are not dominated.

    A point (x, y) dominates (x', y') if x <= x' and y >= y'.
    Here x = memory size (lower is better), y = retrieval quality (higher is better).
    """
    points = []
    for r in results:
        points.append({
            'controller': r.controller,
            'x': r.metrics.get(x_metric, 0.0),
            'y': r.metrics.get(y_metric, 0.0),
            'metrics': r.metrics,
        })
    # sort by x ascending, y descending
    points.sort(key=lambda p: (p['x'], -p['y']))

    frontier: list = []
    best_y = -float('inf')
    for p in points:
        if p['y'] > best_y:
            frontier.append(p)
            best_y = p['y']
    return frontier


def pareto_table(results: List[EvaluationResult],
                 metrics: Optional[List[str]] = None) -> str:
    """Generate a markdown table comparing all controllers."""
    if not metrics:
        metrics = ['persistent_size', 'persistent_derivable_fraction',
                   'persistent_stale_mean', 'turn_hit@k', 'session_hit@k',
                   'turn_precision@k', 'qa_f1', 'judge_score']
    header = '| Controller | ' + ' | '.join(metrics) + ' |'
    sep = '|' + '|'.join(['---'] * (len(metrics) + 1)) + '|'
    rows = [header, sep]
    for r in results:
        vals = [f"{r.metrics.get(m, 0.0):.3f}" for m in metrics]
        rows.append(f"| {r.controller} | " + ' | '.join(vals) + ' |')
    return '\n'.join(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Error analysis by question type
# ═══════════════════════════════════════════════════════════════════════════

def error_breakdown(result: EvaluationResult) -> Dict[str, Dict[str, float]]:
    """Break down metrics by question_type."""
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for ex in result.per_example:
        qtype = ex.get('metadata', {}).get('question_type',
                ex.get('metadata', {}).get('category', 'unknown'))
        merged: Dict[str, float] = {}
        merged.update(ex.get('retrieval', {}))
        merged.update(ex.get('memory', {}))
        if 'qa' in ex:
            merged.update(ex['qa'])
        by_type[qtype].append(merged)

    breakdown: Dict[str, Dict[str, float]] = {}
    for qtype, rows in by_type.items():
        keys = sorted({k for d in rows for k in d})
        avg = {k: float(np.mean([d.get(k, 0.0) for d in rows])) for k in keys}
        avg['count'] = float(len(rows))
        breakdown[qtype] = avg
    return breakdown


# ═══════════════════════════════════════════════════════════════════════════
# Case study generator
# ═══════════════════════════════════════════════════════════════════════════

def find_case_studies(results_a: EvaluationResult,
                      results_b: EvaluationResult,
                      metric: str = 'turn_hit@k',
                      n: int = 5) -> List[Dict[str, Any]]:
    """Find examples where controller A beats controller B the most.

    Useful for paper case studies: "DAC-Mem gets this right but Store-All doesn't."
    """
    a_by_id = {ex['example_id']: ex for ex in results_a.per_example}
    b_by_id = {ex['example_id']: ex for ex in results_b.per_example}
    common_ids = set(a_by_id) & set(b_by_id)

    diffs: list = []
    for eid in common_ids:
        a_val = a_by_id[eid].get('retrieval', {}).get(metric, 0.0)
        b_val = b_by_id[eid].get('retrieval', {}).get(metric, 0.0)
        if 'qa' in a_by_id[eid]:
            a_val = a_by_id[eid]['qa'].get(metric, a_val)
        if 'qa' in b_by_id[eid]:
            b_val = b_by_id[eid]['qa'].get(metric, b_val)
        diffs.append({
            'example_id': eid,
            'a_score': a_val, 'b_score': b_val,
            'diff': a_val - b_val,
            'a_example': a_by_id[eid],
            'b_example': b_by_id[eid],
        })
    diffs.sort(key=lambda x: x['diff'], reverse=True)
    return diffs[:n]


def format_case_study(case: Dict[str, Any]) -> str:
    """Format a single case study for inclusion in the paper."""
    lines = [
        f"### Example: {case['example_id']}",
        f"Question: {case['a_example']['question']}",
        f"Gold answer: {case['a_example']['gold_answer']}",
        f"",
        f"**{case['a_example']['controller']}** (score={case['a_score']:.2f}):",
        f"  Prediction: {case['a_example'].get('prediction', 'N/A')}",
        f"  Persistent memory size: {case['a_example']['memory']['persistent_size']}",
        f"",
        f"**{case['b_example']['controller']}** (score={case['b_score']:.2f}):",
        f"  Prediction: {case['b_example'].get('prediction', 'N/A')}",
        f"  Persistent memory size: {case['b_example']['memory']['persistent_size']}",
    ]
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Scalability analysis
# ═══════════════════════════════════════════════════════════════════════════

def memory_growth_curve(result: EvaluationResult) -> Dict[str, List[float]]:
    """Track how persistent memory size grows across examples.

    For plotting: x = number of processed examples, y = avg persistent size.
    """
    sizes = [ex['memory']['persistent_size']
             for ex in result.per_example]
    cumulative_avg = [float(np.mean(sizes[:i+1])) for i in range(len(sizes))]
    return {
        'controller': result.controller,
        'per_example_size': sizes,
        'cumulative_avg': cumulative_avg,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX table export
# ═══════════════════════════════════════════════════════════════════════════

def results_to_latex(results: List[EvaluationResult],
                     metrics: Optional[List[str]] = None,
                     caption: str = 'Main results',
                     label: str = 'tab:main') -> str:
    """Export results as a LaTeX table for the paper."""
    if not metrics:
        metrics = ['persistent_size', 'persistent_derivable_fraction',
                   'turn_hit@k', 'session_hit@k', 'qa_f1', 'judge_score']
    short = {
        'persistent_size': 'Mem Size',
        'persistent_derivable_fraction': 'Deriv \\%',
        'persistent_stale_mean': 'Stale',
        'turn_hit@k': 'T-Hit',
        'session_hit@k': 'S-Hit',
        'turn_precision@k': 'T-Prec',
        'qa_f1': 'F1',
        'qa_em': 'EM',
        'judge_score': 'Judge',
    }
    cols = ' & '.join(short.get(m, m) for m in metrics)
    lines = [
        f'\\begin{{table}}[t]',
        f'\\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        f'\\begin{{tabular}}{{l{"c" * len(metrics)}}}',
        f'\\toprule',
        f'Controller & {cols} \\\\',
        f'\\midrule',
    ]
    # find best value per metric
    best = {}
    for m in metrics:
        vals = [r.metrics.get(m, 0.0) for r in results]
        if m == 'persistent_size':
            best[m] = min(vals) if vals else 0
        else:
            best[m] = max(vals) if vals else 0

    for r in results:
        vals = []
        for m in metrics:
            v = r.metrics.get(m, 0.0)
            s = f'{v:.2f}'
            if abs(v - best[m]) < 1e-6:
                s = f'\\textbf{{{s}}}'
            vals.append(s)
        name = r.controller.replace('_', '\\_')
        lines.append(f'{name} & {" & ".join(vals)} \\\\')

    lines += [
        f'\\bottomrule',
        f'\\end{{tabular}}',
        f'\\end{{table}}',
    ]
    return '\n'.join(lines)


def save_analysis(results: List[EvaluationResult],
                  output_dir: str = 'results/analysis') -> None:
    """Save all analysis outputs to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Pareto
    frontier = pareto_frontier(results)
    (out / 'pareto_frontier.json').write_text(
        json.dumps(frontier, indent=2, default=str))

    # Markdown table
    (out / 'comparison_table.md').write_text(pareto_table(results))

    # LaTeX table
    (out / 'main_table.tex').write_text(results_to_latex(results))

    # Error breakdowns
    for r in results:
        bd = error_breakdown(r)
        (out / f'error_breakdown_{r.controller}.json').write_text(
            json.dumps(bd, indent=2, default=str))

    # Growth curves
    growth = {r.controller: memory_growth_curve(r) for r in results}
    (out / 'memory_growth.json').write_text(
        json.dumps(growth, indent=2, default=str))
