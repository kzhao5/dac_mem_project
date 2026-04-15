"""
Human evaluation framework for DAC-Mem.

Supports:
  - Sampling examples for annotation
  - Generating annotation sheets (JSON + TSV)
  - Computing inter-annotator agreement (Cohen's kappa, Fleiss' kappa)
  - Aggregating human judgements
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import EvaluationResult


# ═══════════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════════

def sample_for_annotation(result: EvaluationResult,
                          n: int = 100,
                          stratify_by: str = 'memory_type',
                          seed: int = 42) -> List[Dict[str, Any]]:
    """Sample memory-write decisions for human annotation.

    Returns items from per-example results where annotators judge whether
    PERSIST / EPHEMERAL / SKIP was correct.
    """
    rng = random.Random(seed)
    all_items: list = []
    for ex in result.per_example:
        for r in ex.get('retrieved', []):
            all_items.append({
                'example_id': ex['example_id'],
                'question': ex['question'],
                'gold_answer': ex['gold_answer'],
                'controller': ex['controller'],
                'retrieved_text': r.get('text', ''),
                'action': r.get('metadata', {}).get('action', 'unknown'),
                'memory_type': r.get('metadata', {}).get('memory_type', 'unknown'),
                'source': r.get('source', 'unknown'),
            })
    if len(all_items) <= n:
        return all_items

    # stratified sampling
    by_stratum: Dict[str, list] = defaultdict(list)
    for item in all_items:
        key = item.get(stratify_by, 'unknown')
        by_stratum[key].append(item)

    sampled: list = []
    per_stratum = max(1, n // max(len(by_stratum), 1))
    for key, items in by_stratum.items():
        sampled.extend(rng.sample(items, min(per_stratum, len(items))))

    # fill remaining
    remaining = n - len(sampled)
    if remaining > 0:
        pool = [it for it in all_items if it not in sampled]
        sampled.extend(rng.sample(pool, min(remaining, len(pool))))

    return sampled[:n]


# ═══════════════════════════════════════════════════════════════════════════
# Annotation sheet generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_annotation_sheet(items: List[Dict[str, Any]],
                               output_path: str,
                               format: str = 'tsv') -> Path:
    """Generate annotation sheets for human evaluators.

    Each row: item text, controller decision, columns for annotator labels.
    Annotators mark each decision as: correct / incorrect / uncertain.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        records = []
        for i, item in enumerate(items):
            records.append({
                'id': i,
                'example_id': item['example_id'],
                'question': item['question'],
                'text': item['retrieved_text'],
                'controller_action': item['action'],
                'memory_type': item['memory_type'],
                # annotator fills these
                'annotator_1': '',
                'annotator_2': '',
                'annotator_3': '',
            })
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    else:  # TSV
        with open(out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([
                'id', 'example_id', 'question', 'text',
                'controller_action', 'memory_type',
                'annotator_1', 'annotator_2', 'annotator_3',
            ])
            for i, item in enumerate(items):
                writer.writerow([
                    i, item['example_id'], item['question'],
                    item['retrieved_text'], item['action'],
                    item['memory_type'], '', '', '',
                ])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Inter-annotator agreement
# ═══════════════════════════════════════════════════════════════════════════

def cohens_kappa(labels_a: Sequence[str],
                 labels_b: Sequence[str]) -> float:
    """Cohen's kappa for two annotators."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0

    cats = sorted(set(labels_a) | set(labels_b))
    # observed agreement
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # expected agreement
    pe = 0.0
    for c in cats:
        pa = sum(1 for x in labels_a if x == c) / n
        pb = sum(1 for x in labels_b if x == c) / n
        pe += pa * pb

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(annotations: List[List[str]]) -> float:
    """Fleiss' kappa for multiple annotators.

    annotations[i] = list of labels from all annotators for item i.
    """
    if not annotations:
        return 0.0

    n_items = len(annotations)
    n_raters = len(annotations[0])
    cats = sorted({label for row in annotations for label in row})
    n_cats = len(cats)
    cat_idx = {c: i for i, c in enumerate(cats)}

    # Count matrix
    counts = np.zeros((n_items, n_cats))
    for i, row in enumerate(annotations):
        for label in row:
            counts[i, cat_idx[label]] += 1

    # P_i (agreement per item)
    P_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = float(np.mean(P_i))

    # P_e (expected agreement)
    p_j = np.sum(counts, axis=0) / (n_items * n_raters)
    P_e = float(np.sum(p_j ** 2))

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate annotations
# ═══════════════════════════════════════════════════════════════════════════

def load_annotations(path: str | Path) -> List[Dict[str, Any]]:
    """Load completed annotation sheet."""
    p = Path(path)
    if p.suffix == '.json':
        return json.loads(p.read_text())
    # TSV
    records: list = []
    with open(p, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            records.append(dict(row))
    return records


def aggregate_annotations(records: List[Dict[str, Any]],
                           annotator_keys: Sequence[str] = (
                               'annotator_1', 'annotator_2', 'annotator_3')
                           ) -> Dict[str, Any]:
    """Compute agreement statistics from completed annotations."""
    # filter records with at least 2 non-empty annotations
    valid: list = []
    for r in records:
        labels = [r.get(k, '').strip() for k in annotator_keys
                  if r.get(k, '').strip()]
        if len(labels) >= 2:
            valid.append(labels)

    if len(valid) < 2:
        return {'n_annotated': len(valid), 'kappa': 0.0}

    # pairwise Cohen's kappa
    n_ann = len(valid[0])
    pairwise: list = []
    for i in range(n_ann):
        for j in range(i + 1, n_ann):
            a = [row[i] for row in valid if len(row) > max(i, j)]
            b = [row[j] for row in valid if len(row) > max(i, j)]
            if a and b:
                pairwise.append(cohens_kappa(a, b))

    # Fleiss
    fk = fleiss_kappa(valid) if valid else 0.0

    # majority vote accuracy
    majority: list = []
    for labels in valid:
        c = Counter(labels)
        majority.append(c.most_common(1)[0][0])

    return {
        'n_annotated': len(valid),
        'fleiss_kappa': fk,
        'avg_cohens_kappa': float(np.mean(pairwise)) if pairwise else 0.0,
        'majority_labels': dict(Counter(majority)),
    }
