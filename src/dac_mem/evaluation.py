from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

from .schema import Example, MemoryItem, RetrievedItem
from .utils import exact_match, token_f1


def retrieval_metrics(example: Example, retrieved: Sequence[RetrievedItem]) -> Dict[str, float]:
    retrieved_turns = [r.turn_id for r in retrieved if r.turn_id]
    retrieved_sessions = [r.session_id for r in retrieved if r.session_id]
    evidence_turns = set(example.evidence_turn_ids)
    evidence_sessions = set(example.evidence_session_ids)

    turn_hit = float(any(t in evidence_turns for t in retrieved_turns)) if evidence_turns else 0.0
    session_hit = float(any(s in evidence_sessions for s in retrieved_sessions)) if evidence_sessions else 0.0

    precision_turn = 0.0
    if retrieved_turns:
        precision_turn = sum(t in evidence_turns for t in retrieved_turns) / len(retrieved_turns)

    precision_session = 0.0
    if retrieved_sessions:
        precision_session = sum(s in evidence_sessions for s in retrieved_sessions) / len(retrieved_sessions)

    recall_turn = 0.0
    if evidence_turns:
        recall_turn = len(set(retrieved_turns) & evidence_turns) / len(evidence_turns)

    recall_session = 0.0
    if evidence_sessions:
        recall_session = len(set(retrieved_sessions) & evidence_sessions) / len(evidence_sessions)

    return {
        'turn_hit@k': turn_hit,
        'session_hit@k': session_hit,
        'turn_precision@k': precision_turn,
        'session_precision@k': precision_session,
        'turn_recall@k': recall_turn,
        'session_recall@k': recall_session,
    }



def memory_quality_metrics(persistent: Sequence[MemoryItem], ephemeral: Sequence[MemoryItem]) -> Dict[str, float]:
    p_size = len(persistent)
    e_size = len(ephemeral)
    if p_size:
        derivable_fraction = sum(m.derivability >= 0.5 for m in persistent) / p_size
        stale_mean = sum(m.stale_risk for m in persistent) / p_size
        dup_fraction = sum((1.0 - m.novelty) >= 0.88 for m in persistent) / p_size
    else:
        derivable_fraction = stale_mean = dup_fraction = 0.0
    return {
        'persistent_size': float(p_size),
        'ephemeral_size': float(e_size),
        'persistent_derivable_fraction': float(derivable_fraction),
        'persistent_stale_mean': float(stale_mean),
        'persistent_duplicate_fraction': float(dup_fraction),
    }



def qa_metrics(prediction: str, gold: str) -> Dict[str, float]:
    return {
        'qa_em': exact_match(prediction, gold),
        'qa_f1': token_f1(prediction, gold),
    }



def aggregate(per_example: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for d in per_example for k in d.keys()})
    return {k: float(mean([d.get(k, 0.0) for d in per_example])) for k in keys}
