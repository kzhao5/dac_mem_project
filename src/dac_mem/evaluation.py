"""
Evaluation metrics for DAC-Mem.

Four categories (per the proposal):
  A. Downstream task metrics  — QA accuracy, F1, LLM-judge score
  B. Memory quality metrics   — size, stale rate, duplicate rate, derivable frac
  C. Retrieval quality metrics — precision, recall, hit@k
  D. Efficiency metrics        — latency, token cost, context length

Plus: bootstrap significance testing and LLM-as-judge.
"""
from __future__ import annotations

import json
import time
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import Example, JudgeResult, MemoryItem, RetrievedItem
from .utils import exact_match, token_f1

# ═══════════════════════════════════════════════════════════════════════════
# A. Downstream task metrics
# ═══════════════════════════════════════════════════════════════════════════

def qa_metrics(prediction: str, gold: str) -> Dict[str, float]:
    return {
        'qa_em': exact_match(prediction, gold),
        'qa_f1': token_f1(prediction, gold),
    }


# ── LLM-as-judge ──────────────────────────────────────────────────────────

LLM_JUDGE_SYSTEM = """You are an expert judge evaluating the quality of an answer to a question about a conversation.

Score the answer on a scale of 1 to 5:
  1 = Completely wrong or irrelevant
  2 = Partially relevant but mostly incorrect
  3 = Partially correct but missing key information
  4 = Mostly correct with minor gaps
  5 = Fully correct and complete

Consider: factual accuracy, completeness, and relevance to the question."""

LLM_JUDGE_PROMPT = """Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}

Evaluate the predicted answer. Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""


class LLMJudge:
    """Uses an LLM to evaluate answer quality on a 1–5 scale."""

    def __init__(self, llm):
        self.llm = llm

    def judge(self, question: str, gold: str, prediction: str) -> JudgeResult:
        prompt = LLM_JUDGE_PROMPT.format(
            question=question, gold=gold, prediction=prediction)
        resp = self.llm.generate(prompt, system=LLM_JUDGE_SYSTEM,
                                 temperature=0.0, max_tokens=200)
        try:
            clean = resp.strip()
            if clean.startswith('```'):
                clean = clean.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
            parsed = json.loads(clean)
            return JudgeResult(
                score=float(parsed['score']),
                reasoning=parsed.get('reasoning', ''),
                judge_model=self.llm.model_name,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return JudgeResult(score=3.0, reasoning=f'parse_error: {resp[:100]}',
                               judge_model=self.llm.model_name)

    def judge_batch(self, examples: List[Tuple[str, str, str]]) -> List[JudgeResult]:
        return [self.judge(q, g, p) for q, g, p in examples]


# ═══════════════════════════════════════════════════════════════════════════
# B. Memory quality metrics
# ═══════════════════════════════════════════════════════════════════════════

def memory_quality_metrics(persistent: Sequence[MemoryItem],
                           ephemeral: Sequence[MemoryItem]) -> Dict[str, float]:
    p_size = len(persistent)
    e_size = len(ephemeral)
    if p_size:
        derivable_frac = sum(m.derivability >= 0.5 for m in persistent) / p_size
        stale_mean = sum(m.stale_risk for m in persistent) / p_size
        dup_frac = sum((1.0 - m.novelty) >= 0.88 for m in persistent) / p_size
    else:
        derivable_frac = stale_mean = dup_frac = 0.0
    return {
        'persistent_size': float(p_size),
        'ephemeral_size': float(e_size),
        'persistent_derivable_fraction': float(derivable_frac),
        'persistent_stale_mean': float(stale_mean),
        'persistent_duplicate_fraction': float(dup_frac),
    }


# ═══════════════════════════════════════════════════════════════════════════
# C. Retrieval quality metrics
# ═══════════════════════════════════════════════════════════════════════════

def retrieval_metrics(example: Example,
                      retrieved: Sequence[RetrievedItem]) -> Dict[str, float]:
    r_turns = [r.turn_id for r in retrieved if r.turn_id]
    r_sessions = [r.session_id for r in retrieved if r.session_id]
    ev_turns = set(example.evidence_turn_ids)
    ev_sessions = set(example.evidence_session_ids)

    turn_hit = float(any(t in ev_turns for t in r_turns)) if ev_turns else 0.0
    sess_hit = float(any(s in ev_sessions for s in r_sessions)) if ev_sessions else 0.0

    prec_turn = (sum(t in ev_turns for t in r_turns) / len(r_turns)
                 if r_turns else 0.0)
    prec_sess = (sum(s in ev_sessions for s in r_sessions) / len(r_sessions)
                 if r_sessions else 0.0)
    rec_turn = (len(set(r_turns) & ev_turns) / len(ev_turns)
                if ev_turns else 0.0)
    rec_sess = (len(set(r_sessions) & ev_sessions) / len(ev_sessions)
                if ev_sessions else 0.0)

    return {
        'turn_hit@k': turn_hit, 'session_hit@k': sess_hit,
        'turn_precision@k': prec_turn, 'session_precision@k': prec_sess,
        'turn_recall@k': rec_turn, 'session_recall@k': rec_sess,
    }


# ═══════════════════════════════════════════════════════════════════════════
# D. Efficiency metrics
# ═══════════════════════════════════════════════════════════════════════════

def efficiency_metrics(start_time: float, end_time: float,
                       retrieved: Sequence[RetrievedItem],
                       persistent_size: int) -> Dict[str, float]:
    ctx_len = sum(len(r.text.split()) for r in retrieved)
    return {
        'latency_sec': end_time - start_time,
        'retrieved_context_words': float(ctx_len),
        'persistent_size': float(persistent_size),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate(per_example: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for d in per_example for k in d})
    return {k: float(mean([d.get(k, 0.0) for d in per_example])) for k in keys}


# ═══════════════════════════════════════════════════════════════════════════
# Statistical significance testing
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values: Sequence[float], n_bootstrap: int = 10000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (mean, lower, upper)."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(42)
    means = np.array([arr[rng.integers(0, n, size=n)].mean()
                      for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return float(arr.mean()), float(np.percentile(means, 100 * alpha)), \
           float(np.percentile(means, 100 * (1 - alpha)))


def paired_bootstrap_test(scores_a: Sequence[float],
                          scores_b: Sequence[float],
                          n_bootstrap: int = 10000) -> Dict[str, float]:
    """Paired bootstrap test: is system A significantly better than B?"""
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    assert len(a) == len(b)
    n = len(a)
    observed_diff = float((a - b).mean())
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if (a[idx] - b[idx]).mean() <= 0:
            count += 1
    p_value = count / n_bootstrap
    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'significant_at_05': p_value < 0.05,
        'significant_at_01': p_value < 0.01,
    }


def significance_matrix(results: Dict[str, List[float]],
                         metric_name: str = '') -> Dict[str, Dict[str, float]]:
    """Compute pairwise significance between all systems."""
    systems = sorted(results.keys())
    matrix: Dict[str, Dict[str, float]] = {}
    for a in systems:
        matrix[a] = {}
        for b in systems:
            if a == b:
                matrix[a][b] = 1.0
            else:
                test = paired_bootstrap_test(results[a], results[b])
                matrix[a][b] = test['p_value']
    return matrix
