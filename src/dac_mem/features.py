from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import MemoryItem
from .utils import exp_recency, specificity_score


class FeatureComputer:
    def __init__(self):
        self.vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False, norm='l2')

    def similarity(self, text: str, bank: Iterable[MemoryItem]) -> float:
        texts = [m.text for m in bank]
        if not texts:
            return 0.0
        X = self.vectorizer.transform(texts + [text])
        sims = cosine_similarity(X[-1], X[:-1]).ravel()
        return float(sims.max()) if len(sims) else 0.0

    def compute_novelty(self, item: MemoryItem, bank: List[MemoryItem]) -> float:
        return 1.0 - self.similarity(item.text, bank)

    def compute_recency(self, item: MemoryItem, ref_ts: str) -> float:
        return exp_recency(item.timestamp, ref_ts)

    def speaker_bonus(self, item: MemoryItem) -> float:
        speaker = item.speaker.lower()
        if speaker in {'user', 'speaker_a', 'a', 'human'}:
            return 0.12
        return 0.05

    def compute_utility(self, item: MemoryItem) -> float:
        spec = specificity_score(item.text)
        update_bonus = 0.08 if any(k in item.text.lower() for k in ['changed', 'updated', 'moved', 'switched', 'instead']) else 0.0
        utility = 0.45 * item.type_prior + 0.25 * spec + 0.15 * self.speaker_bonus(item) + 0.15 * item.confidence + update_bonus
        return float(max(0.0, min(1.0, utility)))

    def compute_stale_risk(self, item: MemoryItem) -> float:
        extra = 0.0
        txt = item.text.lower()
        if any(k in txt for k in ['today', 'tomorrow', 'this week', 'next week', 'current', 'currently']):
            extra += 0.15
        if any(k in txt for k in ['version', 'branch', 'status', 'temporary', 'debug']):
            extra += 0.1
        return float(max(0.0, min(1.0, item.stale_risk + extra)))
