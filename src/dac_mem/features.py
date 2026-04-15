"""
Feature computation for memory candidates.

SemanticFeatureComputer uses real sentence embeddings (default).
FeatureComputer uses legacy hashing vectors (for ablation only).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .embedding import BaseEmbedder, get_embedder
from .schema import MemoryItem
from .utils import contains_temporal_cue, exp_recency, has_path_or_artifact, specificity_score


class SemanticFeatureComputer:
    """Compute novelty, recency, utility, and stale-risk using real embeddings."""

    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        self.embedder = embedder or get_embedder('bge-large')

    def compute_novelty(self, item: MemoryItem, bank: List[MemoryItem]) -> float:
        if not bank:
            return 1.0
        texts = [m.text for m in bank]
        sim = self.embedder.max_similarity(item.text, texts)
        return 1.0 - sim

    def compute_recency(self, item: MemoryItem, ref_ts: str) -> float:
        return exp_recency(item.timestamp, ref_ts)

    def speaker_bonus(self, item: MemoryItem) -> float:
        s = item.speaker.lower()
        return 0.12 if s in ('user', 'speaker_a', 'a', 'human') else 0.05

    def compute_utility(self, item: MemoryItem) -> float:
        spec = specificity_score(item.text)
        update = 0.08 if any(k in item.text.lower()
                             for k in ('changed', 'updated', 'moved',
                                       'switched', 'instead')) else 0.0
        u = (0.45 * item.type_prior + 0.25 * spec +
             0.15 * self.speaker_bonus(item) + 0.15 * item.confidence + update)
        return float(max(0.0, min(1.0, u)))

    def compute_stale_risk(self, item: MemoryItem) -> float:
        extra = 0.0
        txt = item.text.lower()
        if any(k in txt for k in ('today', 'tomorrow', 'this week',
                                  'next week', 'current', 'currently')):
            extra += 0.15
        if any(k in txt for k in ('version', 'branch', 'status',
                                  'temporary', 'debug')):
            extra += 0.1
        return float(max(0.0, min(1.0, item.stale_risk + extra)))


# Legacy hashing-based computer — only for ablation: "effect of real embeddings"
class FeatureComputer(SemanticFeatureComputer):
    """Hashing-vector baseline (kept for ablation comparison)."""

    def __init__(self):
        super().__init__(embedder=get_embedder('hashing'))
