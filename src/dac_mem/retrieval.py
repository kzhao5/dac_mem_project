"""
Hybrid retrieval: BM25 + dense semantic embeddings.

Uses real sentence embeddings by default (upgrade from legacy hashing).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from .embedding import BaseEmbedder, get_embedder
from .schema import MemoryItem, RetrievedItem
from .tool_index import ToolIndex, query_tools_for_question
from .utils import tokenize


class _FallbackBM25:
    """Simple term-overlap BM25 substitute when rank_bm25 is unavailable."""
    def __init__(self, corpus_tokens: Sequence[Sequence[str]]):
        self.corpus = [list(x) for x in corpus_tokens]

    def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
        q = Counter(query_tokens)
        scores = []
        for doc in self.corpus:
            d = Counter(doc)
            overlap = sum(min(q[k], d[k]) for k in q)
            scores.append(overlap / max(len(doc), 1))
        return scores


class HybridRetriever:
    """BM25 + dense semantic retrieval with score fusion."""

    def __init__(self, bm25_weight: float = 0.55, dense_weight: float = 0.45,
                 embedder: Optional[BaseEmbedder] = None):
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.embedder = embedder or get_embedder('bge-large')

    def _make_bm25(self, tokenized: Sequence[Sequence[str]]):
        if BM25Okapi is not None:
            return BM25Okapi(tokenized)
        return _FallbackBM25(tokenized)

    def _retrieve_bank(self, question: str, bank: Sequence[MemoryItem],
                       source_name: str, top_k: int) -> List[RetrievedItem]:
        if not bank:
            return []
        # BM25
        tokenized = [tokenize(m.text) for m in bank]
        bm25 = self._make_bm25(tokenized)
        bm25_scores = np.array(bm25.get_scores(tokenize(question)), dtype=float)
        if bm25_scores.max() > 0:
            bm25_scores /= bm25_scores.max()

        # Dense
        texts = [m.text for m in bank]
        dense_scores = self.embedder.similarity(question, texts)
        if dense_scores.max() > 0:
            dense_scores /= dense_scores.max()

        scores = self.bm25_weight * bm25_scores + self.dense_weight * dense_scores
        idx = np.argsort(-scores)[:top_k]
        return [
            RetrievedItem(
                memory_id=bank[int(i)].memory_id,
                text=bank[int(i)].text,
                source=source_name,
                score=float(scores[int(i)]),
                turn_id=bank[int(i)].turn_id,
                session_id=bank[int(i)].session_id,
                metadata={'memory_type': bank[int(i)].memory_type,
                          'action': bank[int(i)].action},
            )
            for i in idx
        ]

    def retrieve(self, question: str,
                 persistent_bank: Sequence[MemoryItem],
                 ephemeral_bank: Sequence[MemoryItem],
                 tool_index: ToolIndex,
                 top_k: int = 8,
                 top_k_tools: int = 3) -> List[RetrievedItem]:
        tool_types = query_tools_for_question(question)
        tool_hits = tool_index.query(question, tool_types, top_k=top_k_tools)
        p_hits = self._retrieve_bank(question, persistent_bank,
                                     'persistent', top_k=top_k)
        e_hits = self._retrieve_bank(question, ephemeral_bank,
                                     'ephemeral', top_k=max(1, top_k // 2))
        # merge by best score per turn_id
        merged: Dict[str, RetrievedItem] = {}
        for it in tool_hits + p_hits + e_hits:
            key = it.turn_id or it.memory_id
            if key not in merged or it.score > merged[key].score:
                merged[key] = it
        out = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return out[:top_k]
