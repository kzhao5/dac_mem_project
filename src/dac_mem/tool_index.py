from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import MemoryItem, RetrievedItem
from .utils import normalize_text


@dataclass
class ToolEntry:
    entry_id: str
    tool_type: str
    slot: str
    text: str
    source_turn_id: str
    source_session_id: str
    timestamp: str


class ToolIndex:
    def __init__(self):
        self.entries: Dict[str, List[ToolEntry]] = defaultdict(list)
        self.vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False, norm='l2')

    def add_entries(self, entries: Iterable[ToolEntry]) -> None:
        for e in entries:
            self.entries[e.tool_type].append(e)

    def _texts(self, tool_type: str) -> List[str]:
        return [e.text for e in self.entries.get(tool_type, [])]

    def probe_recoverability(self, item: MemoryItem, relevant_tools: List[str], max_tools: int = 2) -> Tuple[float, Dict[str, float]]:
        """
        Returns a recoverability score in [0,1]. We reward strong tool matches and penalize expensive multi-tool access.
        """
        if not relevant_tools:
            return 0.0, {}
        query = normalize_text(item.text)
        tool_scores: Dict[str, float] = {}
        best = 0.0
        for tool in relevant_tools[:max_tools]:
            texts = self._texts(tool)
            if not texts:
                tool_scores[tool] = 0.0
                continue
            X = self.vectorizer.transform(texts + [query])
            sims = cosine_similarity(X[-1], X[:-1]).ravel()
            score = float(sims.max()) if len(sims) else 0.0
            tool_scores[tool] = score
            best = max(best, score)
        cost_penalty = 0.05 * max(0, len(relevant_tools[:max_tools]) - 1)
        final = max(0.0, min(1.0, best - cost_penalty))
        return final, tool_scores

    def query(self, question: str, query_tool_types: List[str], top_k: int = 3) -> List[RetrievedItem]:
        out: List[RetrievedItem] = []
        if not query_tool_types:
            return out
        q = normalize_text(question)
        for tool in query_tool_types:
            entries = self.entries.get(tool, [])
            if not entries:
                continue
            texts = [e.text for e in entries]
            X = self.vectorizer.transform(texts + [q])
            sims = cosine_similarity(X[-1], X[:-1]).ravel()
            top_idx = np.argsort(-sims)[:top_k]
            for idx in top_idx:
                e = entries[int(idx)]
                out.append(RetrievedItem(
                    memory_id=e.entry_id,
                    text=e.text,
                    source=f'tool:{tool}',
                    score=float(sims[int(idx)]),
                    turn_id=e.source_turn_id,
                    session_id=e.source_session_id,
                    metadata={'tool_type': tool, 'slot': e.slot},
                ))
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k]


def relevant_tools_for_type(memory_type: str) -> List[str]:
    mapping = {
        'stable_profile_fact': ['profile'],
        'event_state': ['calendar'],
        'tool_recoverable_artifact': ['artifact'],
        'preference': [],
        'decision_rationale': [],
        'plan_intent': [],
        'transient_debug_state': ['artifact'],
        'generic_fact': [],
    }
    return mapping.get(memory_type, [])


def query_tools_for_question(question: str) -> List[str]:
    q = question.lower()
    tools: List[str] = []
    if any(k in q for k in ['when', 'date', 'time', 'schedule', 'meeting', 'appointment', 'before', 'after']):
        tools.append('calendar')
    if any(k in q for k in ['name', 'live', 'work', 'occupation', 'from', 'profile']):
        tools.append('profile')
    if any(k in q for k in ['file', 'path', 'version', 'url', 'link', 'branch', 'repo', 'artifact']):
        tools.append('artifact')
    return tools
