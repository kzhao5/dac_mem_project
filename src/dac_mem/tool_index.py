"""
Tool-conditioned derivability probing — the core contribution of DAC-Mem.

Three probe implementations:
  1. EmbeddingDerivabilityProbe  — cosine similarity against tool index
  2. LLMDerivabilityProbe        — LLM judges recoverability (lightweight)
  3. ToolGroundedRecoveryProbe   — bounded recovery agent (full version)

The paper reports results with all three and compares them in ablation.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .embedding import BaseEmbedder, get_embedder
from .schema import MemoryItem, RetrievedItem
from .utils import normalize_text


# ── data ───────────────────────────────────────────────────────────────────

@dataclass
class ToolEntry:
    entry_id: str
    tool_type: str           # 'profile' | 'calendar' | 'artifact'
    slot: str
    text: str
    source_turn_id: str
    source_session_id: str
    timestamp: str


# ── tool index ─────────────────────────────────────────────────────────────

class ToolIndex:
    """Stores structured facts and supports semantic retrieval against them."""

    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        self.entries: Dict[str, List[ToolEntry]] = defaultdict(list)
        self.embedder = embedder or get_embedder('bge-large')

    def add_entries(self, entries: Iterable[ToolEntry]) -> None:
        for e in entries:
            self.entries[e.tool_type].append(e)

    def _texts(self, tool_type: str) -> List[str]:
        return [e.text for e in self.entries.get(tool_type, [])]

    # --- embedding-based probe ---
    def probe_recoverability(self, item: MemoryItem,
                             relevant_tools: List[str],
                             max_tools: int = 2) -> Tuple[float, Dict[str, float]]:
        """Cosine-similarity probe: can `item` be recovered from tools?"""
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
            score = self.embedder.max_similarity(query, texts)
            tool_scores[tool] = score
            best = max(best, score)
        cost_penalty = 0.05 * max(0, len(relevant_tools[:max_tools]) - 1)
        final = max(0.0, min(1.0, best - cost_penalty))
        return final, tool_scores

    # --- retrieval ---
    def query(self, question: str, query_tool_types: List[str],
              top_k: int = 3) -> List[RetrievedItem]:
        out: List[RetrievedItem] = []
        if not query_tool_types:
            return out
        q = normalize_text(question)
        for tool in query_tool_types:
            entries = self.entries.get(tool, [])
            if not entries:
                continue
            texts = [e.text for e in entries]
            sims = self.embedder.similarity(q, texts)
            top_idx = np.argsort(-sims)[:top_k]
            for idx in top_idx:
                e = entries[int(idx)]
                out.append(RetrievedItem(
                    memory_id=e.entry_id, text=e.text,
                    source=f'tool:{tool}', score=float(sims[int(idx)]),
                    turn_id=e.source_turn_id,
                    session_id=e.source_session_id,
                    metadata={'tool_type': tool, 'slot': e.slot},
                ))
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k]


# ── LLM derivability judge ────────────────────────────────────────────────

DERIVABILITY_JUDGE_SYSTEM = """You are an expert at judging whether a piece of information can be recovered later from external tools.

Available future tools:
- profile: stores stable user facts (name, job, location, education, bio)
- calendar: stores time-bound events, meetings, schedule changes
- artifact: stores file paths, URLs, version numbers, config values
- search: can search previous conversation summaries (but NOT raw transcripts)

Given a candidate memory, judge how easily it can be recovered from these tools in the future."""

DERIVABILITY_JUDGE_PROMPT = """Candidate memory: "{text}"
Memory type: {memory_type}

On a scale of 0.0 to 1.0, how easily can this information be recovered from the available tools (profile, calendar, artifact, search) in the future?

0.0 = impossible to recover (e.g., subjective preference, private rationale)
0.5 = partially recoverable (e.g., a fact that might be in profile but not certain)
1.0 = trivially recoverable (e.g., a file path that can be found by searching)

Respond with ONLY a JSON object: {{"derivability": <float>, "reason": "<brief explanation>"}}"""


class LLMDerivabilityProbe:
    """Lightweight derivability estimation via LLM judge."""

    def __init__(self, llm, batch_size: int = 16):
        self.llm = llm
        self.batch_size = batch_size

    def probe(self, item: MemoryItem) -> float:
        return self.probe_batch([item])[0]

    def probe_batch(self, items: List[MemoryItem]) -> List[float]:
        prompts = [
            DERIVABILITY_JUDGE_PROMPT.format(
                text=it.text, memory_type=it.memory_type)
            for it in items
        ]
        scores: List[float] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            responses = self.llm.batch_generate(
                batch, system=DERIVABILITY_JUDGE_SYSTEM,
                temperature=0.0, max_tokens=120)
            for resp in responses:
                try:
                    clean = resp.strip()
                    if clean.startswith('```'):
                        clean = clean.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
                    parsed = json.loads(clean)
                    scores.append(float(parsed['derivability']))
                except (json.JSONDecodeError, KeyError, ValueError):
                    scores.append(0.3)  # conservative default
        return scores


# ── Tool-grounded recovery probe (full version) ───────────────────────────

RECOVERY_PROBE_SYSTEM = """You are a bounded-budget recovery agent. You are given a target fact and a set of tool outputs. Your job is to determine whether the target fact can be recovered from the tool outputs.

You must ONLY use the information provided in the tool outputs. Do NOT use your own knowledge."""

RECOVERY_PROBE_PROMPT = """Target fact to recover: "{target}"

Available tool outputs:
{tool_outputs}

Can the target fact be fully recovered from these tool outputs?
Respond with ONLY a JSON object:
{{"recovered": true/false, "confidence": <0.0-1.0>, "evidence": "<which tool output supports recovery>"}}"""


class ToolGroundedRecoveryProbe:
    """Full recovery probe: actually queries tools and asks LLM if the
    information can be reconstructed from tool results.

    This is the paper's main version of the derivability probe — it's more
    expensive but more principled than the LLM judge alone.
    """

    def __init__(self, llm, tool_index: ToolIndex, max_tool_queries: int = 3):
        self.llm = llm
        self.tool_index = tool_index
        self.max_queries = max_tool_queries

    def probe(self, item: MemoryItem, relevant_tools: List[str]) -> Tuple[float, Dict]:
        """Attempt to recover `item` from tool results under a bounded budget."""
        if not relevant_tools:
            return 0.0, {'reason': 'no relevant tools'}

        # Step 1: query each relevant tool
        tool_outputs: List[str] = []
        for tool_type in relevant_tools[:self.max_queries]:
            results = self.tool_index.query(
                item.text, [tool_type], top_k=3)
            if results:
                entries_str = '\n'.join(
                    f"  [{tool_type}] {r.text} (score={r.score:.2f})"
                    for r in results
                )
                tool_outputs.append(f"Tool '{tool_type}':\n{entries_str}")

        if not tool_outputs:
            return 0.0, {'reason': 'no tool results'}

        # Step 2: ask LLM whether the fact can be recovered
        prompt = RECOVERY_PROBE_PROMPT.format(
            target=item.text,
            tool_outputs='\n\n'.join(tool_outputs),
        )
        resp = self.llm.generate(prompt, system=RECOVERY_PROBE_SYSTEM,
                                 temperature=0.0, max_tokens=150)
        try:
            clean = resp.strip()
            if clean.startswith('```'):
                clean = clean.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
            parsed = json.loads(clean)
            recovered = parsed.get('recovered', False)
            conf = float(parsed.get('confidence', 0.0))
            score = conf if recovered else conf * 0.3
            return min(1.0, score), parsed
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0.2, {'raw': resp}


# ── helper functions ───────────────────────────────────────────────────────

def relevant_tools_for_type(memory_type: str) -> List[str]:
    return {
        'stable_profile_fact': ['profile'],
        'event_state': ['calendar'],
        'tool_recoverable_artifact': ['artifact'],
        'preference': [],
        'decision_rationale': [],
        'plan_intent': [],
        'transient_debug_state': ['artifact'],
        'generic_fact': [],
    }.get(memory_type, [])


def query_tools_for_question(question: str) -> List[str]:
    q = question.lower()
    tools: List[str] = []
    if any(k in q for k in ('when', 'date', 'time', 'schedule', 'meeting',
                             'appointment', 'before', 'after')):
        tools.append('calendar')
    if any(k in q for k in ('name', 'live', 'work', 'occupation', 'from',
                             'profile', 'who')):
        tools.append('profile')
    if any(k in q for k in ('file', 'path', 'version', 'url', 'link',
                             'branch', 'repo', 'artifact')):
        tools.append('artifact')
    return tools
