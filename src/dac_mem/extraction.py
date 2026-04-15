"""
Candidate extraction and memory-type classification.

Two extractors:
  - CandidateExtractor  : fast, rule-based (always available)
  - LLMCandidateExtractor: uses an LLM for 8-way type classification
"""
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional

from .schema import MemoryItem, Turn
from .utils import (
    DEBUG_PATTERN, PLAN_PATTERN, PREFERENCE_PATTERN, PROFILE_PATTERN,
    RATIONALE_PATTERN, confidence_from_text, contains_temporal_cue,
    has_path_or_artifact, normalize_text, simple_sentence_split, stable_hash,
)

# ── priors ─────────────────────────────────────────────────────────────────

MEMORY_TYPES = [
    'preference', 'decision_rationale', 'stable_profile_fact',
    'generic_fact', 'event_state', 'tool_recoverable_artifact',
    'plan_intent', 'transient_debug_state',
]

TYPE_PRIORS: Dict[str, float] = {
    'preference': 0.95, 'decision_rationale': 0.9,
    'stable_profile_fact': 0.85, 'generic_fact': 0.55,
    'event_state': 0.45, 'tool_recoverable_artifact': 0.35,
    'plan_intent': 0.3, 'transient_debug_state': 0.2,
}

STALE_PRIORS: Dict[str, float] = {
    'preference': 0.15, 'decision_rationale': 0.2,
    'stable_profile_fact': 0.2, 'generic_fact': 0.35,
    'event_state': 0.75, 'tool_recoverable_artifact': 0.8,
    'plan_intent': 0.7, 'transient_debug_state': 0.9,
}

RECOVERABILITY_PRIORS: Dict[str, float] = {
    'preference': 0.15, 'decision_rationale': 0.1,
    'stable_profile_fact': 0.45, 'generic_fact': 0.25,
    'event_state': 0.8, 'tool_recoverable_artifact': 0.9,
    'plan_intent': 0.35, 'transient_debug_state': 0.65,
}

# ── LLM prompt for type classification ─────────────────────────────────────

TYPE_CLASSIFICATION_SYSTEM = """You are a memory-type classifier for a long-horizon LLM agent.
Given a single candidate memory sentence, classify it into exactly ONE of these 8 types:

1. preference — user likes, dislikes, habits, style preferences
2. decision_rationale — reasons behind a decision, why something was chosen/rejected
3. stable_profile_fact — long-term identity facts (name, job, location, education)
4. generic_fact — general factual statement, not clearly fitting other categories
5. event_state — time-bound event, meeting, schedule change, current status
6. tool_recoverable_artifact — file path, URL, version number, branch name, config value
7. plan_intent — future plan, intention, to-do, next step
8. transient_debug_state — error, bug, temporary workaround, debug observation

Respond with ONLY a JSON object: {"type": "<type_name>", "confidence": <0.0-1.0>}"""

TYPE_CLASSIFICATION_PROMPT = """Classify this memory candidate:
Speaker: {speaker}
Text: "{text}"
"""

# ── Rule-based extractor ───────────────────────────────────────────────────

class CandidateExtractor:
    """Fast rule-based extraction and type classification."""

    def __init__(self, min_chars: int = 18):
        self.min_chars = min_chars

    def classify_type(self, text: str, speaker: str) -> str:
        if PREFERENCE_PATTERN.search(text):
            return 'preference'
        if RATIONALE_PATTERN.search(text):
            return 'decision_rationale'
        if PROFILE_PATTERN.search(text):
            return 'stable_profile_fact'
        if DEBUG_PATTERN.search(text):
            return 'transient_debug_state'
        if PLAN_PATTERN.search(text):
            return 'plan_intent'
        if has_path_or_artifact(text):
            return 'tool_recoverable_artifact'
        if contains_temporal_cue(text):
            return 'event_state'
        return 'generic_fact'

    def extract(self, turn: Turn) -> List[MemoryItem]:
        items: List[MemoryItem] = []
        for idx, sent in enumerate(simple_sentence_split(turn.text)):
            sent = normalize_text(sent)
            if len(sent) < self.min_chars:
                continue
            mtype = self.classify_type(sent, turn.speaker)
            items.append(MemoryItem(
                memory_id=f"{turn.turn_id}_{idx}_{stable_hash(sent)}",
                text=sent,
                turn_id=turn.turn_id,
                session_id=turn.session_id,
                speaker=turn.speaker,
                timestamp=turn.timestamp,
                memory_type=mtype,
                confidence=confidence_from_text(sent),
                type_prior=TYPE_PRIORS.get(mtype, 0.5),
                stale_risk=STALE_PRIORS.get(mtype, 0.4),
                derivability=RECOVERABILITY_PRIORS.get(mtype, 0.2),
                metadata={'source': 'turn_sentence'},
            ))
        return items


# ── LLM-based extractor ───────────────────────────────────────────────────

class LLMCandidateExtractor:
    """Uses an LLM to classify memory types — much more accurate than regex.

    Falls back to rule-based classification on LLM failure.
    """

    def __init__(self, llm, min_chars: int = 18, batch_size: int = 32):
        self.llm = llm
        self.min_chars = min_chars
        self.batch_size = batch_size
        self._rule_fallback = CandidateExtractor(min_chars=min_chars)

    def _classify_batch(self, items: List[MemoryItem]) -> List[MemoryItem]:
        """Classify a batch of items using the LLM."""
        prompts = [
            TYPE_CLASSIFICATION_PROMPT.format(speaker=it.speaker, text=it.text)
            for it in items
        ]
        responses = self.llm.batch_generate(
            prompts, system=TYPE_CLASSIFICATION_SYSTEM,
            temperature=0.0, max_tokens=80,
        )
        for item, resp in zip(items, responses):
            try:
                # handle potential markdown wrapping
                clean = resp.strip()
                if clean.startswith('```'):
                    clean = clean.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
                parsed = json.loads(clean)
                mtype = parsed.get('type', 'generic_fact')
                if mtype not in MEMORY_TYPES:
                    mtype = 'generic_fact'
                item.llm_type = mtype
                item.memory_type = mtype
                item.confidence = float(parsed.get('confidence', item.confidence))
                item.type_prior = TYPE_PRIORS.get(mtype, 0.5)
                item.stale_risk = STALE_PRIORS.get(mtype, 0.4)
                item.derivability = RECOVERABILITY_PRIORS.get(mtype, 0.2)
            except (json.JSONDecodeError, KeyError, ValueError):
                # fallback to rule-based
                mtype = self._rule_fallback.classify_type(item.text, item.speaker)
                item.memory_type = mtype
                item.type_prior = TYPE_PRIORS.get(mtype, 0.5)
                item.stale_risk = STALE_PRIORS.get(mtype, 0.4)
                item.derivability = RECOVERABILITY_PRIORS.get(mtype, 0.2)
        return items

    def extract(self, turn: Turn) -> List[MemoryItem]:
        """Extract candidates and classify types with LLM."""
        raw_items: List[MemoryItem] = []
        for idx, sent in enumerate(simple_sentence_split(turn.text)):
            sent = normalize_text(sent)
            if len(sent) < self.min_chars:
                continue
            raw_items.append(MemoryItem(
                memory_id=f"{turn.turn_id}_{idx}_{stable_hash(sent)}",
                text=sent,
                turn_id=turn.turn_id,
                session_id=turn.session_id,
                speaker=turn.speaker,
                timestamp=turn.timestamp,
                confidence=confidence_from_text(sent),
                metadata={'source': 'turn_sentence'},
            ))
        # batch classify
        for i in range(0, len(raw_items), self.batch_size):
            batch = raw_items[i:i + self.batch_size]
            self._classify_batch(batch)
        return raw_items


# ── Tool fact extraction ───────────────────────────────────────────────────

class ToolFactExtractor:
    """Extract structured facts for the simulated tool index."""

    def extract_profile_facts(self, item: MemoryItem) -> Iterable[Dict[str, str]]:
        if item.memory_type != 'stable_profile_fact':
            return []
        low = item.text.lower()
        kind = None
        if 'my name is' in low:
            kind = 'name'
        elif any(k in low for k in ('work as', 'work at', 'work in')):
            kind = 'occupation'
        elif any(k in low for k in ('live in', "i'm from", 'i am from')):
            kind = 'location'
        elif 'study' in low:
            kind = 'education'
        elif low.startswith(('i am', "i'm")):
            kind = 'bio'
        if not kind:
            return []
        return [{'tool_type': 'profile', 'slot': kind, 'value': item.text}]

    def extract_event_facts(self, item: MemoryItem) -> Iterable[Dict[str, str]]:
        if item.memory_type != 'event_state':
            return []
        return [{'tool_type': 'calendar', 'slot': 'event', 'value': item.text}]

    def extract_artifact_facts(self, item: MemoryItem) -> Iterable[Dict[str, str]]:
        if item.memory_type != 'tool_recoverable_artifact':
            return []
        return [{'tool_type': 'artifact', 'slot': 'artifact', 'value': item.text}]

    def extract(self, item: MemoryItem) -> List[Dict[str, str]]:
        facts: List[Dict[str, str]] = []
        facts.extend(self.extract_profile_facts(item))
        facts.extend(self.extract_event_facts(item))
        facts.extend(self.extract_artifact_facts(item))
        return facts
