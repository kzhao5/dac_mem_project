from __future__ import annotations

import re
from typing import Dict, Iterable, List

from .schema import MemoryItem, Turn
from .utils import (
    DEBUG_PATTERN,
    PLAN_PATTERN,
    PREFERENCE_PATTERN,
    PROFILE_PATTERN,
    RATIONALE_PATTERN,
    confidence_from_text,
    normalize_text,
    simple_sentence_split,
    stable_hash,
)


TYPE_PRIORS = {
    'preference': 0.95,
    'decision_rationale': 0.9,
    'stable_profile_fact': 0.85,
    'generic_fact': 0.55,
    'event_state': 0.45,
    'tool_recoverable_artifact': 0.35,
    'plan_intent': 0.3,
    'transient_debug_state': 0.2,
}


STALE_PRIORS = {
    'preference': 0.15,
    'decision_rationale': 0.2,
    'stable_profile_fact': 0.2,
    'generic_fact': 0.35,
    'event_state': 0.75,
    'tool_recoverable_artifact': 0.8,
    'plan_intent': 0.7,
    'transient_debug_state': 0.9,
}


RECOVERABILITY_PRIORS = {
    'preference': 0.15,
    'decision_rationale': 0.1,
    'stable_profile_fact': 0.45,
    'generic_fact': 0.25,
    'event_state': 0.8,
    'tool_recoverable_artifact': 0.9,
    'plan_intent': 0.35,
    'transient_debug_state': 0.65,
}


PROFILE_FACT_HINTS = [
    re.compile(r"\bmy name is\b", re.I),
    re.compile(r"\bi am\b", re.I),
    re.compile(r"\bi'm\b", re.I),
    re.compile(r"\bi work as\b", re.I),
    re.compile(r"\bi live in\b", re.I),
    re.compile(r"\bi'm from\b", re.I),
    re.compile(r"\bi study\b", re.I),
]


class CandidateExtractor:
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
        if self._is_artifact(text):
            return 'tool_recoverable_artifact'
        if self._is_event(text):
            return 'event_state'
        return 'generic_fact'

    def _is_event(self, text: str) -> bool:
        from .utils import contains_temporal_cue
        return contains_temporal_cue(text)

    def _is_artifact(self, text: str) -> bool:
        from .utils import has_path_or_artifact
        return has_path_or_artifact(text)

    def extract(self, turn: Turn) -> List[MemoryItem]:
        items: List[MemoryItem] = []
        for idx, sent in enumerate(simple_sentence_split(turn.text)):
            sent = normalize_text(sent)
            if len(sent) < self.min_chars:
                continue
            mtype = self.classify_type(sent, turn.speaker)
            item = MemoryItem(
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
            )
            items.append(item)
        return items


class ToolFactExtractor:
    """
    Narrow structured extractors for external tools.
    We intentionally keep these tools conservative so that not everything becomes derivable.
    """

    def extract_profile_facts(self, item: MemoryItem) -> Iterable[Dict[str, str]]:
        txt = item.text
        if item.memory_type != 'stable_profile_fact':
            return []
        lowered = txt.lower()
        kind = None
        if 'my name is' in lowered:
            kind = 'name'
        elif 'work as' in lowered or 'work at' in lowered or 'work in' in lowered:
            kind = 'occupation'
        elif 'live in' in lowered or "i'm from" in lowered:
            kind = 'location'
        elif 'study' in lowered:
            kind = 'education'
        elif lowered.startswith('i am') or lowered.startswith("i'm"):
            kind = 'bio'
        if not kind:
            return []
        return [{'tool_type': 'profile', 'slot': kind, 'value': txt}]

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
