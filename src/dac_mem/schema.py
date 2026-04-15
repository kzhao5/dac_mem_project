"""
Core data classes for DAC-Mem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Turn:
    turn_id: str
    session_id: str
    speaker: str
    text: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Example:
    example_id: str
    dataset: str
    question: str
    answer: str
    question_type: str
    turns: List[Turn]
    question_timestamp: Optional[str] = None
    evidence_turn_ids: List[str] = field(default_factory=list)
    evidence_session_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryItem:
    memory_id: str
    text: str
    turn_id: str
    session_id: str
    speaker: str
    timestamp: str
    # --- type and features ---
    memory_type: str = 'generic_fact'
    utility: float = 0.0
    confidence: float = 1.0
    novelty: float = 1.0
    recency: float = 1.0
    type_prior: float = 0.5
    derivability: float = 0.0
    stale_risk: float = 0.0
    # --- decision ---
    action: str = 'SKIP'
    score: float = 0.0
    # --- LLM-based enrichments ---
    llm_type: Optional[str] = None          # LLM-predicted type
    llm_derivability: Optional[float] = None  # LLM judge derivability
    probe_derivability: Optional[float] = None  # tool-grounded probe score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedItem:
    memory_id: str
    text: str
    source: str
    score: float
    turn_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    """Result of LLM-as-judge evaluation."""
    score: float                 # 1-5 scale
    reasoning: str = ''
    judge_model: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    controller: str
    dataset: str
    n_examples: int
    metrics: Dict[str, float]
    per_example: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ControllerState:
    persistent: List[MemoryItem] = field(default_factory=list)
    ephemeral: List[MemoryItem] = field(default_factory=list)
    tool_entries: Dict[str, List[MemoryItem]] = field(default_factory=dict)


def as_dict_sequence(items: Sequence[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        if hasattr(item, '__dict__'):
            out.append(dict(item.__dict__))
        else:
            out.append(dict(item))
    return out
