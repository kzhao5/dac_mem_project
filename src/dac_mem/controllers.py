"""
Memory write controllers.

Baselines:
  1. StoreAllController       — persist everything
  2. RelevanceOnlyController  — utility threshold
  3. NoveltyRecencyController — novelty + recency
  4. AMACLiteController       — A-MAC 5-factor admission

Our method:
  5. MemJudgeController       — LLM-judged derivability sieve (3-way)
     Uses LLM-as-judge to estimate tool-recoverability,
     falls back to embedding probe when no LLM is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .embedding import BaseEmbedder
from .features import SemanticFeatureComputer
from .schema import ControllerState, MemoryItem
from .tool_index import (
    LLMDerivabilityProbe, ToolIndex, relevant_tools_for_type,
)


@dataclass
class ControllerConfig:
    name: str
    persist_threshold: float = 0.55
    ephemeral_threshold: float = 0.55
    novelty_weight: float = 0.4
    recency_weight: float = 0.2
    utility_weight: float = 1.0
    confidence_weight: float = 0.2
    type_weight: float = 0.3
    derivability_weight: float = 1.0
    stale_weight: float = 0.8
    duplicate_threshold: float = 0.88
    eph_session_window: int = 2


class BaseController:
    def __init__(self, config: Optional[ControllerConfig] = None,
                 embedder: Optional[BaseEmbedder] = None):
        self.config = config or ControllerConfig(name=self.__class__.__name__)
        self.feats = SemanticFeatureComputer(embedder)

    @property
    def name(self) -> str:
        return self.config.name

    def score_item(self, item: MemoryItem, state: ControllerState,
                   current_ts: str, tool_index: ToolIndex) -> MemoryItem:
        item.novelty = self.feats.compute_novelty(item, state.persistent)
        item.recency = self.feats.compute_recency(item, current_ts)
        item.utility = self.feats.compute_utility(item)
        item.stale_risk = self.feats.compute_stale_risk(item)
        return item

    def decide(self, item: MemoryItem, state: ControllerState,
               current_ts: str, tool_index: ToolIndex) -> str:
        raise NotImplementedError

    def apply(self, item: MemoryItem, state: ControllerState,
              current_ts: str, tool_index: ToolIndex) -> str:
        item = self.score_item(item, state, current_ts, tool_index)
        decision = self.decide(item, state, current_ts, tool_index)
        item.action = decision
        if decision == 'PERSIST':
            state.persistent.append(item)
        elif decision == 'EPHEMERAL':
            state.ephemeral.append(item)
        return decision


# ── Baselines ──────────────────────────────────────────────────────────────

class StoreAllController(BaseController):
    def __init__(self, **kw):
        super().__init__(ControllerConfig(name='store_all'), **kw)

    def decide(self, item, state, current_ts, tool_index):
        item.score = 1.0
        return 'PERSIST'


class RelevanceOnlyController(BaseController):
    def __init__(self, threshold: float = 0.58, **kw):
        super().__init__(ControllerConfig(name='relevance_only',
                                          persist_threshold=threshold), **kw)

    def decide(self, item, state, current_ts, tool_index):
        item.score = item.utility
        return 'PERSIST' if item.utility >= self.config.persist_threshold else 'SKIP'


class NoveltyRecencyController(BaseController):
    def __init__(self, threshold: float = 0.55, **kw):
        super().__init__(ControllerConfig(
            name='novelty_recency', persist_threshold=threshold,
            novelty_weight=0.65, recency_weight=0.35), **kw)

    def decide(self, item, state, current_ts, tool_index):
        score = (self.config.novelty_weight * item.novelty +
                 self.config.recency_weight * item.recency)
        item.score = score
        return 'PERSIST' if score >= self.config.persist_threshold else 'SKIP'


class AMACLiteController(BaseController):
    """A-MAC 5-factor admission: utility, confidence, novelty, recency, type_prior."""
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 threshold: float = 0.62, **kw):
        super().__init__(ControllerConfig(name='amac_lite',
                                          persist_threshold=threshold), **kw)
        self.weights = weights or {
            'utility': 0.38, 'confidence': 0.14, 'novelty': 0.22,
            'recency': 0.10, 'type_prior': 0.16,
        }

    def decide(self, item, state, current_ts, tool_index):
        score = (
            self.weights['utility'] * item.utility +
            self.weights['confidence'] * item.confidence +
            self.weights['novelty'] * item.novelty +
            self.weights['recency'] * item.recency +
            self.weights['type_prior'] * item.type_prior
        )
        item.score = float(score)
        return 'PERSIST' if score >= self.config.persist_threshold else 'SKIP'


# ── Our method: MemJudge ───────────────────────────────────────────────────

class MemJudgeController(BaseController):
    """MemJudge: LLM-as-Judge for Tool-Conditioned Memory Writing.

    Core idea: an LLM judges whether each candidate memory is
    tool-recoverable. Recoverable info is filtered out (EPHEMERAL/SKIP),
    only truly non-recoverable valuable info is persisted.

    Three-way decision: PERSIST / EPHEMERAL / SKIP.
    """

    def __init__(self, config: Optional[ControllerConfig] = None,
                 llm=None, **kw):
        cfg = config or ControllerConfig(
            name='memjudge',
            persist_threshold=0.55,
            ephemeral_threshold=0.58,
            utility_weight=1.05,
            confidence_weight=0.15,
            novelty_weight=0.35,
            recency_weight=0.10,
            type_weight=0.35,
            derivability_weight=1.0,
            stale_weight=0.75,
        )
        super().__init__(cfg, **kw)
        self.llm = llm
        self._probe: Optional[LLMDerivabilityProbe] = None

    def score_item(self, item: MemoryItem, state: ControllerState,
                   current_ts: str, tool_index: ToolIndex) -> MemoryItem:
        item = super().score_item(item, state, current_ts, tool_index)
        relevant_tools = relevant_tools_for_type(item.memory_type)

        if self.llm is not None:
            # LLM-as-judge for derivability (main approach)
            if self._probe is None:
                self._probe = LLMDerivabilityProbe(self.llm)
            llm_score = self._probe.probe(item)
            item.llm_derivability = llm_score
            item.derivability = max(item.derivability, llm_score)
        else:
            # Fallback: embedding-based probe
            emb_score, tool_scores = tool_index.probe_recoverability(
                item, relevant_tools=relevant_tools)
            item.metadata['tool_scores'] = tool_scores
            item.derivability = max(item.derivability, emb_score)

        return item

    def decide(self, item: MemoryItem, state: ControllerState,
               current_ts: str, tool_index: ToolIndex) -> str:
        dup_pen = 0.2 if (1.0 - item.novelty) >= self.config.duplicate_threshold else 0.0

        persist_score = (
            self.config.utility_weight * item.utility +
            self.config.confidence_weight * item.confidence +
            self.config.novelty_weight * item.novelty +
            self.config.recency_weight * item.recency +
            self.config.type_weight * item.type_prior -
            self.config.derivability_weight * item.derivability -
            self.config.stale_weight * item.stale_risk -
            dup_pen
        )
        ephemeral_score = (
            0.65 * item.utility +
            0.15 * item.recency +
            0.55 * item.derivability +
            0.30 * item.stale_risk +
            0.15 * item.novelty
        )
        item.metadata['persist_score'] = float(persist_score)
        item.metadata['ephemeral_score'] = float(ephemeral_score)
        item.score = float(max(persist_score, ephemeral_score))

        if persist_score >= self.config.persist_threshold:
            return 'PERSIST'
        if ephemeral_score >= self.config.ephemeral_threshold:
            return 'EPHEMERAL'
        return 'SKIP'


# ── factory ────────────────────────────────────────────────────────────────

def controller_from_name(name: str, llm=None, embedder=None):
    kw = {'embedder': embedder} if embedder else {}
    key = name.lower()
    if key in ('store_all', 'storeall'):
        return StoreAllController(**kw)
    if key in ('relevance', 'relevance_only'):
        return RelevanceOnlyController(**kw)
    if key in ('novelty', 'novelty_recency'):
        return NoveltyRecencyController(**kw)
    if key in ('amac', 'amac_lite'):
        return AMACLiteController(**kw)
    if key in ('dac', 'memjudge', 'dac_mem', 'dacmem'):
        return MemJudgeController(llm=llm, **kw)
    raise ValueError(f'Unknown controller: {name}')
