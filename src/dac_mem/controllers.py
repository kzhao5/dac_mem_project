"""
Memory write controllers for DAC-Mem.

Controllers (baselines + ours):
  1. StoreAllController       — persist everything
  2. RelevanceOnlyController  — utility threshold
  3. NoveltyRecencyController — novelty + recency weighted
  4. AMACLiteController       — A-MAC 5-factor admission
  5. DACMemController         — our method: derivability-aware 3-way decision

Additional baselines (in baselines.py):
  6. MemoryBankController     — timestamp-weighted
  7. MemoryR1Controller       — LLM-prompted CRUD manager
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedding import BaseEmbedder
from .features import SemanticFeatureComputer
from .schema import ControllerState, MemoryItem
from .tool_index import (
    LLMDerivabilityProbe,
    ToolEntry,
    ToolGroundedRecoveryProbe,
    ToolIndex,
    relevant_tools_for_type,
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
    # probe mode: 'embedding' | 'llm_judge' | 'tool_grounded'
    probe_mode: str = 'embedding'


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


# ── Baseline 1: Store-All ──────────────────────────────────────────────────

class StoreAllController(BaseController):
    def __init__(self, **kw):
        super().__init__(ControllerConfig(name='store_all'), **kw)

    def decide(self, item, state, current_ts, tool_index):
        item.score = 1.0
        return 'PERSIST'


# ── Baseline 2: Relevance-Only ─────────────────────────────────────────────

class RelevanceOnlyController(BaseController):
    def __init__(self, threshold: float = 0.58, **kw):
        super().__init__(ControllerConfig(name='relevance_only',
                                          persist_threshold=threshold), **kw)

    def decide(self, item, state, current_ts, tool_index):
        item.score = item.utility
        return 'PERSIST' if item.utility >= self.config.persist_threshold else 'SKIP'


# ── Baseline 3: Novelty / Recency ──────────────────────────────────────────

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


# ── Baseline 4: A-MAC-lite ─────────────────────────────────────────────────

class AMACLiteController(BaseController):
    """Reimplementation of A-MAC 5-factor admission control:
    utility, confidence, novelty, recency, type_prior.
    """
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


# ── Our method: DAC-Mem ────────────────────────────────────────────────────

class DACMemController(BaseController):
    """Derivability-Aware Controller for Persistent Memory.

    Core formula:
        persist_score = α·U - β·D - γ·S + δ·T + ...
        ephemeral_score = ...

    Three-way decision: PERSIST / EPHEMERAL / SKIP.

    Supports three probe modes for derivability:
        'embedding'      — cosine similarity against tool index (default)
        'llm_judge'      — LLM rates derivability (lightweight)
        'tool_grounded'  — bounded recovery agent (full version)
    """

    def __init__(self, config: Optional[ControllerConfig] = None,
                 llm=None, **kw):
        cfg = config or ControllerConfig(
            name='dac_mem',
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
        self._llm_probe: Optional[LLMDerivabilityProbe] = None
        self._grounded_probe: Optional[ToolGroundedRecoveryProbe] = None

    def _get_llm_probe(self) -> LLMDerivabilityProbe:
        if self._llm_probe is None:
            assert self.llm is not None, \
                "LLM required for probe_mode='llm_judge'"
            self._llm_probe = LLMDerivabilityProbe(self.llm)
        return self._llm_probe

    def _get_grounded_probe(self, tool_index: ToolIndex) -> ToolGroundedRecoveryProbe:
        if self._grounded_probe is None:
            assert self.llm is not None, \
                "LLM required for probe_mode='tool_grounded'"
            self._grounded_probe = ToolGroundedRecoveryProbe(
                self.llm, tool_index)
        return self._grounded_probe

    def score_item(self, item: MemoryItem, state: ControllerState,
                   current_ts: str, tool_index: ToolIndex) -> MemoryItem:
        item = super().score_item(item, state, current_ts, tool_index)
        relevant_tools = relevant_tools_for_type(item.memory_type)
        mode = self.config.probe_mode

        # 1. Embedding probe (always computed as base)
        emb_score, tool_scores = tool_index.probe_recoverability(
            item, relevant_tools=relevant_tools)
        item.metadata['tool_scores'] = tool_scores

        # 2. LLM judge (if requested)
        if mode == 'llm_judge' and self.llm is not None:
            llm_score = self._get_llm_probe().probe(item)
            item.llm_derivability = llm_score
            # blend: 0.6 LLM + 0.4 embedding
            item.derivability = max(item.derivability,
                                    0.6 * llm_score + 0.4 * emb_score)

        # 3. Tool-grounded recovery probe (if requested)
        elif mode == 'tool_grounded' and self.llm is not None:
            probe = self._get_grounded_probe(tool_index)
            grounded_score, probe_info = probe.probe(item, relevant_tools)
            item.probe_derivability = grounded_score
            item.metadata['probe_info'] = probe_info
            # blend: 0.7 grounded + 0.3 embedding
            item.derivability = max(item.derivability,
                                    0.7 * grounded_score + 0.3 * emb_score)

        # 4. Embedding-only (default)
        else:
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
    key = name.lower()
    kw = {'embedder': embedder} if embedder else {}
    if key in ('store_all', 'storeall'):
        return StoreAllController(**kw)
    if key in ('relevance', 'relevance_only'):
        return RelevanceOnlyController(**kw)
    if key in ('novelty', 'novelty_recency'):
        return NoveltyRecencyController(**kw)
    if key in ('amac', 'amac_lite'):
        return AMACLiteController(**kw)
    if key in ('dac', 'dac_mem', 'dacmem'):
        return DACMemController(llm=llm, **kw)
    raise ValueError(f'Unknown controller: {name}')
