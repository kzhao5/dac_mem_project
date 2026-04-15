from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .features import FeatureComputer
from .schema import ControllerState, MemoryItem
from .tool_index import ToolEntry, ToolIndex, relevant_tools_for_type


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
    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig(name=self.__class__.__name__)
        self.feats = FeatureComputer()

    @property
    def name(self) -> str:
        return self.config.name

    def score_item(self, item: MemoryItem, state: ControllerState, current_ts: str, tool_index: ToolIndex) -> MemoryItem:
        item.novelty = self.feats.compute_novelty(item, state.persistent)
        item.recency = self.feats.compute_recency(item, current_ts)
        item.utility = self.feats.compute_utility(item)
        item.stale_risk = self.feats.compute_stale_risk(item)
        return item

    def decide(self, item: MemoryItem, state: ControllerState, current_ts: str, tool_index: ToolIndex) -> str:
        raise NotImplementedError

    def apply(self, item: MemoryItem, state: ControllerState, current_ts: str, tool_index: ToolIndex) -> str:
        item = self.score_item(item, state, current_ts, tool_index)
        decision = self.decide(item, state, current_ts, tool_index)
        item.action = decision
        if decision == 'PERSIST':
            state.persistent.append(item)
        elif decision == 'EPHEMERAL':
            state.ephemeral.append(item)
        return decision


class StoreAllController(BaseController):
    def __init__(self):
        super().__init__(ControllerConfig(name='store_all'))

    def decide(self, item: MemoryItem, state: ControllerState, current_ts: str, tool_index: ToolIndex) -> str:
        item.score = 1.0
        return 'PERSIST'


class RelevanceOnlyController(BaseController):
    def __init__(self, threshold: float = 0.58):
        super().__init__(ControllerConfig(name='relevance_only', persist_threshold=threshold))

    def decide(self, item: MemoryItem, state: ControllerState, current_ts: str, tool_index: ToolIndex) -> str:
        item.score = item.utility
        return 'PERSIST' if item.utility >= self.config.persist_threshold else 'SKIP'


class NoveltyRecencyController(BaseController):
    def __init__(self, threshold: float = 0.55):
        super().__init__(ControllerConfig(name='novelty_recency', persist_threshold=threshold, novelty_weight=0.65, recency_weight=0.35))

    def decide(self, item: MemoryItem, state: ControllerState, current_ts: str, tool_index: ToolIndex) -> str:
        score = self.config.novelty_weight * item.novelty + self.config.recency_weight * item.recency
        item.score = score
        return 'PERSIST' if score >= self.config.persist_threshold else 'SKIP'


class AMACLiteController(BaseController):
    """
    Lightweight reimplementation of A-MAC style weighted admission using
    utility, confidence, novelty, recency, and type prior.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None, threshold: float = 0.62):
        cfg = ControllerConfig(name='amac_lite', persist_threshold=threshold)
        super().__init__(cfg)
        self.weights = weights or {
            'utility': 0.38,
            'confidence': 0.14,
            'novelty': 0.22,
            'recency': 0.10,
            'type_prior': 0.16,
        }

    def decide(self, item: MemoryItem, state: ControllerState, current_ts: str, tool_index: ToolIndex) -> str:
        score = (
            self.weights['utility'] * item.utility +
            self.weights['confidence'] * item.confidence +
            self.weights['novelty'] * item.novelty +
            self.weights['recency'] * item.recency +
            self.weights['type_prior'] * item.type_prior
        )
        item.score = float(score)
        return 'PERSIST' if score >= self.config.persist_threshold else 'SKIP'


class DACMemController(BaseController):
    def __init__(self, config: Optional[ControllerConfig] = None):
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
        super().__init__(cfg)

    def score_item(self, item: MemoryItem, state: ControllerState, current_ts: str, tool_index: ToolIndex) -> MemoryItem:
        item = super().score_item(item, state, current_ts, tool_index)
        relevant_tools = relevant_tools_for_type(item.memory_type)
        probe_score, tool_scores = tool_index.probe_recoverability(item, relevant_tools=relevant_tools)
        item.derivability = max(item.derivability, probe_score)
        item.metadata['tool_scores'] = tool_scores
        return item

    def decide(self, item: MemoryItem, state: ControllerState, current_ts: str, tool_index: ToolIndex) -> str:
        duplicate_penalty = 0.2 if (1.0 - item.novelty) >= self.config.duplicate_threshold else 0.0
        persist_score = (
            self.config.utility_weight * item.utility +
            self.config.confidence_weight * item.confidence +
            self.config.novelty_weight * item.novelty +
            self.config.recency_weight * item.recency +
            self.config.type_weight * item.type_prior -
            self.config.derivability_weight * item.derivability -
            self.config.stale_weight * item.stale_risk -
            duplicate_penalty
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


def controller_from_name(name: str):
    key = name.lower()
    if key in {'store_all', 'storeall'}:
        return StoreAllController()
    if key in {'relevance', 'relevance_only'}:
        return RelevanceOnlyController()
    if key in {'novelty', 'novelty_recency'}:
        return NoveltyRecencyController()
    if key in {'amac', 'amac_lite'}:
        return AMACLiteController()
    if key in {'dac', 'dac_mem', 'dacmem'}:
        return DACMemController()
    raise ValueError(f'Unknown controller: {name}')


def tune_amac_weights(examples, candidate_weight_grid: Optional[Dict[str, List[float]]] = None) -> Dict[str, float]:
    """
    Lightweight simplex search over A-MAC weights. Returns normalized weights.
    This function intentionally stays simple and fast.
    """
    grid = candidate_weight_grid or {
        'utility': [0.30, 0.38, 0.45],
        'confidence': [0.10, 0.14, 0.18],
        'novelty': [0.18, 0.22, 0.28],
        'recency': [0.08, 0.10, 0.14],
        'type_prior': [0.12, 0.16, 0.22],
    }
    best = None
    best_score = -1e9
    for values in product(*grid.values()):
        weights = dict(zip(grid.keys(), values))
        total = sum(weights.values())
        if total <= 0:
            continue
        weights = {k: v / total for k, v in weights.items()}
        # Proxy objective: favor utility and novelty, avoid over-reliance on recency.
        proxy = 0.45 * weights['utility'] + 0.30 * weights['novelty'] + 0.15 * weights['type_prior'] + 0.10 * weights['confidence'] - 0.05 * weights['recency']
        if proxy > best_score:
            best_score = proxy
            best = weights
    assert best is not None
    return best
