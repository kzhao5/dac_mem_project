"""
Additional baseline controllers beyond the four in controllers.py.

  - MemoryBankController  — timestamp-weighted (MemoryBank-style)
  - MemoryR1Controller    — LLM-prompted CRUD manager (Memory-R1 inspired)
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from .controllers import BaseController, ControllerConfig
from .schema import ControllerState, MemoryItem
from .tool_index import ToolIndex

# ═══════════════════════════════════════════════════════════════════════════
# MemoryBank-style baseline
# ═══════════════════════════════════════════════════════════════════════════

class MemoryBankController(BaseController):
    """MemoryBank-style: persist everything but weight by recency and
    speaker importance during retrieval. At write time, it admits anything
    with non-trivial length and novelty, mimicking a passive append store
    with light deduplication.

    This baseline tests: "does a recency-weighted store-all already work?"
    """

    def __init__(self, novelty_threshold: float = 0.3, **kw):
        cfg = ControllerConfig(name='memory_bank', persist_threshold=0.0,
                               duplicate_threshold=0.92)
        super().__init__(cfg, **kw)
        self.novelty_threshold = novelty_threshold

    def decide(self, item: MemoryItem, state: ControllerState,
               current_ts: str, tool_index: ToolIndex) -> str:
        # near-duplicate → skip
        if item.novelty < self.novelty_threshold:
            return 'SKIP'
        item.score = 0.5 * item.recency + 0.3 * item.utility + 0.2 * item.novelty
        return 'PERSIST'


# ═══════════════════════════════════════════════════════════════════════════
# Memory-R1-style CRUD manager (LLM-prompted)
# ═══════════════════════════════════════════════════════════════════════════

MEMORY_R1_SYSTEM = """You are a memory manager for a long-horizon agent. For each candidate memory item, decide what action to take.

Available actions:
- ADD: store this item in long-term memory
- UPDATE: this updates an existing memory (add it)
- NOOP: skip this item, it's not worth storing
- DELETE: if it contradicts or obsoletes an existing memory (treat as ADD for the new version)

Consider: Is this information useful for future tasks? Is it novel? Is it likely to change soon?"""

MEMORY_R1_PROMPT = """Current persistent memory ({n_items} items):
{memory_summary}

New candidate:
  Type: {memory_type}
  Speaker: {speaker}
  Text: "{text}"

What action should be taken? Respond with ONLY a JSON object:
{{"action": "ADD"|"UPDATE"|"NOOP"|"DELETE", "reason": "<brief>"}}"""


class MemoryR1Controller(BaseController):
    """Memory-R1-inspired controller: uses an LLM to decide ADD/UPDATE/NOOP.

    This is a re-implementation of the core idea (LLM-driven CRUD) without
    the RL training loop, making it a strong prompted baseline.
    """

    def __init__(self, llm, max_memory_summary: int = 20, **kw):
        cfg = ControllerConfig(name='memory_r1')
        super().__init__(cfg, **kw)
        self.llm = llm
        self.max_summary = max_memory_summary

    def _summarise_memory(self, state: ControllerState) -> str:
        if not state.persistent:
            return "(empty)"
        items = state.persistent[-self.max_summary:]
        lines = [f"  [{i+1}] [{m.memory_type}] {m.text[:120]}"
                 for i, m in enumerate(items)]
        return '\n'.join(lines)

    def decide(self, item: MemoryItem, state: ControllerState,
               current_ts: str, tool_index: ToolIndex) -> str:
        prompt = MEMORY_R1_PROMPT.format(
            n_items=len(state.persistent),
            memory_summary=self._summarise_memory(state),
            memory_type=item.memory_type,
            speaker=item.speaker,
            text=item.text,
        )
        resp = self.llm.generate(prompt, system=MEMORY_R1_SYSTEM,
                                 temperature=0.0, max_tokens=100)
        try:
            clean = resp.strip()
            if clean.startswith('```'):
                clean = clean.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
            parsed = json.loads(clean)
            action = parsed.get('action', 'NOOP').upper()
        except (json.JSONDecodeError, KeyError, ValueError):
            action = 'NOOP'

        if action in ('ADD', 'UPDATE', 'DELETE'):
            item.score = 0.8
            return 'PERSIST'
        item.score = 0.2
        return 'SKIP'
