from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from .controllers import BaseController
from .evaluation import aggregate, memory_quality_metrics, qa_metrics, retrieval_metrics
from .extraction import CandidateExtractor, ToolFactExtractor
from .reader import make_reader
from .retrieval import HybridRetriever
from .schema import ControllerState, EvaluationResult, Example, MemoryItem
from .tool_index import ToolEntry, ToolIndex
from .utils import parse_timestamp


class MemoryPipeline:
    def __init__(
        self,
        controller: BaseController,
        top_k: int = 8,
        reader_mode: str = 'none',
        reader_model: Optional[str] = None,
        ephemeral_session_window: int = 2,
    ):
        self.controller = controller
        self.extractor = CandidateExtractor()
        self.tool_extractor = ToolFactExtractor()
        self.retriever = HybridRetriever()
        self.reader = make_reader(reader_mode, reader_model)
        self.top_k = top_k
        self.ephemeral_session_window = ephemeral_session_window

    def _build_state(self, example: Example) -> tuple[ControllerState, ToolIndex]:
        state = ControllerState()
        tool_index = ToolIndex()
        turns = sorted(example.turns, key=lambda t: parse_timestamp(t.timestamp))
        for turn in turns:
            candidates = self.extractor.extract(turn)
            for item in candidates:
                # update external tools regardless of controller decision
                tool_entries = [
                    ToolEntry(
                        entry_id=f"tool_{item.memory_id}_{idx}",
                        tool_type=fact['tool_type'],
                        slot=fact['slot'],
                        text=fact['value'],
                        source_turn_id=item.turn_id,
                        source_session_id=item.session_id,
                        timestamp=item.timestamp,
                    )
                    for idx, fact in enumerate(self.tool_extractor.extract(item))
                ]
                tool_index.add_entries(tool_entries)
                self.controller.apply(item, state, current_ts=turn.timestamp, tool_index=tool_index)
        return state, tool_index

    def _filter_ephemeral(self, example: Example, state: ControllerState) -> List[MemoryItem]:
        if not state.ephemeral:
            return []
        # keep only items from last N sessions relative to question time
        session_order = []
        seen = set()
        for t in sorted(example.turns, key=lambda x: parse_timestamp(x.timestamp)):
            if t.session_id not in seen:
                session_order.append(t.session_id)
                seen.add(t.session_id)
        keep_sessions = set(session_order[-self.ephemeral_session_window:])
        return [m for m in state.ephemeral if m.session_id in keep_sessions]

    def run_example(self, example: Example) -> Dict:
        state, tool_index = self._build_state(example)
        eph = self._filter_ephemeral(example, state)
        retrieved = self.retriever.retrieve(
            question=example.question,
            persistent_bank=state.persistent,
            ephemeral_bank=eph,
            tool_index=tool_index,
            top_k=self.top_k,
        )
        result = {
            'example_id': example.example_id,
            'question': example.question,
            'gold_answer': example.answer,
            'controller': self.controller.name,
            'dataset': example.dataset,
            'retrieval': retrieval_metrics(example, retrieved),
            'memory': memory_quality_metrics(state.persistent, eph),
            'retrieved': [r.__dict__ for r in retrieved],
        }
        if self.reader is not None:
            prediction = self.reader.answer(example.question, retrieved)
            result['prediction'] = prediction
            result['qa'] = qa_metrics(prediction, example.answer)
        return result


def run_controller_on_examples(
    controller: BaseController,
    examples: Sequence[Example],
    dataset_name: str,
    top_k: int = 8,
    reader_mode: str = 'none',
    reader_model: Optional[str] = None,
    limit: Optional[int] = None,
) -> EvaluationResult:
    pipe = MemoryPipeline(controller, top_k=top_k, reader_mode=reader_mode, reader_model=reader_model)
    per_example: List[Dict] = []
    subset = list(examples[:limit] if limit else examples)
    for ex in tqdm(subset, desc=f'{controller.name}:{dataset_name}'):
        per_example.append(pipe.run_example(ex))
    metric_rows = []
    for row in per_example:
        merged = {}
        merged.update(row['retrieval'])
        merged.update(row['memory'])
        if 'qa' in row:
            merged.update(row['qa'])
        metric_rows.append(merged)
    return EvaluationResult(
        controller=controller.name,
        dataset=dataset_name,
        n_examples=len(subset),
        metrics=aggregate(metric_rows),
        per_example=per_example,
    )
