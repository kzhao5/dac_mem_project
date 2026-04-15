"""
Main evaluation pipeline for DAC-Mem.

Orchestrates: extraction → tool index → controller decisions → retrieval
→ reading → evaluation (retrieval + memory quality + QA + LLM judge).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from .controllers import BaseController
from .embedding import BaseEmbedder, get_embedder
from .evaluation import (
    LLMJudge, aggregate, efficiency_metrics, memory_quality_metrics,
    qa_metrics, retrieval_metrics,
)
from .extraction import CandidateExtractor, LLMCandidateExtractor, ToolFactExtractor
from .reader import make_reader
from .retrieval import HybridRetriever
from .schema import ControllerState, EvaluationResult, Example, MemoryItem
from .tool_index import ToolEntry, ToolIndex
from .utils import parse_timestamp


class MemoryPipeline:
    """Full evaluation pipeline for a single controller."""

    def __init__(
        self,
        controller: BaseController,
        top_k: int = 8,
        reader_mode: str = 'heuristic',
        reader_model: Optional[str] = None,
        ephemeral_session_window: int = 2,
        llm=None,
        embedder: Optional[BaseEmbedder] = None,
        judge_llm=None,
        use_llm_extractor: bool = False,
    ):
        self.controller = controller
        emb = embedder or get_embedder('bge-large')
        # extractor
        if use_llm_extractor and llm is not None:
            self.extractor = LLMCandidateExtractor(llm)
        else:
            self.extractor = CandidateExtractor()
        self.tool_extractor = ToolFactExtractor()
        self.retriever = HybridRetriever(embedder=emb)
        self.reader = make_reader(reader_mode, reader_model, llm=llm)
        self.judge = LLMJudge(judge_llm) if judge_llm else None
        self.top_k = top_k
        self.eph_window = ephemeral_session_window

    def _build_state(self, example: Example):
        state = ControllerState()
        tool_index = ToolIndex(self.retriever.embedder)
        turns = sorted(example.turns, key=lambda t: parse_timestamp(t.timestamp))
        for turn in turns:
            candidates = self.extractor.extract(turn)
            for item in candidates:
                # always populate tool index
                entries = [
                    ToolEntry(
                        entry_id=f"tool_{item.memory_id}_{i}",
                        tool_type=f['tool_type'], slot=f['slot'],
                        text=f['value'],
                        source_turn_id=item.turn_id,
                        source_session_id=item.session_id,
                        timestamp=item.timestamp,
                    )
                    for i, f in enumerate(self.tool_extractor.extract(item))
                ]
                tool_index.add_entries(entries)
                self.controller.apply(item, state, current_ts=turn.timestamp,
                                      tool_index=tool_index)
        return state, tool_index

    def _filter_ephemeral(self, example: Example,
                          state: ControllerState) -> List[MemoryItem]:
        if not state.ephemeral:
            return []
        session_order: list = []
        seen: set = set()
        for t in sorted(example.turns, key=lambda x: parse_timestamp(x.timestamp)):
            if t.session_id not in seen:
                session_order.append(t.session_id)
                seen.add(t.session_id)
        keep = set(session_order[-self.eph_window:])
        return [m for m in state.ephemeral if m.session_id in keep]

    def run_example(self, example: Example) -> Dict:
        t0 = time.time()
        state, tool_index = self._build_state(example)
        eph = self._filter_ephemeral(example, state)
        retrieved = self.retriever.retrieve(
            question=example.question,
            persistent_bank=state.persistent,
            ephemeral_bank=eph,
            tool_index=tool_index,
            top_k=self.top_k,
        )
        t1 = time.time()

        result: Dict = {
            'example_id': example.example_id,
            'question': example.question,
            'gold_answer': example.answer,
            'controller': self.controller.name,
            'dataset': example.dataset,
            'retrieval': retrieval_metrics(example, retrieved),
            'memory': memory_quality_metrics(state.persistent, eph),
            'efficiency': efficiency_metrics(t0, t1, retrieved,
                                             len(state.persistent)),
            'retrieved': [r.__dict__ for r in retrieved],
        }

        # QA
        if self.reader is not None:
            prediction = self.reader.answer(example.question, retrieved)
            result['prediction'] = prediction
            result['qa'] = qa_metrics(prediction, example.answer)

            # LLM judge
            if self.judge is not None:
                jr = self.judge.judge(example.question, example.answer,
                                     prediction)
                result['judge'] = {
                    'score': jr.score,
                    'reasoning': jr.reasoning,
                    'judge_model': jr.judge_model,
                }
                result['qa']['judge_score'] = jr.score
        return result


def run_controller_on_examples(
    controller: BaseController,
    examples: Sequence[Example],
    dataset_name: str,
    top_k: int = 8,
    reader_mode: str = 'heuristic',
    reader_model: Optional[str] = None,
    llm=None,
    embedder: Optional[BaseEmbedder] = None,
    judge_llm=None,
    use_llm_extractor: bool = False,
    limit: Optional[int] = None,
) -> EvaluationResult:
    """Run a controller on a list of examples and return aggregated metrics."""
    pipe = MemoryPipeline(
        controller, top_k=top_k,
        reader_mode=reader_mode, reader_model=reader_model,
        llm=llm, embedder=embedder, judge_llm=judge_llm,
        use_llm_extractor=use_llm_extractor,
    )
    subset = list(examples[:limit] if limit else examples)
    per_example: List[Dict] = []
    for ex in tqdm(subset, desc=f'{controller.name}:{dataset_name}'):
        per_example.append(pipe.run_example(ex))

    # aggregate all metric dicts
    metric_rows: list = []
    for row in per_example:
        merged: Dict = {}
        merged.update(row.get('retrieval', {}))
        merged.update(row.get('memory', {}))
        merged.update(row.get('efficiency', {}))
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
