from __future__ import annotations

from typing import Optional, Sequence

from .schema import RetrievedItem


class HeuristicReader:
    def answer(self, question: str, retrieved: Sequence[RetrievedItem]) -> str:
        if not retrieved:
            return "I don't know"
        best = retrieved[0].text
        # very simple default: return the top retrieved sentence
        return best


class HFExtractiveReader:
    def __init__(self, model_name: str = 'distilbert-base-cased-distilled-squad'):
        from transformers import pipeline
        self.pipe = pipeline('question-answering', model=model_name, tokenizer=model_name)

    def answer(self, question: str, retrieved: Sequence[RetrievedItem]) -> str:
        if not retrieved:
            return "I don't know"
        context = ' '.join([r.text for r in retrieved[:8]])
        pred = self.pipe(question=question, context=context)
        answer = pred.get('answer', '').strip()
        return answer or "I don't know"


def make_reader(mode: str = 'heuristic', model_name: Optional[str] = None):
    if mode == 'none':
        return None
    if mode == 'heuristic':
        return HeuristicReader()
    if mode == 'hf_extractive':
        return HFExtractiveReader(model_name or 'distilbert-base-cased-distilled-squad')
    raise ValueError(f'Unknown reader mode: {mode}')
