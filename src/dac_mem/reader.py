"""
QA readers — produce an answer string from retrieved context.

  - HeuristicReader    : returns top-1 retrieved text (baseline)
  - HFExtractiveReader : DistilBERT extractive QA (lightweight)
  - LLMReader          : GPT-4o / Claude / Gemini / Qwen reader (main)
"""
from __future__ import annotations

from typing import Optional, Sequence

from .schema import RetrievedItem

# ── LLM reader prompt ─────────────────────────────────────────────────────

LLM_READER_SYSTEM = """You are a precise question-answering assistant. Answer the question using ONLY the provided context. If the context does not contain enough information, say "I don't know". Be concise — answer in one sentence or a short phrase when possible."""

LLM_READER_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


# ── readers ────────────────────────────────────────────────────────────────

class HeuristicReader:
    """Returns the top-1 retrieved sentence verbatim."""
    def answer(self, question: str, retrieved: Sequence[RetrievedItem]) -> str:
        return retrieved[0].text if retrieved else "I don't know"


class HFExtractiveReader:
    """DistilBERT extractive QA — fast but limited to span extraction."""
    def __init__(self, model_name: str = 'distilbert-base-cased-distilled-squad'):
        from transformers import pipeline
        self.pipe = pipeline('question-answering', model=model_name,
                             tokenizer=model_name)

    def answer(self, question: str, retrieved: Sequence[RetrievedItem]) -> str:
        if not retrieved:
            return "I don't know"
        ctx = ' '.join(r.text for r in retrieved[:8])
        pred = self.pipe(question=question, context=ctx)
        ans = pred.get('answer', '').strip()
        return ans or "I don't know"


class LLMReader:
    """Full LLM reader — the main reader for EMNLP experiments."""
    def __init__(self, llm, max_context_items: int = 10):
        self.llm = llm
        self.max_items = max_context_items

    def answer(self, question: str, retrieved: Sequence[RetrievedItem]) -> str:
        if not retrieved:
            return "I don't know"
        ctx_parts = [f"[{i+1}] {r.text}" for i, r in
                     enumerate(retrieved[:self.max_items])]
        ctx = '\n'.join(ctx_parts)
        prompt = LLM_READER_PROMPT.format(context=ctx, question=question)
        resp = self.llm.generate(prompt, system=LLM_READER_SYSTEM,
                                 temperature=0.0, max_tokens=256)
        return resp.strip() or "I don't know"


# ── factory ────────────────────────────────────────────────────────────────

def make_reader(mode: str = 'heuristic', model_name: Optional[str] = None,
                llm=None):
    """Create a reader.

    Parameters
    ----------
    mode : 'none', 'heuristic', 'hf_extractive', 'llm'
    model_name : HF model name for extractive reader
    llm : BaseLLM instance (required when mode='llm')
    """
    if mode == 'none':
        return None
    if mode == 'heuristic':
        return HeuristicReader()
    if mode == 'hf_extractive':
        return HFExtractiveReader(model_name or 'distilbert-base-cased-distilled-squad')
    if mode == 'llm':
        if llm is None:
            raise ValueError("mode='llm' requires an llm argument")
        return LLMReader(llm)
    raise ValueError(f'Unknown reader mode: {mode}')
