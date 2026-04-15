"""
Semantic embedding models for DAC-Mem.

Supports sentence-transformers (BGE, GTE, MiniLM), OpenAI embeddings,
and a legacy hashing fallback for ablation.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Encode texts into dense vectors and compute similarities."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return (n, d) float32 array of L2-normalised vectors."""
        ...

    def similarity(self, query: str, candidates: List[str]) -> np.ndarray:
        """Cosine similarities between *query* and each candidate."""
        if not candidates:
            return np.array([], dtype=np.float32)
        vecs = self.encode(candidates + [query])
        q = vecs[-1:]
        c = vecs[:-1]
        return (c @ q.T).ravel()          # already L2-normed

    def max_similarity(self, query: str, candidates: List[str]) -> float:
        sims = self.similarity(query, candidates)
        return float(sims.max()) if len(sims) else 0.0


# ---------------------------------------------------------------------------
# Sentence-Transformers (default for research)
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5',
                 device: str = 'cpu', cache_dir: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device,
                                         cache_folder=cache_dir)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=128)


# ---------------------------------------------------------------------------
# OpenAI embeddings
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = 'text-embedding-3-small',
                 batch_size: int = 512):
        import openai
        self.client = openai.OpenAI()
        self.model = model
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        all_embs: list = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embeddings.create(input=batch, model=self.model)
            all_embs.extend([d.embedding for d in resp.data])
        arr = np.asarray(all_embs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)


# ---------------------------------------------------------------------------
# Legacy hashing (for ablation: "replace real embeddings with bag-of-words")
# ---------------------------------------------------------------------------

class HashingEmbedder(BaseEmbedder):
    """Bag-of-words hashing — used only for ablation comparison."""
    def __init__(self, n_features: int = 2 ** 18):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vec = HashingVectorizer(n_features=n_features,
                                     alternate_sign=False, norm='l2')

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype(np.float32)


# ---------------------------------------------------------------------------
# Disk cache wrapper
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Per-text disk cache (numpy .npy files)."""
    def __init__(self, cache_dir: str = '.cache/embeddings'):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str, model: str) -> Path:
        h = hashlib.md5(f"{model}\n{text}".encode()).hexdigest()
        return self.dir / f"{h}.npy"

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        p = self._path(text, model)
        return np.load(p) if p.exists() else None

    def put(self, text: str, model: str, vec: np.ndarray) -> None:
        np.save(self._path(text, model), vec)


class CachedEmbedder(BaseEmbedder):
    """Wraps any embedder with on-disk caching."""
    def __init__(self, embedder: BaseEmbedder, model_name: str,
                 cache_dir: str = '.cache/embeddings'):
        self.inner = embedder
        self.name = model_name
        self.cache = EmbeddingCache(cache_dir)

    def encode(self, texts: List[str]) -> np.ndarray:
        results: list = [None] * len(texts)
        miss_idx: list = []
        miss_txt: list = []
        for i, t in enumerate(texts):
            v = self.cache.get(t, self.name)
            if v is not None:
                results[i] = v
            else:
                miss_idx.append(i)
                miss_txt.append(t)
        if miss_txt:
            computed = self.inner.encode(miss_txt)
            for j, idx in enumerate(miss_idx):
                results[idx] = computed[j]
                self.cache.put(miss_txt[j], self.name, computed[j])
        return np.stack(results)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ALIASES = {
    'bge-large': 'BAAI/bge-large-en-v1.5',
    'bge': 'BAAI/bge-large-en-v1.5',
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'all-minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'gte-large': 'thenlper/gte-large',
    'gte': 'thenlper/gte-large',
    'bge-small': 'BAAI/bge-small-en-v1.5',
}


def get_embedder(name: str = 'bge-large', device: str = 'cpu',
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True) -> BaseEmbedder:
    """Create an embedder by short name or HuggingFace model id."""
    if name in ('hashing', 'legacy'):
        return HashingEmbedder()
    if name == 'openai' or name.startswith('openai:'):
        model = name.split(':', 1)[1] if ':' in name else 'text-embedding-3-small'
        emb = OpenAIEmbedder(model=model)
    else:
        hf_name = _ALIASES.get(name, name)
        emb = SentenceTransformerEmbedder(hf_name, device=device,
                                          cache_dir=cache_dir)
    if use_cache:
        return CachedEmbedder(emb, name, cache_dir or '.cache/embeddings')
    return emb
