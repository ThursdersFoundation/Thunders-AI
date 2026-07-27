"""Embedding engine for text vectorisation and similarity search."""

from __future__ import annotations

import hashlib
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from thunders_ai.config import ThundersConfig
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class EmbeddingEngine:
    """Generates text embeddings and performs similarity search.

    Supports pluggable embedding models, batch embedding, on-disk caching,
    and cosine-similarity based retrieval.

    Args:
        config: ThundersConfig instance.
        model_name: Name of the sentence-transformer / embedding model.

    Example::

        emb = EmbeddingEngine(config, "all-MiniLM-L6-v2")
        vec = emb.embed("Hello world")
        results = emb.search("query", corpus_vectors, top_k=5)
    """

    _MAX_CACHE = 4096

    def __init__(
        self,
        config: ThundersConfig,
        model_name: Optional[str] = None,
    ) -> None:
        self._config = config
        self._model_name = model_name or getattr(config, "embedding_model", "all-MiniLM-L6-v2")
        self._model: Optional[Any] = None
        self._dim: int = getattr(config, "embedding_dim", 384)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_dir: Optional[str] = getattr(config, "embedding_cache_dir", None)

        if self._cache_dir:
            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)

        self._load_model()
        logger.info("EmbeddingEngine ready – model=%s, dim=%d", self._model_name, self._dim)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the embedding model (sentence-transformers or fallback)."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info("Loaded embedding model: %s (dim=%d)", self._model_name, self._dim)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed – using random fallback embeddings."
            )
            self._model = None
        except Exception as exc:
            logger.warning("Failed to load embedding model '%s': %s", self._model_name, exc)
            self._model = None

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate an embedding vector for a single text string.

        Args:
            text: Input text.
            use_cache: Return cached vector when available.

        Returns:
            1-D numpy array of shape ``(dim,)``.
        """
        key = hashlib.md5(text.encode()).hexdigest()
        if use_cache and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        # Disk cache
        if use_cache and self._cache_dir:
            path = Path(self._cache_dir) / f"{key}.pkl"
            if path.exists():
                vec = pickle.loads(path.read_bytes())
                self._cache[key] = vec
                self._cache.move_to_end(key)
                return vec

        vec = self._compute([text])[0]
        if use_cache:
            self._cache[key] = vec
            self._cache.move_to_end(key)
            if len(self._cache) > self._MAX_CACHE:
                self._cache.popitem(last=False)
            if self._cache_dir:
                (Path(self._cache_dir) / f"{key}.pkl").write_bytes(pickle.dumps(vec))
        return vec

    def batch_embed(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of input strings.
            use_cache: Use cached vectors where available.

        Returns:
            2-D numpy array of shape ``(len(texts), dim)``.
        """
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        results: List[Optional[np.ndarray]] = [None] * len(texts)

        if use_cache:
            for i, text in enumerate(texts):
                key = hashlib.md5(text.encode()).hexdigest()
                if key in self._cache:
                    results[i] = self._cache[key]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        if uncached_texts:
            vectors = self._compute(uncached_texts)
            for idx, vec in zip(uncached_indices, vectors):
                results[idx] = vec
                key = hashlib.md5(texts[idx].encode()).hexdigest()
                self._cache[key] = vec

        return np.stack(results)

    def _compute(self, texts: List[str]) -> List[np.ndarray]:
        """Compute raw embeddings, falling back to random vectors."""
        if self._model is not None:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return [embeddings[i] for i in range(len(texts))]
        # Deterministic random fallback
        rng = np.random.default_rng(42)
        return [rng.standard_normal(self._dim).astype(np.float32) for _ in texts]

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def search(
        self,
        query: str,
        corpus: List[Dict[str, Any]],
        top_k: int = 5,
        embedding_key: str = "embedding",
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search a corpus by embedding similarity.

        Args:
            query: Query text.
            corpus: List of dicts, each containing an ``embedding_key``.
            top_k: Number of results to return.
            embedding_key: Key in each corpus dict holding the embedding vector.

        Returns:
            List of ``(index, score, document)`` tuples sorted by score desc.
        """
        query_vec = self.embed(query)
        scored: List[Tuple[int, float, Dict[str, Any]]] = []
        for i, doc in enumerate(corpus):
            vec = doc.get(embedding_key)
            if vec is None:
                continue
            score = self.cosine_similarity(query_vec, np.asarray(vec))
            scored.append((i, score, doc))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared.")
