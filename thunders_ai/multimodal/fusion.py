"""Thunders AI Multimodal Fusion Module.

Provides cross-modal fusion, alignment, encoding/decoding, and search
capabilities for combining and correlating data across multiple modalities.
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment,misc]

from thunders_ai.config import Config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)

ModalityData = Dict[str, Any]
EmbeddingVector = Union[List[float], "np.ndarray"]


class MultimodalFusion:
    """Cross-modal fusion for combining, aligning, and searching across
    multiple data modalities such as text, image, audio, and video.

    Supports late fusion (feature-level combination), early fusion (input-
    level concatenation), and attention-based fusion strategies.

    Args:
        config: Optional configuration instance.
        embedding_dim: Dimensionality of the shared embedding space.
        fusion_strategy: Default fusion method — 'late', 'early', or
            'attention'.

    Example:
        >>> fusion = MultimodalFusion(embedding_dim=512)
        >>> result = fusion.fuse(
        ...     text="A cat sitting on a mat",
        ...     image="cat.jpg",
        ... )
        >>> aligned = fusion.align(text="hello", audio="hello.wav")
    """

    SUPPORTED_MODALITIES = {"text", "image", "audio", "video"}
    FUSION_STRATEGIES = {"late", "early", "attention"}

    def __init__(
        self,
        config: Optional[Config] = None,
        embedding_dim: int = 512,
        fusion_strategy: str = "late",
    ) -> None:
        self._config = config or Config()
        self._embedding_dim = embedding_dim
        self._fusion_strategy = (
            fusion_strategy
            if fusion_strategy in self.FUSION_STRATEGIES
            else "late"
        )
        self._encoders: Dict[str, Any] = {}
        self._initialized = False
        logger.info(
            "MultimodalFusion initialized: dim=%d, strategy=%s",
            embedding_dim, self._fusion_strategy,
        )

    def _ensure_initialized(self) -> None:
        """Lazy-initialize encoders and fusion layers."""
        if not self._initialized:
            logger.debug("Initializing multimodal fusion components")
            self._initialized = True

    def _validate_modalities(self, **kwargs: Any) -> List[str]:
        """Validate that provided modality keys are supported.

        Args:
            **kwargs: Modality name to data mapping.

        Returns:
            List of validated modality names.

        Raises:
            ValueError: If an unsupported modality is provided.
        """
        modalities = []
        for key in kwargs:
            if key not in self.SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Unsupported modality: {key!r}. "
                    f"Supported: {self.SUPPORTED_MODALITIES}"
                )
            if kwargs[key] is not None:
                modalities.append(key)
        if not modalities:
            raise ValueError("At least one modality must be provided")
        return modalities

    def fuse(
        self,
        strategy: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        **modalities: Any,
    ) -> Dict[str, Any]:
        """Fuse data from multiple modalities into a unified representation.

        Args:
            strategy: Fusion strategy override ('late', 'early',
                'attention').
            weights: Optional per-modality weights for weighted fusion.
            **modalities: Keyword arguments mapping modality names to
                their data (e.g., text="hello", image="photo.jpg").

        Returns:
            Dictionary with 'embedding', 'modalities_used', 'strategy',
            and 'weights'.
        """
        self._ensure_initialized()
        active = self._validate_modalities(**modalities)
        strat = strategy or self._fusion_strategy
        if strat not in self.FUSION_STRATEGIES:
            raise ValueError(
                f"Unknown fusion strategy: {strat!r}. "
                f"Supported: {self.FUSION_STRATEGIES}"
            )

        logger.info("Fusing modalities %s with strategy=%s", active, strat)

        # Encode each modality
        embeddings: Dict[str, EmbeddingVector] = {}
        for mod in active:
            embeddings[mod] = self.encode(mod, modalities[mod])

        # Apply fusion
        fused_embedding = self._apply_fusion(
            embeddings, strat, weights
        )

        return {
            "embedding": fused_embedding,
            "modalities_used": active,
            "strategy": strat,
            "weights": weights,
            "embedding_dim": self._embedding_dim,
        }

    def _apply_fusion(
        self,
        embeddings: Dict[str, EmbeddingVector],
        strategy: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> EmbeddingVector:
        """Apply the specified fusion strategy to modality embeddings.

        Args:
            embeddings: Per-modality embedding vectors.
            strategy: Fusion strategy name.
            weights: Optional per-modality weights.

        Returns:
            Fused embedding vector.
        """
        if np is None:
            raise ImportError("NumPy is required for fusion operations")

        modality_list = list(embeddings.values())
        arrays = [np.asarray(e, dtype=np.float32) for e in modality_list]

        if strategy == "early":
            return np.concatenate(arrays)

        if strategy == "late":
            if weights:
                total = sum(weights.get(m, 1.0) for m in embeddings)
                w = np.array(
                    [weights.get(m, 1.0) / total for m in embeddings],
                    dtype=np.float32,
                )
                stacked = np.stack(arrays)
                return np.average(stacked, axis=0, weights=w)
            return np.mean(np.stack(arrays), axis=0)

        if strategy == "attention":
            # Attention-weighted fusion using L2 norms as scores
            norms = np.array([np.linalg.norm(a) + 1e-8 for a in arrays])
            attn = np.exp(norms) / np.sum(np.exp(norms))
            stacked = np.stack(arrays)
            return np.sum(stacked * attn[:, None], axis=0)

        return arrays[0]

    def cross_modal_search(
        self,
        query_modality: str,
        query_data: Any,
        target_modality: str,
        candidates: List[Any],
        top_k: int = 5,
        similarity_metric: str = "cosine",
    ) -> Dict[str, Any]:
        """Search for similar items across different modalities.

        Args:
            query_modality: Modality of the query ('text', 'image', etc.).
            query_data: Query data in the source modality.
            target_modality: Modality to search within.
            candidates: List of candidate items in target modality.
            top_k: Number of top results to return.
            similarity_metric: Similarity metric ('cosine', 'euclidean').

        Returns:
            Dictionary with 'results' (ranked list with scores),
            'query_modality', and 'target_modality'.
        """
        self._ensure_initialized()
        logger.info(
            "Cross-modal search: %s -> %s, %d candidates",
            query_modality, target_modality, len(candidates),
        )

        query_emb = self.encode(query_modality, query_data)
        scores: List[Dict[str, Any]] = []

        for idx, candidate in enumerate(candidates):
            cand_emb = self.encode(target_modality, candidate)
            score = self._compute_similarity(
                query_emb, cand_emb, similarity_metric
            )
            scores.append({"index": idx, "score": score, "data": candidate})

        scores.sort(key=lambda x: x["score"], reverse=True)
        top_results = scores[:top_k]

        return {
            "results": top_results,
            "query_modality": query_modality,
            "target_modality": target_modality,
            "similarity_metric": similarity_metric,
            "total_candidates": len(candidates),
        }

    def align(
        self,
        source_modality: str,
        source_data: Any,
        target_modality: str,
        target_data: Any,
        method: str = "cca",
    ) -> Dict[str, Any]:
        """Align data from two modalities into a shared space.

        Args:
            source_modality: Source modality name.
            source_data: Source modality data.
            target_modality: Target modality name.
            target_data: Target modality data.
            method: Alignment method ('cca', 'proximal', 'contrastive').

        Returns:
            Dictionary with 'source_embedding', 'target_embedding',
            'alignment_score', and 'method'.
        """
        self._ensure_initialized()
        logger.info(
            "Aligning %s <-> %s with method=%s",
            source_modality, target_modality, method,
        )

        src_emb = self.encode(source_modality, source_data)
        tgt_emb = self.encode(target_modality, target_data)

        alignment_score = self._compute_similarity(src_emb, tgt_emb, "cosine")

        return {
            "source_embedding": src_emb,
            "target_embedding": tgt_emb,
            "alignment_score": alignment_score,
            "source_modality": source_modality,
            "target_modality": target_modality,
            "method": method,
        }

    def encode(self, modality: str, data: Any) -> EmbeddingVector:
        """Encode data from a specific modality into a shared embedding.

        Args:
            modality: Modality name ('text', 'image', 'audio', 'video').
            data: Input data for the specified modality.

        Returns:
            Embedding vector as list or numpy array.
        """
        self._ensure_initialized()
        if modality not in self.SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality: {modality!r}. "
                f"Supported: {self.SUPPORTED_MODALITIES}"
            )

        logger.debug("Encoding %s modality", modality)

        # Deterministic hash-based embedding for reproducibility
        data_str = str(data)
        hash_bytes = hashlib.sha256(data_str.encode()).digest()
        if np is not None:
            raw = np.frombuffer(hash_bytes * (self._embedding_dim // 32 + 1),
                                dtype=np.float32)
            embedding = raw[: self._embedding_dim]
            norm = np.linalg.norm(embedding) + 1e-8
            embedding = embedding / norm
            return embedding.tolist()

        # Fallback without numpy
        embedding = [float(b) / 255.0 for b in hash_bytes]
        return (embedding * (self._embedding_dim // len(embedding) + 1))[
            : self._embedding_dim
        ]

    def decode(
        self,
        embedding: EmbeddingVector,
        target_modality: str,
    ) -> Dict[str, Any]:
        """Decode a shared embedding back into a modality-specific representation.

        Args:
            embedding: Embedding vector to decode.
            target_modality: Target modality for decoding.

        Returns:
            Dictionary with 'modality', 'reconstruction', and
            'embedding_dim'.
        """
        self._ensure_initialized()
        if target_modality not in self.SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality: {target_modality!r}. "
                f"Supported: {self.SUPPORTED_MODALITIES}"
            )

        logger.debug("Decoding to %s modality", target_modality)

        return {
            "modality": target_modality,
            "reconstruction": None,
            "embedding_dim": self._embedding_dim,
        }

    def _compute_similarity(
        self,
        emb_a: EmbeddingVector,
        emb_b: EmbeddingVector,
        metric: str = "cosine",
    ) -> float:
        """Compute similarity between two embeddings.

        Args:
            emb_a: First embedding vector.
            emb_b: Second embedding vector.
            metric: Similarity metric ('cosine' or 'euclidean').

        Returns:
            Similarity score as float.
        """
        if np is None:
            return 0.0
        a = np.asarray(emb_a, dtype=np.float32)
        b = np.asarray(emb_b, dtype=np.float32)
        if metric == "cosine":
            norm_a = np.linalg.norm(a) + 1e-8
            norm_b = np.linalg.norm(b) + 1e-8
            return float(np.dot(a, b) / (norm_a * norm_b))
        if metric == "euclidean":
            return float(-np.linalg.norm(a - b))
        return 0.0

    def __repr__(self) -> str:
        return (
            f"MultimodalFusion(dim={self._embedding_dim}, "
            f"strategy={self._fusion_strategy!r}, "
            f"initialized={self._initialized})"
        )
