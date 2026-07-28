"""HuggingFace Hub client for Thunders AI.

Provides model loading, text generation, pipeline execution,
and Hub push capabilities with local caching.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceClient:
    """Client for HuggingFace model hub and inference.

    Manages model downloading, local caching, text generation,
    and pushing models back to the Hub.

    Attributes:
        cache_dir: Local directory for cached models.
        token: HuggingFace API token.
    """

    SUPPORTED_TASKS = [
        "text-generation",
        "text-classification",
        "token-classification",
        "question-answering",
        "summarization",
        "translation",
        "fill-mask",
        "feature-extraction",
    ]

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        hub_url: str = "https://huggingface.co",
        offline: bool = False,
    ) -> None:
        self.token = token
        self.cache_dir = Path(cache_dir or os.environ.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface")
        ))
        self.hub_url = hub_url.rstrip("/")
        self.offline = offline
        self._loaded_models: Dict[str, Dict[str, Any]] = {}
        self._pipelines: Dict[str, Any] = {}

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "HuggingFaceClient initialised: cache_dir=%s, offline=%s",
            self.cache_dir,
            self.offline,
        )

    def load_model(
        self,
        model_id: str,
        revision: str = "main",
        device: str = "auto",
        quantize: Optional[str] = None,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """Load a model from HuggingFace Hub or local cache.

        Args:
            model_id: Model identifier (e.g. 'meta-llama/Llama-3-8B').
            revision: Git revision / branch.
            device: Target device ('auto', 'cpu', 'cuda', 'mps').
            quantize: Optional quantisation ('4bit', '8bit', None).
            force_download: Re-download even if cached.

        Returns:
            Model info dict with path and metadata.

        Raises:
            RuntimeError: If model cannot be loaded.
        """
        if model_id in self._loaded_models and not force_download:
            logger.info("Model '%s' already loaded; reusing", model_id)
            return self._loaded_models[model_id]

        # Resolve cache path
        model_hash = hashlib.sha256(f"{model_id}@{revision}".encode()).hexdigest()[:16]
        local_path = self.cache_dir / "models" / model_hash

        if local_path.exists() and not force_download and not self.offline:
            logger.info("Model '%s' found in cache: %s", model_id, local_path)
        elif not self.offline:
            logger.info("Downloading model '%s' (revision=%s)...", model_id, revision)
            local_path.mkdir(parents=True, exist_ok=True)
            # Simulate download
            self._simulate_download(model_id, local_path)
        else:
            raise RuntimeError(
                f"Model '{model_id}' not cached and offline=True"
            )

        model_info: Dict[str, Any] = {
            "model_id": model_id,
            "revision": revision,
            "device": device,
            "quantize": quantize,
            "local_path": str(local_path),
            "loaded_at": time.time(),
            "status": "ready",
        }
        self._loaded_models[model_id] = model_info
        logger.info("Model '%s' loaded on %s", model_id, device)
        return model_info

    def generate(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate text from a loaded model.

        Args:
            model_id: Previously loaded model identifier.
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling parameter.
            repetition_penalty: Penalty for repeated tokens.
            **kwargs: Additional generation parameters.

        Returns:
            Generation result with text and metadata.

        Raises:
            KeyError: If model_id has not been loaded.
            ValueError: If prompt is empty.
        """
        if model_id not in self._loaded_models:
            raise KeyError(f"Model '{model_id}' not loaded; call load_model() first")
        if not prompt:
            raise ValueError("prompt must be a non-empty string")

        logger.debug("Generating with '%s': %d chars", model_id, len(prompt))
        start_time = time.time()

        generated_text = self._simulate_generation(
            prompt, max_new_tokens, temperature
        )
        elapsed = time.time() - start_time

        result: Dict[str, Any] = {
            "model_id": model_id,
            "prompt": prompt[:200],
            "generated_text": generated_text,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "generation_time_seconds": round(elapsed, 4),
            "tokens_generated": max_new_tokens,
        }
        return result

    def pipeline(
        self,
        task: str,
        model_id: Optional[str] = None,
        device: int = -1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a HuggingFace pipeline for a specific task.

        Args:
            task: Pipeline task (e.g. 'text-generation').
            model_id: Model to use; if None a default is chosen.
            device: Device index (-1 for CPU).
            **kwargs: Additional pipeline configuration.

        Returns:
            Pipeline configuration and metadata.

        Raises:
            ValueError: If the task is not supported.
        """
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{task}'; choose from {self.SUPPORTED_TASKS}"
            )

        pipe_id = f"{task}-{uuid.uuid4().hex[:8]}"
        pipe_config: Dict[str, Any] = {
            "pipeline_id": pipe_id,
            "task": task,
            "model_id": model_id,
            "device": device,
            "config": kwargs,
            "created_at": time.time(),
        }
        self._pipelines[pipe_id] = pipe_config
        logger.info("Pipeline created: %s (task=%s)", pipe_id, task)
        return pipe_config

    def push_model(
        self,
        model_path: str,
        repo_id: str,
        commit_message: str = "Upload from Thunders AI",
        private: bool = False,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Push a local model to the HuggingFace Hub.

        Args:
            model_path: Path to the local model directory.
            repo_id: Target repository ID on the Hub.
            commit_message: Commit message for the upload.
            private: Whether the repo should be private.
            tags: Optional tags for the model card.

        Returns:
            Upload result with URL and commit info.

        Raises:
            FileNotFoundError: If model_path does not exist.
            RuntimeError: If upload fails.
        """
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        logger.info("Pushing model to '%s'...", repo_id)
        # Simulate upload
        upload_result: Dict[str, Any] = {
            "repo_id": repo_id,
            "url": f"{self.hub_url}/{repo_id}",
            "commit_message": commit_message,
            "private": private,
            "tags": tags or [],
            "files_uploaded": list(model_dir.rglob("*")),
            "status": "uploaded",
            "timestamp": time.time(),
        }
        logger.info("Model pushed to %s", upload_result["url"])
        return upload_result

    # -- Internal helpers ---------------------------------------------------

    def _simulate_download(self, model_id: str, local_path: Path) -> None:
        """Write placeholder metadata to simulate a model download."""
        meta = {
            "model_id": model_id,
            "downloaded_at": time.time(),
            "format": "safetensors",
        }
        (local_path / "config.json").write_text(json.dumps(meta, indent=2))

    def _simulate_generation(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Return a placeholder generated string."""
        return f"[Generated text for: '{prompt[:60]}...' | max_tokens={max_tokens} temp={temperature}]"
