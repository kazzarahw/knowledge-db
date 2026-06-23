"""Embedder protocol and model-specific implementations."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

_embedder: SentenceTransformerEmbedder | None = None


def get_embedder(
    model_name: str | None = None,
    device: str | None = None,
    config_dir: str | None = None,
) -> SentenceTransformerEmbedder:
    """Return a cached SentenceTransformerEmbedder singleton.

    Loads model name and device from ``config.yaml`` if present under
    *config_dir*, falling back to arguments, then defaults.

    The model is loaded once on first call and reused. This avoids
    3-5s model-load overhead on every ``kdb search`` invocation.
    """
    global _embedder
    if config_dir:
        from knowledge.config import load_config

        cfg = load_config(Path(config_dir))
        if model_name is None:
            cfg_model = cfg.get("model")
            model_name = (
                str(cfg_model)
                if cfg_model is not None
                else "LiquidAI/LFM2.5-Embedding-350M"
            )
        if device is None:
            device = cfg.get("device")
    if model_name is None:
        model_name = "LiquidAI/LFM2.5-Embedding-350M"
    if (
        _embedder is None
        or _embedder.model_name != model_name
        or _embedder._device != device
    ):
        _embedder = SentenceTransformerEmbedder(model_name, device=device)
    return _embedder


def _resolve_device() -> str:
    """Auto-detect best available device. GPU preferred, CPU fallback with warning."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    warnings.warn(
        "No CUDA-capable GPU detected — falling back to CPU. "
        "Embedding will be significantly slower (~5-10x). "
        "Install PyTorch with CUDA support for GPU acceleration.",
        stacklevel=3,
    )
    return "cpu"


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding models. Must provide dim, model_name, embed(), embed_query()."""

    dim: int
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns shape (N, dim), float32."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,), float32."""
        ...


class SentenceTransformerEmbedder(Embedder):
    """Sentence-transformers based embedder.

    Prompt handling uses model's prompt config: embed_query()
    passes prompt_name="query" when the model defines query prompts.
    Models without prompts (e.g., all-MiniLM-L6-v2) encode raw text.
    """

    dim: int
    model_name: str

    def __init__(
        self,
        model_name: str = "LiquidAI/LFM2.5-Embedding-350M",
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        resolved = device if device is not None else _resolve_device()
        self._model = SentenceTransformer(
            model_name,
            device=resolved,
            trust_remote_code=True,
        )
        self.model_name = model_name
        self._device = resolved
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts (no prompt prefix for docs)."""
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string with query prompt if model defines one."""
        return self._model.encode(
            query,
            prompt_name="query" if "query" in self._model.prompts else None,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
