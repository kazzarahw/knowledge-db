"""Embedder protocol and model-specific implementations."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from knowledge.config import DEFAULT_MODEL


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

    Cache is stored as a function attribute to avoid module-level
    mutable state (``_embedder`` would be a global anti-pattern).
    """
    from knowledge.config import load_config

    cfg = None
    trust_remote_code = True
    if config_dir:
        cfg = load_config(Path(config_dir))
        if model_name is None:
            model_name = cfg.embed.model or DEFAULT_MODEL
        if device is None:
            device = cfg.embed.device
        trust_remote_code = cfg.embed.trust_remote_code
    if model_name is None:
        model_name = DEFAULT_MODEL
    dtype = cfg.embed.dtype if cfg is not None else None
    cached = getattr(get_embedder, "_cached", None)
    if cached is not None and cached.model_name == model_name:
        if (device is None or cached._device == device) and cached._dtype == dtype:
            return cached

    cached = SentenceTransformerEmbedder(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )
    get_embedder._cached = cached
    return cached


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
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        trust_remote_code: bool = True,
        dtype: str | None = None,
    ) -> None:
        import logging
        import torch
        from sentence_transformers import SentenceTransformer

        _log = logging.getLogger(__name__)
        resolved = device if device is not None else _resolve_device()
        model_kwargs: dict[str, object] = {}
        _dtype = dtype  # resolved by caller from config.yaml

        if _dtype is None or _dtype == "auto":
            if resolved.startswith("cuda") and torch.cuda.is_available():
                # Resolve device index for get_device_capability
                # "cuda:0" → 0, "cuda" → None (current device)
                dev_idx: int | None = (
                    int(resolved.split(":", 1)[1]) if ":" in resolved else None
                )
                cap = torch.cuda.get_device_capability(dev_idx)
                if cap is not None and cap[0] >= 8:  # Ampere+
                    model_kwargs["torch_dtype"] = torch.bfloat16
            # CPU bf16 is opt-in via dtype="bf16" in config.yaml — skip auto on CPU
            # because older CPUs emulate bf16 in software (slower than fp32)
        elif _dtype == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        # else _dtype == "fp32" or "auto" on pre-Ampere → fp32 default (no kwarg)

        # Flash Attention 2 — gated on importability
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

        try:
            self._model = SentenceTransformer(
                model_name,
                device=resolved,
                trust_remote_code=trust_remote_code,
                model_kwargs=model_kwargs,
            )
        except Exception as exc:
            if "attn_implementation" in model_kwargs:
                _log.debug(
                    "SentenceTransformer load failed with FA2 (%.200s) — retrying without attn_implementation",
                    exc,
                )
                del model_kwargs["attn_implementation"]
                self._model = SentenceTransformer(
                    model_name,
                    device=resolved,
                    trust_remote_code=trust_remote_code,
                    model_kwargs=model_kwargs,
                )
            else:
                raise
        # Monkey-patch for LiquidAI/LFM2.5-Embedding-350M compatibility.
        # transformers v5.12+ passes ``seq_idx`` as a kwarg to decoder-layer
        # forward methods.  The custom model's ``_noncausal_shortconv_forward``
        # does not accept ``**kwargs``, so the unexpected kwarg crashes
        # ``encode()`` with::
        #
        #   TypeError: _noncausal_shortconv_forward() got an unexpected
        #   keyword argument 'seq_idx'
        #
        # The patch wraps ``Lfm2ShortConv.slow_forward`` (which after model
        # loading is ``_noncausal_shortconv_forward``) to discard unexpected
        # kwargs before forwarding.
        #
        # NOTE: this MUST run *after* ``SentenceTransformer(...)`` because the
        # custom model's ``_install_patches()`` replaces ``slow_forward`` at
        # module-import time, which would overwrite any prior patch.
        if "lfm2" in model_name.lower() or "liquid" in model_name.lower():
            import transformers.models.lfm2.modeling_lfm2 as _lfm2_mod

            _orig_slow_forward = _lfm2_mod.Lfm2ShortConv.slow_forward

            def _patched_slow_forward(
                self,
                hidden_states: torch.Tensor,
                *args: object,
                **kwargs: object,
            ) -> torch.Tensor:
                kept = {
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in {
                        "past_key_values",
                        "cache_position",
                        "attention_mask",
                    }
                }
                return _orig_slow_forward(self, hidden_states, *args, **kept)

            _lfm2_mod.Lfm2ShortConv.slow_forward = _patched_slow_forward
        self.model_name = model_name
        self._device = resolved
        self._dtype = dtype
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
