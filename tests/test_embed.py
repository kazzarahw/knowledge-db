"""Tests for knowledge.embed — Embedder protocol + SentenceTransformerEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from knowledge.config import Config, EmbedConfig
from knowledge.embed import (
    Embedder,
    SentenceTransformerEmbedder,
    _resolve_device,
    get_embedder,
)


def test_resolve_device_returns_string():
    device = _resolve_device()
    assert device in ("cuda", "cpu")


def test_embedder_has_dim():
    embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    assert embedder.dim == 384
    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_embed_returns_correct_shape():
    embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    result = embedder.embed(["hello world", "test"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 384)
    assert result.dtype == np.float32


def test_embed_query_returns_correct_shape():
    embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    result = embedder.embed_query("hello world")
    assert isinstance(result, np.ndarray)
    assert result.shape == (384,)
    assert result.dtype == np.float32


def test_sentence_transformer_embedder_is_embedder():
    """SentenceTransformerEmbedder conforms to Embedder protocol."""
    embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    assert isinstance(embedder, Embedder)


def test_get_embedder_config_with_null_model():
    """YAML model: null must not produce str(None) = 'None'."""
    with (
        patch("knowledge.config.load_config") as mock_load,
        patch("knowledge.embed.SentenceTransformerEmbedder") as MockST,
    ):
        mock_load.return_value = Config(embed=EmbedConfig(model=None))
        MockST.return_value = MagicMock(model_name="LiquidAI/LFM2.5-Embedding-350M")

        result = get_embedder(config_dir="/tmp/fake")

        model_name_arg = MockST.call_args[0][0]
        assert model_name_arg == "LiquidAI/LFM2.5-Embedding-350M"
        assert "None" not in model_name_arg
        assert result is MockST.return_value


def test_get_embedder_recreates_on_device_change():
    """get_embedder must recreate embedder when device changes."""
    # Clear function-attribute cache before test
    if hasattr(get_embedder, "_cached"):
        del get_embedder._cached
    with patch("knowledge.embed.SentenceTransformerEmbedder") as MockST:
        MockST.side_effect = lambda model_name, device=None, trust_remote_code=True, dtype=None: (
            MagicMock(
                model_name=model_name,
                _device=device or "cpu",
                _dtype=dtype,
            )
        )

        get_embedder(model_name="test-m", device="cpu")
        assert MockST.call_count == 1

        get_embedder(model_name="test-m", device="cuda")
        assert MockST.call_count == 2  # device changed → recreate

        get_embedder(model_name="test-m", device="cuda")
        assert MockST.call_count == 2  # same device → no recreate


def test_model_kwargs_with_flash_attn() -> None:
    """When flash-attn is importable, model_kwargs includes attn_implementation."""
    import sys
    import torch

    with (
        patch.dict("sys.modules", {"flash_attn": MagicMock()}),
        patch("sentence_transformers.SentenceTransformer") as MockST,
        patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        patch("torch.cuda.is_available", return_value=True),
    ):
        MockST.side_effect = lambda model_name, device=None, trust_remote_code=True, model_kwargs=None: (
            MagicMock(
                model_name=model_name,
                _device=device or "cpu",
                dim=384,
                prompts={},
            )
        )
        embedder = SentenceTransformerEmbedder(model_name="test")
        assert embedder is not None
        call_kwargs = MockST.call_args[1]
        assert "model_kwargs" in call_kwargs
        assert call_kwargs["model_kwargs"]["attn_implementation"] == "flash_attention_2"
        assert call_kwargs["model_kwargs"]["torch_dtype"] == torch.bfloat16


def test_model_kwargs_without_flash_attn() -> None:
    """When flash-attn is NOT importable, attn_implementation is absent."""
    import sys
    import torch

    with (
        patch.dict("sys.modules"),
        patch("sentence_transformers.SentenceTransformer") as MockST,
        patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        patch("torch.cuda.is_available", return_value=True),
    ):
        sys.modules.pop("flash_attn", None)
        MockST.side_effect = lambda model_name, device=None, trust_remote_code=True, model_kwargs=None: (
            MagicMock(
                model_name=model_name,
                _device=device or "cpu",
                dim=384,
                prompts={},
            )
        )
        _ = SentenceTransformerEmbedder(model_name="test")
        call_kwargs = MockST.call_args[1]
        assert "model_kwargs" in call_kwargs
        assert "attn_implementation" not in call_kwargs["model_kwargs"]
        assert call_kwargs["model_kwargs"]["torch_dtype"] == torch.bfloat16


def test_bf16_gated_on_gpu_capability() -> None:
    """bf16 is NOT set on pre-Ampere GPUs (compute < 8.0)."""
    import sys

    with (
        patch.dict("sys.modules"),
        patch("sentence_transformers.SentenceTransformer") as MockST,
        patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        patch("torch.cuda.is_available", return_value=True),
    ):
        sys.modules.pop("flash_attn", None)
        MockST.side_effect = lambda model_name, device=None, trust_remote_code=True, model_kwargs=None: (
            MagicMock(model_name=model_name, _device="cuda", dim=384, prompts={})
        )
        _ = SentenceTransformerEmbedder(model_name="test", device="cuda")
        call_kwargs = MockST.call_args[1]
        model_kw = call_kwargs.get("model_kwargs", {})
        assert "torch_dtype" not in model_kw  # no bf16 on pre-Ampere


def test_bf16_enabled_on_ampere_plus() -> None:
    """bf16 IS set on Ampere+ GPUs (compute >= 8.0)."""
    import sys
    import torch

    with (
        patch.dict("sys.modules"),
        patch("sentence_transformers.SentenceTransformer") as MockST,
        patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        patch("torch.cuda.is_available", return_value=True),
    ):
        sys.modules.pop("flash_attn", None)
        MockST.side_effect = lambda model_name, device=None, trust_remote_code=True, model_kwargs=None: (
            MagicMock(model_name=model_name, _device="cuda", dim=384, prompts={})
        )
        _ = SentenceTransformerEmbedder(model_name="test", device="cuda")
        call_kwargs = MockST.call_args[1]
        assert call_kwargs["model_kwargs"]["torch_dtype"] == torch.bfloat16


def test_config_dtype_fp32_disables_bf16_on_ampere() -> None:
    """dtype='fp32' in config disables bf16 even on Ampere+."""
    import sys

    with (
        patch.dict("sys.modules"),
        patch("sentence_transformers.SentenceTransformer") as MockST,
        patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        patch("torch.cuda.is_available", return_value=True),
    ):
        sys.modules.pop("flash_attn", None)
        MockST.side_effect = lambda model_name, device=None, trust_remote_code=True, model_kwargs=None: (
            MagicMock(model_name=model_name, _device="cuda", dim=384, prompts={})
        )
        _ = SentenceTransformerEmbedder(model_name="test", device="cuda", dtype="fp32")
        call_kwargs = MockST.call_args[1]
        model_kw = call_kwargs.get("model_kwargs", {})
        assert "torch_dtype" not in model_kw


def test_config_dtype_bf16_override_on_pre_ampere() -> None:
    """dtype='bf16' forces bf16 even on pre-Ampere GPUs (compute < 8.0)."""
    import sys
    import torch

    with (
        patch.dict("sys.modules"),
        patch("sentence_transformers.SentenceTransformer") as MockST,
        patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        patch("torch.cuda.is_available", return_value=True),
    ):
        sys.modules.pop("flash_attn", None)
        MockST.side_effect = lambda model_name, device=None, trust_remote_code=True, model_kwargs=None: (
            MagicMock(model_name=model_name, _device="cuda", dim=384, prompts={})
        )
        _ = SentenceTransformerEmbedder(model_name="test", device="cuda", dtype="bf16")
        call_kwargs = MockST.call_args[1]
        assert call_kwargs["model_kwargs"]["torch_dtype"] == torch.bfloat16
