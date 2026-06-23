"""Tests for knowledge.embed — Embedder protocol + SentenceTransformerEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
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
        mock_load.return_value = {"model": None}
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
        MockST.side_effect = lambda model_name, device=None: MagicMock(
            model_name=model_name,
            _device=device or "cpu",
        )

        get_embedder(model_name="test-m", device="cpu")
        assert MockST.call_count == 1

        get_embedder(model_name="test-m", device="cuda")
        assert MockST.call_count == 2  # device changed → recreate

        get_embedder(model_name="test-m", device="cuda")
        assert MockST.call_count == 2  # same device → no recreate
