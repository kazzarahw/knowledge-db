"""Tests for knowledge.embed — Embedder protocol + SentenceTransformerEmbedder."""

from __future__ import annotations

import numpy as np
from knowledge.embed import _resolve_device, SentenceTransformerEmbedder


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
