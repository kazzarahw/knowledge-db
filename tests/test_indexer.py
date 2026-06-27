"""Tests for knowledge.indexer — indexing pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from knowledge.chunk import Section


def test_index_source_sorts_sections_by_length(tmp_path: Path) -> None:
    """Sections should be sorted by body length before embedding."""
    from knowledge.indexer import _index_source
    from knowledge.sources import Source

    source = Source(
        name="test",
        source_type="git",
        url="https://example.com/repo.git",
        category="docs",
        title="Test",
    )
    embedder = MagicMock()
    embedder.embed.return_value = np.zeros((4, 384), dtype=np.float32)
    conn = MagicMock()

    data_dir = tmp_path
    (data_dir / "sources" / "test").mkdir(parents=True)

    sections = [
        Section(
            source="test",
            title="long",
            category="docs",
            path="a.md",
            heading_path="",
            body="x" * 1000,
        ),
        Section(
            source="test",
            title="short",
            category="docs",
            path="a.md",
            heading_path="",
            body="x" * 10,
        ),
        Section(
            source="test",
            title="medium",
            category="docs",
            path="a.md",
            heading_path="",
            body="x" * 100,
        ),
        Section(
            source="test",
            title="tiny",
            category="docs",
            path="a.md",
            heading_path="",
            body="x",
        ),
    ]

    with (
        patch(
            "knowledge.indexer._walk_files",
            return_value=[data_dir / "sources" / "test" / "a.md"],
        ),
        patch("knowledge.indexer.chunk_file") as mock_chunk,
        patch("knowledge.indexer.load_config") as mock_cfg,
    ):
        mock_chunk.side_effect = lambda fpath, src, cat, rel_path: sections
        mock_cfg.return_value = MagicMock(embed=MagicMock(batch_size=32))

        _index_source(source, embedder, conn, data_dir, verbose=False)
        called_texts = embedder.embed.call_args[0][0]
        assert len(called_texts) == 4
        assert called_texts[0] == "x"  # shortest
        assert called_texts[1] == "x" * 10
        assert called_texts[2] == "x" * 100
        assert called_texts[3] == "x" * 1000  # longest
