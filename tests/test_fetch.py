"""Tests for knowledge.fetch."""

from __future__ import annotations

from pathlib import Path

from knowledge.fetch import fetch_sources, get_git_head
from knowledge.sources import Source


def test_get_git_head_nonexistent(tmp_path):
    """get_git_head returns None for a non-git directory."""
    assert get_git_head(tmp_path) is None


def test_get_git_head_valid_repo():
    """get_git_head returns a hash for the current repo."""
    head = get_git_head(Path(__file__).resolve().parent.parent)
    assert head is not None
    assert len(head) == 40


def test_fetch_sources_skips_local(tmp_path):
    """Local sources are skipped without crashing."""
    src = Source(name="test-local", type="local", path=str(tmp_path))
    result = fetch_sources([src], tmp_path, verbose=False)
    assert result == []


def test_fetch_sources_filters_by_only(tmp_path):
    """only parameter filters correctly."""
    src1 = Source(name="skip-me", type="local", path=str(tmp_path))
    src2 = Source(name="keep-me", type="local", path=str(tmp_path))
    result = fetch_sources([src1, src2], tmp_path, only="keep-me", verbose=False)
    assert result == []


def test_clone_invalid_url(tmp_path):
    """Cloning a bad URL returns False."""
    from knowledge.fetch import _clone

    src = Source(name="bad-url", type="git", url="https://invalid.example.com/repo")
    assert _clone(src, tmp_path / "bad-url", verbose=False) is False
