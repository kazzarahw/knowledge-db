"""Tests for knowledge.fetch."""

from pathlib import Path

from knowledge.fetch import get_git_head


def test_get_git_head_nonexistent(tmp_path):
    """get_git_head returns None for a non-git directory."""
    assert get_git_head(tmp_path) is None
