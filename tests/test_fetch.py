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
    src = Source(name="test-local", source_type="local", path=str(tmp_path))
    result = fetch_sources([src], tmp_path, verbose=False)
    assert result == []


def test_fetch_sources_filters_by_only(tmp_path):
    """only parameter filters correctly."""
    src1 = Source(name="skip-me", source_type="local", path=str(tmp_path))
    src2 = Source(name="keep-me", source_type="local", path=str(tmp_path))
    result = fetch_sources([src1, src2], tmp_path, only="keep-me", verbose=False)
    assert result == []


def test_clone_invalid_url(tmp_path):
    """Cloning a bad URL returns False."""
    from knowledge.fetch import _clone

    src = Source(
        name="bad-url", source_type="git", url="https://invalid.example.com/repo"
    )
    assert _clone(src, tmp_path / "bad-url", verbose=False, git_timeout=300) is False


def test_pull_nonexistent_dir_returns_false(tmp_path: Path) -> None:
    """_pull on nonexistent directory returns False, not crash."""
    from knowledge.fetch import _pull

    src = Source(
        name="ghost",
        source_type="git",
        url="https://github.com/user/repo.git",
    )
    assert _pull(src, tmp_path / "ghost", verbose=False, git_timeout=300) is False


def test_pull_local_repo_no_remote(tmp_path: Path) -> None:
    """_pull on a git repo with no remote returns False."""
    import subprocess
    from knowledge.fetch import _pull

    src = Source(
        name="local-pull",
        source_type="git",
        url="https://github.com/user/repo.git",
    )
    dest = tmp_path / "local-pull"
    dest.mkdir()
    subprocess.run(["git", "init"], cwd=dest, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test"],
        cwd=dest,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=dest,
        capture_output=True,
        check=True,
    )
    (dest / "file.md").write_text("# test")
    subprocess.run(["git", "add", "."], cwd=dest, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=dest, capture_output=True, check=True
    )
    # No remote → pull fails → returns False
    result = _pull(src, dest, verbose=False, git_timeout=300)
    assert result is False
