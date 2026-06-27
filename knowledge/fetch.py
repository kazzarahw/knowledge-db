"""Git clone/pull and local source resolution."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from knowledge.config import load_config
from knowledge.sources import Source


@dataclass(frozen=True, slots=True)
class _GitResult:
    """Result of a git subprocess execution."""

    stdout: str
    stderr: str
    returncode: int


def _git_run(cmd: list[str], timeout: int) -> _GitResult | None:
    """Run a subprocess command. Returns None on timeout.

    Note: callers use ``git -C <path>`` in the command list so no
    separate cwd parameter is needed.
    """
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return _GitResult(r.stdout, r.stderr, r.returncode)
    except subprocess.TimeoutExpired:
        return None


def _handle_lfs(repo_dir: str | Path, source_name: str, git_timeout: int) -> None:
    """Pull LFS objects if the repo uses Git LFS. Prints warnings on failure.

    LFS failures are non-fatal -- placeholder text may appear in indexed
    content but does not prevent the source from being usable.
    """
    lfs_check = _git_run(
        ["git", "-C", str(repo_dir), "lfs", "track"],
        git_timeout,
    )
    if lfs_check is not None and lfs_check.returncode == 0 and lfs_check.stdout.strip():
        if shutil.which("git-lfs"):
            lfs_result = _git_run(
                ["git", "-C", str(repo_dir), "lfs", "pull"],
                git_timeout,
            )
            if lfs_result is None:
                print(
                    f"  Warning: LFS pull timed out for {source_name}",
                    file=sys.stderr,
                )
            elif lfs_result.returncode != 0:
                print(
                    f"  Warning: LFS pull failed for {source_name}",
                    file=sys.stderr,
                )
        else:
            print(
                f"  Warning: {source_name} uses Git LFS but git-lfs is not installed",
                file=sys.stderr,
            )


def fetch_sources(
    sources: list[Source],
    data_dir: Path,
    only: str | None = None,
    verbose: bool = False,
    config_dir: str | None = None,
) -> list[str]:
    """Clone/pull all (or one) configured git sources.

    Args:
        sources: List of configured sources.
        data_dir: Root data directory containing sources/ subdir.
        only: If set, only process the source with this name.
        verbose: Print per-source progress to stdout.
        config_dir: Config directory for git timeout settings.

    Returns:
        List of source names whose HEAD changed during fetch.
    """
    cfg = load_config(config_dir)
    git_timeout = cfg.fetch.git_timeout

    changed = []
    for src in sources:
        if only and src.name != only:
            continue
        if src.source_type == "local":
            continue
        if _fetch_git_source(src, data_dir, verbose, git_timeout):
            changed.append(src.name)
    return changed


def _fetch_git_source(
    source: Source, data_dir: Path, verbose: bool, git_timeout: int
) -> bool:
    """Clone or pull a single git source. Returns True if HEAD changed."""
    dest = data_dir / "sources" / source.name

    if not dest.exists():
        return _clone_source(source, dest, verbose, git_timeout)
    return _pull_source(source, dest, verbose, git_timeout)


def _clone_source(source: Source, dest: Path, verbose: bool, git_timeout: int) -> bool:
    """Atomically clone a git repo into a temp dir, then rename."""
    repo_url = source.url
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=parent, prefix=f".tmp.{source.name}."
    ) as tmpdir:
        cmd = ["git", "clone", "--depth", "1"]
        if source.sparse:
            cmd += ["--filter=blob:none", "--sparse"]
        if source.branch:
            cmd += ["--branch", source.branch]
        cmd += [repo_url, tmpdir]

        if verbose:
            print(f"  Cloning {source.name} from {repo_url}")

        result = _git_run(cmd, git_timeout)
        if result is None:
            print(
                f"  Error: {source.name} clone timed out after {git_timeout}s",
                file=sys.stderr,
            )
            return False
        if result.returncode != 0:
            print(
                f"  Error cloning {source.name}: {result.stderr.strip()}",
                file=sys.stderr,
            )
            return False

        if source.sparse:
            sparse_result = _git_run(
                ["git", "-C", tmpdir, "sparse-checkout", "set", *source.sparse],
                git_timeout,
            )
            if sparse_result is None:
                print(
                    f"  Error: {source.name} sparse-checkout timed out after {git_timeout}s",
                    file=sys.stderr,
                )
                return False
            if sparse_result.returncode != 0:
                print(
                    f"  Error setting sparse-checkout for {source.name}: {sparse_result.stderr.strip()}",
                    file=sys.stderr,
                )
                return False

        _handle_lfs(tmpdir, source.name, git_timeout)

        shutil.move(tmpdir, str(dest))

    return True


def _pull_source(source: Source, dest: Path, verbose: bool, git_timeout: int) -> bool:
    """Pull existing repo. Returns True if HEAD changed."""
    if verbose:
        print(f"  Pulling {source.name}")

    if source.branch:
        checkout_result = _git_run(
            ["git", "-C", str(dest), "checkout", source.branch],
            git_timeout,
        )
        if checkout_result is None:
            print(
                f"  Error: {source.name} checkout timed out after {git_timeout}s",
                file=sys.stderr,
            )
            return False
        if checkout_result.returncode != 0:
            print(
                f"  Error checking out branch {source.branch} for {source.name}: {checkout_result.stderr.strip()}",
                file=sys.stderr,
            )
            return False

    before = get_git_head(dest, git_timeout)
    if before is None:
        print(f"  Error reading HEAD for {source.name}", file=sys.stderr)
        return False

    status = _git_run(
        ["git", "-C", str(dest), "status", "--porcelain"],
        git_timeout,
    )
    if status is None:
        print(
            f"  Error: {source.name} status timed out after {git_timeout}s",
            file=sys.stderr,
        )
        return False
    if status.stdout.strip() and verbose:
        print(f"  {source.name} has uncommitted changes — reindexing anyway")

    result = _git_run(["git", "-C", str(dest), "pull", "--ff-only"], git_timeout)
    if result is None:
        print(
            f"  Error: {source.name} pull timed out after {git_timeout}s",
            file=sys.stderr,
        )
        return False
    if result.returncode != 0:
        fsck = _git_run(["git", "-C", str(dest), "fsck"], git_timeout)
        if fsck is None:
            print(
                f"  Error: {source.name} fsck timed out after {git_timeout}s",
                file=sys.stderr,
            )
            return False
        if fsck.returncode != 0:
            print(
                f"  {source.name} is corrupt (fsck failed). Re-cloning...",
                file=sys.stderr,
            )
            shutil.rmtree(str(dest))
            return _clone_source(source, dest, verbose, git_timeout)
        print(
            f"  Error pulling {source.name}: {result.stderr.strip()}", file=sys.stderr
        )
        return False

    after = get_git_head(dest, git_timeout)
    if after is None:
        return False

    _handle_lfs(dest, source.name, git_timeout)

    return before != after


def get_git_head(source_dir: Path, git_timeout: int = 300) -> str | None:
    """Get current git HEAD for a source directory. Returns None if not a git repo."""
    try:
        r = _git_run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            git_timeout,
        )
        if r is not None and r.returncode == 0:
            return r.stdout.strip()
    except FileNotFoundError:
        pass
    return None
