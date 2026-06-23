"""Git clone/pull and local source resolution."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from knowledge.sources import Source


def fetch_sources(
    sources: list[Source],
    data_dir: Path,
    only: str | None = None,
    verbose: bool = False,
) -> list[str]:
    """Clone/pull all (or one) sources. Returns list of changed source names."""
    changed = []
    for src in sources:
        if only and src.name != only:
            continue
        if src.type == "local":
            continue
        if _fetch_git_source(src, data_dir, verbose):
            changed.append(src.name)
    return changed


def _fetch_git_source(source: Source, data_dir: Path, verbose: bool) -> bool:
    """Clone or pull a single git source. Returns True if HEAD changed."""
    dest = data_dir / "sources" / source.name

    if not dest.exists():
        return _clone(source, dest, verbose)
    return _pull(source, dest, verbose)


def _clone(source: Source, dest: Path, verbose: bool) -> bool:
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
        cmd += [repo_url, tmpdir]

        if verbose:
            print(f"  Cloning {source.name} from {repo_url}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error cloning {source.name}: {result.stderr.strip()}")
            return False

        if source.sparse:
            sparse_cmd = ["git", "-C", tmpdir, "sparse-checkout", "set", *source.sparse]
            subprocess.run(sparse_cmd, capture_output=True, text=True)

        lfs_check = subprocess.run(
            ["git", "-C", tmpdir, "lfs", "track"],
            capture_output=True,
            text=True,
        )
        if lfs_check.returncode == 0 and lfs_check.stdout.strip():
            if shutil.which("git-lfs"):
                subprocess.run(
                    ["git", "-C", tmpdir, "lfs", "pull"], capture_output=True
                )
            else:
                print(
                    f"  Warning: {source.name} uses Git LFS but git-lfs is not installed"
                )

        try:
            os.rename(tmpdir, str(dest))
        except OSError:
            shutil.copytree(tmpdir, str(dest), dirs_exist_ok=True)

    return True


def _pull(source: Source, dest: Path, verbose: bool) -> bool:
    """Pull existing repo. Returns True if HEAD changed."""
    if verbose:
        print(f"  Pulling {source.name}")

    before = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if before.returncode != 0:
        print(f"  Error reading HEAD for {source.name}")
        return False

    status = subprocess.run(
        ["git", "-C", str(dest), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if status.stdout.strip() and verbose:
        print(f"  {source.name} has uncommitted changes — reindexing anyway")

    result = subprocess.run(
        ["git", "-C", str(dest), "pull", "--ff-only"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fsck = subprocess.run(
            ["git", "-C", str(dest), "fsck"],
            capture_output=True,
            text=True,
        )
        if fsck.returncode != 0:
            print(f"  {source.name} is corrupt (fsck failed). Re-cloning...")
            shutil.rmtree(str(dest))
            return _clone(source, dest, verbose)
        print(f"  Error pulling {source.name}: {result.stderr.strip()}")
        return False

    after = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    head_before = before.stdout.strip()
    head_after = after.stdout.strip()
    return head_before != head_after


def get_git_head(source_dir: Path) -> str | None:
    """Get current git HEAD for a source directory. Returns None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return None
