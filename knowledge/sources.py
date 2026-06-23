"""Source manifest loading and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigError(ValueError):
    """Configuration error — invalid or missing source config."""


_GIT_URL_RE = re.compile(r"^(https?://|git@|ssh://git@)[\w.:/-]+(?:\.git)?/?$")


def _validate_git_url(url: str) -> bool:
    """Basic git URL validation."""
    return bool(_GIT_URL_RE.match(url))


@dataclass(frozen=True)
class Source:
    """A single knowledge source."""

    name: str
    type: str
    url: Optional[str] = None
    path: Optional[str] = None
    subdir: Optional[str] = None
    branch: Optional[str] = None
    sparse: tuple[str, ...] = field(default_factory=tuple)
    index_ext: tuple[str, ...] = field(
        default_factory=lambda: (".md", ".mdx", ".rst", ".txt", ".py")
    )
    title: str = ""
    category: str = ""
    docs_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type not in ("git", "local", "notebooks"):
            raise ValueError(f"Invalid source type: {self.type!r}")
        if self.type == "git":
            if not self.url:
                raise ValueError(f"Git source {self.name!r} must have a url")
            if not _validate_git_url(self.url):
                raise ValueError(f"Invalid git URL for {self.name!r}: {self.url!r}")
        if self.type == "local":
            if not self.path:
                raise ValueError(f"Local source {self.name!r} must have a path")
        if self.type == "notebooks":
            if not self.url:
                raise ValueError(f"Notebooks source {self.name!r} must have a url")


def load_sources(path: Path) -> List[Source]:
    """Load and validate the sources manifest YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Sources file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)

    if not isinstance(data, dict) or "sources" not in data:
        raise ValueError("Sources file must contain a 'sources' key with a list")

    sources_raw: List[Dict[str, Any]] = data["sources"]
    if not isinstance(sources_raw, list):
        raise ValueError("'sources' must be a list")

    sources: List[Source] = []
    seen_names: set[str] = set()

    for i, entry in enumerate(sources_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Source entry {i} is not a mapping")

        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"Source entry {i} is missing a valid 'name'")
        if name in seen_names:
            raise ValueError(f"Duplicate source name: {name!r}")
        seen_names.add(name)

        src_type = entry.get("type", "git")
        url = entry.get("url")
        path_val = entry.get("path")
        subdir = entry.get("subdir")
        branch = entry.get("branch")
        sparse = tuple(entry.get("sparse", []))
        index_ext = tuple(
            entry.get("index_ext", (".md", ".mdx", ".rst", ".txt", ".py"))
        )
        category = entry.get("category", "")
        docs_dir = entry.get("docs_dir")
        title = entry.get("title", "")

        source = Source(
            name=name,
            type=src_type,
            url=url,
            path=path_val,
            subdir=subdir,
            branch=branch,
            sparse=sparse,
            index_ext=index_ext,
            title=title,
            category=category,
            docs_dir=docs_dir,
        )
        sources.append(source)

    return sources
