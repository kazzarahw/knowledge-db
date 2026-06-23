"""Source manifest loading and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from yaml import YAMLError


class ConfigError(ValueError):
    """Configuration error — invalid or missing source config."""


_GIT_URL_RE = re.compile(r"^(https?://|git@|ssh://git@)[\w.:/@-]+(?:\.git)?/?$")


def _validate_git_url(url: str) -> bool:
    """Basic git URL validation."""
    return bool(_GIT_URL_RE.match(url))


@dataclass(frozen=True, slots=True)
class Source:
    """A single knowledge source."""

    name: str
    source_type: str
    url: str | None = None
    path: str | None = None
    branch: str | None = None
    sparse: tuple[str, ...] = field(default_factory=tuple)
    index_ext: tuple[str, ...] = field(
        default_factory=lambda: (".md", ".mdx", ".rst", ".txt", ".py")
    )
    title: str = ""
    category: str = ""
    docs_dir: str | None = None

    def __post_init__(self) -> None:
        if self.source_type not in ("git", "local", "notebooks"):
            raise ConfigError(f"Invalid source type: {self.source_type!r}")
        if self.source_type == "git":
            if not self.url:
                raise ConfigError(f"Git source {self.name!r} must have a url")
            if not _validate_git_url(self.url):
                raise ConfigError(f"Invalid git URL for {self.name!r}: {self.url!r}")
        if self.source_type == "local":
            if not self.path:
                raise ConfigError(f"Local source {self.name!r} must have a path")
        if self.source_type == "notebooks":
            if not self.url:
                raise ConfigError(f"Notebooks source {self.name!r} must have a url")


def load_sources(path: Path) -> list[Source]:
    """Load and validate the sources manifest YAML."""
    if not path.exists():
        raise ConfigError(f"Sources file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")

    if not isinstance(data, dict) or "sources" not in data:
        raise ConfigError("Sources file must contain a 'sources' key with a list")

    sources_raw: list[dict[str, Any]] = data["sources"]
    if not isinstance(sources_raw, list):
        raise ConfigError("'sources' must be a list")

    sources: list[Source] = []
    seen_names: set[str] = set()

    for i, entry in enumerate(sources_raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"Source entry {i} is not a mapping")

        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ConfigError(f"Source entry {i} is missing a valid 'name'")
        if name in seen_names:
            raise ConfigError(f"Duplicate source name: {name!r}")
        seen_names.add(name)

        src_type = entry.get("source_type", "git")
        url = entry.get("url")
        path_val = entry.get("path")
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
            source_type=src_type,
            url=url,
            path=path_val,
            branch=branch,
            sparse=sparse,
            index_ext=index_ext,
            title=title,
            category=category,
            docs_dir=docs_dir,
        )
        sources.append(source)

    return sources
