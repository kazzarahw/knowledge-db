"""Path resolution, constants, config.yaml loader, and version."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "0.1.0"
APP_NAME = "knowledge-db"
DATA_DIR_ENV_VAR = "KNOWLEDGE_DB_DIR"
DEFAULT_MODEL = "LiquidAI/LFM2.5-Embedding-350M"


@dataclass(frozen=True, slots=True)
class EmbedConfig:
    model: str = DEFAULT_MODEL
    device: str | None = None
    batch_size: int = 32
    trust_remote_code: bool = True


@dataclass(frozen=True, slots=True)
class FetchConfig:
    git_timeout: int = 300


@dataclass(frozen=True, slots=True)
class IndexConfig:
    doc_extensions: tuple[str, ...] = field(
        default_factory=lambda: (
            ".md",
            ".markdown",
            ".mdx",
            ".rst",
            ".txt",
            ".yml",
            ".yaml",
            ".ipynb",
        )
    )


@dataclass(frozen=True, slots=True)
class SearchConfig:
    default_top_k: int = 10


@dataclass(frozen=True, slots=True)
class Config:
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


def _parse_embed(raw: object) -> EmbedConfig:
    """Parse embed section from raw YAML dict into EmbedConfig."""
    if not isinstance(raw, dict):
        return EmbedConfig()
    model = raw.get("model", DEFAULT_MODEL)
    if not isinstance(model, str):
        model = DEFAULT_MODEL
    device = raw.get("device")
    if not isinstance(device, str):
        device = None
    batch_size = raw.get("batch_size", 32)
    if not isinstance(batch_size, int):
        batch_size = 32
    trust_remote_code = raw.get("trust_remote_code", True)
    if not isinstance(trust_remote_code, bool):
        trust_remote_code = True
    return EmbedConfig(
        model=model,
        device=device,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
    )


def _parse_fetch(raw: object) -> FetchConfig:
    """Parse fetch section from raw YAML dict into FetchConfig."""
    if not isinstance(raw, dict):
        return FetchConfig()
    git_timeout = raw.get("git_timeout", 300)
    if not isinstance(git_timeout, int):
        git_timeout = 300
    return FetchConfig(git_timeout=git_timeout)


def _parse_index(raw: object) -> IndexConfig:
    """Parse index section from raw YAML dict into IndexConfig."""
    if not isinstance(raw, dict):
        return IndexConfig()
    doc_extensions = raw.get("doc_extensions")
    if isinstance(doc_extensions, list) and all(
        isinstance(e, str) for e in doc_extensions
    ):
        return IndexConfig(doc_extensions=tuple(doc_extensions))
    return IndexConfig()


def _parse_search(raw: object) -> SearchConfig:
    """Parse search section from raw YAML dict into SearchConfig."""
    if not isinstance(raw, dict):
        return SearchConfig()
    default_top_k = raw.get("default_top_k", 10)
    if not isinstance(default_top_k, int):
        default_top_k = 10
    return SearchConfig(default_top_k=default_top_k)


def load_config(config_dir: str | Path | None = None) -> Config:
    """Load config.yaml from a config directory.

    Returns a Config dataclass populated from the YAML file.
    Missing or invalid files/sections fall back to defaults silently.

    Args:
        config_dir: Directory containing config.yaml. When None or
            nonexistent, returns fully defaulted Config.

    Returns:
        A Config dataclass with values parsed from config.yaml or defaults.
    """
    if config_dir is None:
        return Config()
    config_path = Path(config_dir) / "config.yaml"
    if not config_path.exists():
        return Config()
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return Config()
        return Config(
            embed=_parse_embed(raw.get("embed")),
            fetch=_parse_fetch(raw.get("fetch")),
            index=_parse_index(raw.get("index")),
            search=_parse_search(raw.get("search")),
        )
    except Exception:
        return Config()


def resolve_data_dir(override: str | None = None) -> Path:
    """Resolve data directory with priority:
    1. --config PATH override
    2. $KNOWLEDGE_DB_DIR env var
    3. $XDG_DATA_HOME/knowledge-db/
    4. ./data/ if pyproject.toml is a sibling
    """
    if override:
        return Path(override)
    if env := os.environ.get(DATA_DIR_ENV_VAR):
        return Path(env)
    xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    candidate = Path(xdg) / APP_NAME
    # Check for project-local mode
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists() and (cwd / "data").exists():
        return cwd / "data"
    return candidate


def ensure_data_dir(data_dir: Path) -> Path:
    """Create data directory and subdirs if they don't exist."""
    (data_dir / "sources").mkdir(parents=True, exist_ok=True)
    return data_dir


def resolve_sources_yaml(config_dir: str | None) -> Path:
    """Resolve sources.yaml path with config-dir fallback.

    Checks *config_dir*/sources.yaml first, then cwd/sources.yaml.
    Prints a warning on stderr when falling back to cwd.

    Args:
        config_dir: Path to config directory. May be None (cwd-only).

    Returns:
        Resolved path to sources.yaml (may not exist — caller validates).
    """
    if config_dir:
        sp = Path(config_dir) / "sources.yaml"
        if sp.exists():
            return sp
        fallback = Path.cwd() / "sources.yaml"
        if fallback.exists():
            print(
                f"Warning: {sp} not found — using {fallback}",
                file=sys.stderr,
            )
            return fallback
        return sp
    return Path("sources.yaml")
