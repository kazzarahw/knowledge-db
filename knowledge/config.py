"""Path resolution, constants, config.yaml loader, and version."""

from __future__ import annotations

import os
from pathlib import Path

VERSION = "0.1.0"
APP_NAME = "knowledge-db"
DATA_DIR_ENV_VAR = "KNOWLEDGE_DB_DIR"
DEFAULT_MODEL = "LiquidAI/LFM2.5-Embedding-350M"


def load_config(config_dir: Path) -> dict[str, object]:
    """Load config.yaml from a config directory.

    Returns a dict with optional keys: model (str), device (str | None).
    Missing or invalid files fall back to defaults silently.
    """
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        return {"model": DEFAULT_MODEL, "device": None}
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {"model": DEFAULT_MODEL, "device": None}
        embed = raw.get("embed", {}) or {}
        if not isinstance(embed, dict):
            return {"model": DEFAULT_MODEL, "device": None}
        return {
            "model": embed.get("model", DEFAULT_MODEL),
            "device": embed.get("device"),
        }
    except Exception:
        return {"model": DEFAULT_MODEL, "device": None}


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
