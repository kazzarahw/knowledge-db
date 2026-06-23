"""Tests for knowledge.config — path resolution, config loader, YAML fallback logic."""

from __future__ import annotations

from pathlib import Path

from knowledge.config import load_config


def test_load_config_null_model_in_yaml(tmp_path: Path) -> None:
    """model: null in YAML must fall back to DEFAULT_MODEL, not produce 'None' string."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("embed:\n  model: null\n")
    result = load_config(tmp_path)
    assert result["model"] == "LiquidAI/LFM2.5-Embedding-350M"
    assert result["model"] is not None
    assert "None" not in result["model"]
