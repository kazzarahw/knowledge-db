"""Tests for knowledge.config — path resolution, config loader, YAML fallback logic."""

from __future__ import annotations

from pathlib import Path

from knowledge.config import (
    ensure_data_dir,
    load_config,
    resolve_data_dir,
    resolve_sources_yaml,
)


def test_load_config_null_model_in_yaml(tmp_path: Path) -> None:
    """model: null in YAML must fall back to DEFAULT_MODEL, not produce 'None' string."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("embed:\n  model: null\n")
    result = load_config(tmp_path)
    assert result["model"] == "LiquidAI/LFM2.5-Embedding-350M"
    assert result["model"] is not None
    assert "None" not in result["model"]


def test_resolve_sources_yaml_docstring_has_sections() -> None:
    """Must have Args: and Returns: sections per code-documentation.md."""
    from knowledge.config import resolve_sources_yaml

    doc = resolve_sources_yaml.__doc__
    assert doc is not None
    assert "Args:" in doc
    assert "Returns:" in doc


# ── load_config tests ──────────────────────────────────────────────────────────


def test_load_config_missing_file(tmp_path: Path) -> None:
    """Missing config file returns defaults."""
    result = load_config(tmp_path / "nonexistent")
    assert result["model"] == "LiquidAI/LFM2.5-Embedding-350M"
    assert result["device"] is None


def test_load_config_empty_file(tmp_path: Path) -> None:
    """Empty config file returns defaults."""
    (tmp_path / "config.yaml").write_text("")
    result = load_config(tmp_path)
    assert result["model"] == "LiquidAI/LFM2.5-Embedding-350M"
    assert result["device"] is None


def test_load_config_not_a_dict(tmp_path: Path) -> None:
    """config.yaml with non-dict root returns defaults."""
    (tmp_path / "config.yaml").write_text("just a string")
    result = load_config(tmp_path)
    assert result["model"] == "LiquidAI/LFM2.5-Embedding-350M"


def test_load_config_embed_not_a_dict(tmp_path: Path) -> None:
    """embed: with non-dict value returns defaults."""
    (tmp_path / "config.yaml").write_text("embed: just a string")
    result = load_config(tmp_path)
    assert result["model"] == "LiquidAI/LFM2.5-Embedding-350M"


def test_load_config_custom_model_and_device(tmp_path: Path) -> None:
    """Valid config values are returned correctly."""
    (tmp_path / "config.yaml").write_text(
        "embed:\n  model: custom/model\n  device: cpu\n"
    )
    result = load_config(tmp_path)
    assert result["model"] == "custom/model"
    assert result["device"] == "cpu"


def test_load_config_device_null(tmp_path: Path) -> None:
    """device: null should return device as None, not 'None'."""
    (tmp_path / "config.yaml").write_text("embed:\n  model: my/model\n  device:\n")
    result = load_config(tmp_path)
    assert result["device"] is None


# ── resolve_data_dir tests ─────────────────────────────────────────────────────


def test_resolve_data_dir_override() -> None:
    """Override parameter takes highest priority."""
    result = resolve_data_dir("/tmp/custom-data")
    assert result == Path("/tmp/custom-data")


def test_resolve_data_dir_env_var(monkeypatch) -> None:
    """$KNOWLEDGE_DB_DIR takes second priority."""
    monkeypatch.setenv("KNOWLEDGE_DB_DIR", "/env/data")
    result = resolve_data_dir(None)
    assert result == Path("/env/data")


# ── resolve_sources_yaml tests ─────────────────────────────────────────────────


def test_resolve_sources_yaml_in_config_dir(tmp_path: Path) -> None:
    """Return path in config_dir if sources.yaml exists there."""
    sp = tmp_path / "sources.yaml"
    sp.write_text("sources: []")
    result = resolve_sources_yaml(str(tmp_path))
    assert result == sp


def test_resolve_sources_yaml_fallback_to_cwd(tmp_path: Path, monkeypatch) -> None:
    """Fall back to cwd/sources.yaml when config_dir has none, with warning."""
    monkeypatch.chdir(tmp_path)
    cwd_sp = tmp_path / "sources.yaml"
    cwd_sp.write_text("sources: []")
    result = resolve_sources_yaml(str(Path("/nonexistent")))
    assert result == cwd_sp


# ── ensure_data_dir tests ──────────────────────────────────────────────────────


def test_ensure_data_dir_creates_subdirs(tmp_path: Path) -> None:
    """ensure_data_dir creates sources/ subdir."""
    result = ensure_data_dir(tmp_path)
    assert (tmp_path / "sources").is_dir()
    assert result == tmp_path


def test_ensure_data_dir_idempotent(tmp_path: Path) -> None:
    """Calling twice does not error."""
    ensure_data_dir(tmp_path)
    ensure_data_dir(tmp_path)  # second call should not raise
