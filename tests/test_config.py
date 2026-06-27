"""Tests for knowledge.config — path resolution, config loader, YAML fallback logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from knowledge.config import (
    Config,
    ensure_data_dir,
    load_config,
    resolve_data_dir,
    resolve_sources_yaml,
)


def test_config_defaults() -> None:
    """Config() with no args uses all defaults."""
    c = Config()
    assert c.fetch.git_timeout == 300
    assert c.search.default_top_k == 10
    assert ".md" in c.index.doc_extensions


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
    assert result.fetch.git_timeout == 300


def test_load_config_empty_file(tmp_path: Path) -> None:
    """Empty config file returns defaults."""
    (tmp_path / "config.yaml").write_text("")
    result = load_config(tmp_path)
    assert result.fetch.git_timeout == 300


def test_load_config_not_a_dict(tmp_path: Path) -> None:
    """config.yaml with non-dict root returns defaults."""
    (tmp_path / "config.yaml").write_text("just a string")
    result = load_config(tmp_path)
    assert result.fetch.git_timeout == 300


def test_load_config_git_timeout(tmp_path: Path) -> None:
    """fetch.git_timeout is parsed from config."""
    (tmp_path / "config.yaml").write_text("fetch:\n  git_timeout: 600\n")
    result = load_config(tmp_path)
    assert result.fetch.git_timeout == 600


def test_load_config_default_top_k(tmp_path: Path) -> None:
    """search.default_top_k is parsed from config."""
    (tmp_path / "config.yaml").write_text("search:\n  default_top_k: 20\n")
    result = load_config(tmp_path)
    assert result.search.default_top_k == 20


def test_load_config_doc_extensions(tmp_path: Path) -> None:
    """index.doc_extensions is parsed from config."""
    (tmp_path / "config.yaml").write_text(
        "index:\n  doc_extensions:\n    - .md\n    - .txt\n"
    )
    result = load_config(tmp_path)
    assert result.index.doc_extensions == (".md", ".txt")


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


@pytest.mark.parametrize(
    "d,key,typ,default,expected",
    [
        ({"x": "a"}, "x", str, "b", "a"),  # present, correct type
        ({"x": 1}, "x", str, "b", "b"),  # present, wrong type
        ({}, "x", str, "b", "b"),  # missing key
        ({"x": None}, "x", str, "b", "b"),  # None value (not str)
        (
            {"x": True},
            "x",
            int,
            0,
            True,
        ),  # bool is int subclass — passes isinstance check
        ({"x": 5}, "x", int, 0, 5),  # present, correct int
        ({"x": "hello"}, "x", str, "", "hello"),  # present, correct str
    ],
)
def test_parse_field(d, key, typ, default, expected) -> None:
    from knowledge.config import _parse_field

    assert _parse_field(d, key, typ, default) == expected
