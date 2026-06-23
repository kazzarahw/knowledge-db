"""Tests for knowledge.sources."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from knowledge.sources import Source, load_sources


class TestSource:
    def test_git_source_valid(self):
        s = Source(name="test", type="git", url="https://github.com/user/repo.git")
        assert s.name == "test"
        assert s.type == "git"

    def test_git_source_no_url_raises(self):
        with pytest.raises(ValueError, match="must have a url"):
            Source(name="bad", type="git")

    def test_git_source_bad_url_raises(self):
        with pytest.raises(ValueError, match="Invalid git URL"):
            Source(name="bad", type="git", url="not-a-url")

    def test_local_source_valid(self):
        s = Source(name="local1", type="local", path="/some/dir")
        assert s.type == "local"

    def test_local_source_no_path_raises(self):
        with pytest.raises(ValueError, match="must have a path"):
            Source(name="bad", type="local")

    def test_notebooks_source_valid(self):
        s = Source(name="nb1", type="notebooks", url="https://github.com/user/repo.git")
        assert s.type == "notebooks"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid source type"):
            Source(name="bad", type="ftp")

    def test_default_config(self):
        s = Source(name="test", type="git", url="https://github.com/user/repo.git")
        assert s.index_ext == (".md", ".mdx", ".rst", ".txt", ".py")
        assert s.sparse == ()

    def test_custom_index_ext(self):
        s = Source(
            name="test",
            type="git",
            url="https://github.com/user/repo.git",
            index_ext=(".md",),
        )
        assert s.index_ext == (".md",)

    def test_frozen_dataclass(self):
        s = Source(name="test", type="git", url="https://github.com/user/repo.git")
        with pytest.raises(AttributeError):
            s.name = "changed"  # type: ignore[misc]


class TestLoadSources:
    def test_basic_sources_yaml(self, tmp_path):
        path = tmp_path / "sources.yaml"
        data = {
            "sources": [
                {"name": "grimoire", "url": "https://github.com/user/grimoire.git"},
                {"name": "local_docs", "type": "local", "path": "/home/user/docs"},
            ]
        }
        path.write_text(yaml.dump(data))
        sources = load_sources(path)
        assert len(sources) == 2
        assert sources[0].name == "grimoire"
        assert sources[0].type == "git"
        assert sources[1].name == "local_docs"
        assert sources[1].type == "local"

    def test_duplicate_names_raises(self, tmp_path):
        path = tmp_path / "sources.yaml"
        data = {
            "sources": [
                {"name": "dup", "url": "https://github.com/a.git"},
                {"name": "dup", "url": "https://github.com/b.git"},
            ]
        }
        path.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="Duplicate source name"):
            load_sources(path)

    def test_missing_sources_key_raises(self, tmp_path):
        path = tmp_path / "sources.yaml"
        path.write_text(yaml.dump({"other": []}))
        with pytest.raises(ValueError, match="must contain a 'sources' key"):
            load_sources(path)

    def test_file_not_found(self, tmp_path):
        path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Sources file not found"):
            load_sources(path)

    def test_sparse_and_branch(self, tmp_path):
        path = tmp_path / "sources.yaml"
        data = {
            "sources": [
                {
                    "name": "sparse1",
                    "url": "https://github.com/user/repo.git",
                    "sparse": ["docs/*"],
                    "branch": "main",
                },
            ]
        }
        path.write_text(yaml.dump(data))
        sources = load_sources(path)
        assert sources[0].sparse == ("docs/*",)
        assert sources[0].branch == "main"
