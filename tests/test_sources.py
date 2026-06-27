"""Tests for knowledge.sources."""

from __future__ import annotations

import pytest
import yaml

from knowledge.sources import ConfigError, Source, load_sources


class TestSource:
    def test_git_source_valid(self):
        s = Source(
            name="test", source_type="git", url="https://github.com/user/repo.git"
        )
        assert s.name == "test"
        assert s.source_type == "git"

    def test_git_source_no_url_raises(self):
        with pytest.raises(ConfigError, match="must have a url"):
            Source(name="bad", source_type="git")

    def test_git_source_bad_url_raises(self):
        with pytest.raises(ConfigError, match="Invalid git URL"):
            Source(name="bad", source_type="git", url="not-a-url")

    def test_local_source_valid(self):
        s = Source(name="local1", source_type="local", path="/some/dir")
        assert s.source_type == "local"

    def test_local_source_no_path_raises(self):
        with pytest.raises(ConfigError, match="must have a path"):
            Source(name="bad", source_type="local")

    def test_notebooks_source_valid(self):
        s = Source(
            name="nb1", source_type="notebooks", url="https://github.com/user/repo.git"
        )
        assert s.source_type == "notebooks"

    def test_invalid_type_raises(self):
        with pytest.raises(ConfigError, match="Invalid source type"):
            Source(name="bad", source_type="ftp")

    def test_default_config(self):
        s = Source(
            name="test", source_type="git", url="https://github.com/user/repo.git"
        )
        assert s.index_ext == (".md", ".mdx", ".rst", ".txt", ".py")
        assert s.sparse == ()

    def test_custom_index_ext(self):
        s = Source(
            name="test",
            source_type="git",
            url="https://github.com/user/repo.git",
            index_ext=(".md",),
        )
        assert s.index_ext == (".md",)

    def test_frozen_dataclass(self):
        s = Source(
            name="test", source_type="git", url="https://github.com/user/repo.git"
        )
        with pytest.raises(AttributeError):
            s.name = "changed"  # type: ignore[misc]  # intentional — frozen dataclass raises AttributeError, not a type error

    def test_default_category_and_docs_dir(self):
        s = Source(
            name="test", source_type="git", url="https://github.com/user/repo.git"
        )
        assert s.category == ""
        assert s.docs_dir is None

    def test_custom_category_and_docs_dir(self):
        s = Source(
            name="test",
            source_type="git",
            url="https://github.com/user/repo.git",
            category="docs",
            docs_dir="docs/",
        )
        assert s.category == "docs"
        assert s.docs_dir == "docs/"

    def test_title_default(self):
        s = Source(
            name="test", source_type="git", url="https://github.com/user/repo.git"
        )
        assert s.title == ""

    def test_custom_title(self):
        s = Source(
            name="test",
            source_type="git",
            url="https://github.com/user/repo.git",
            title="My Docs",
        )
        assert s.title == "My Docs"


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
        assert sources[0].source_type == "git"
        assert sources[1].name == "local_docs"
        assert sources[1].source_type == "local"

    def test_duplicate_names_raises(self, tmp_path):
        path = tmp_path / "sources.yaml"
        data = {
            "sources": [
                {"name": "dup", "url": "https://github.com/a.git"},
                {"name": "dup", "url": "https://github.com/b.git"},
            ]
        }
        path.write_text(yaml.dump(data))
        with pytest.raises(ConfigError, match="Duplicate source name"):
            load_sources(path)

    def test_missing_sources_key_raises(self, tmp_path):
        path = tmp_path / "sources.yaml"
        path.write_text(yaml.dump({"other": []}))
        with pytest.raises(ConfigError, match="must contain a 'sources' key"):
            load_sources(path)

    def test_file_not_found(self, tmp_path):
        path = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigError, match="Sources file not found"):
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

    def test_category_and_docs_dir_from_yaml(self, tmp_path):
        path = tmp_path / "sources.yaml"
        data = {
            "sources": [
                {
                    "name": "categorized",
                    "url": "https://github.com/user/repo.git",
                    "category": "documentation",
                    "docs_dir": "docs/",
                },
            ]
        }
        path.write_text(yaml.dump(data))
        sources = load_sources(path)
        assert sources[0].category == "documentation"
        assert sources[0].docs_dir == "docs/"

    def test_title_from_yaml(self, tmp_path):
        path = tmp_path / "sources.yaml"
        data = {
            "sources": [
                {
                    "name": "titled",
                    "url": "https://github.com/user/repo.git",
                    "title": "My Documentation",
                },
            ]
        }
        path.write_text(yaml.dump(data))
        sources = load_sources(path)
        assert sources[0].title == "My Documentation"

    def test_title_default_from_yaml(self, tmp_path):
        path = tmp_path / "sources.yaml"
        data = {
            "sources": [
                {"name": "untitled", "url": "https://github.com/user/repo.git"},
            ]
        }
        path.write_text(yaml.dump(data))
        sources = load_sources(path)
        assert sources[0].title == ""
