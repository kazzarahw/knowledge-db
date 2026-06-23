"""Tests for knowledge.chunk — heading-aware section parser."""

from __future__ import annotations

from pathlib import Path

from knowledge.chunk import Section, chunk_file, chunk_text

BASIC_MD = """# Intro
Welcome to the docs.

## Installation
Run `pip install foo`.

# Usage
## CLI
Use `foo --help`.

### Subcommands
Run `foo bar`.
"""


class TestChunkText:
    def test_atx_headings(self):
        sections = chunk_text(BASIC_MD, "test", "wikis", "guide.md")
        assert len(sections) == 4
        assert sections[0].title == "Intro"
        assert sections[0].heading_path == "Intro"
        assert "Welcome" in sections[0].body
        assert sections[1].heading_path == "Intro > Installation"
        assert sections[3].heading_path == "Usage > CLI > Subcommands"

    def test_preamble_before_first_heading(self):
        md = "Some preamble text.\n\n# Heading\nBody text."
        sections = chunk_text(md, "test", "wikis", "preamble.md")
        assert len(sections) == 2
        assert sections[0].heading_path == "preamble"
        assert sections[1].title == "Heading"

    def test_no_headings(self):
        md = "Just a paragraph.\n\nAnother paragraph."
        sections = chunk_text(md, "test", "wikis", "plain.txt")
        assert len(sections) == 1
        assert "Just a paragraph" in sections[0].body
        assert "Another paragraph" in sections[0].body

    def test_frontmatter_stripped(self):
        md = "---\ntitle: Test\n---\n# Real Heading\nBody."
        sections = chunk_text(md, "test", "wikis", "fm.md")
        assert len(sections) == 1
        assert sections[0].title == "Real Heading"
        assert "title: Test" not in sections[0].body

    def test_code_block_hash_not_heading(self):
        md = "# Heading\n\n```python\n# This is a comment, not a heading\npass\n```\n\n## Real Subheading\nBody."
        sections = chunk_text(md, "test", "wikis", "code.md")
        assert len(sections) == 2
        assert sections[0].title == "Heading"
        assert sections[1].title == "Real Subheading"

    def test_seven_plus_hashes_not_heading(self):
        md = "# H1\n\n####### Not a heading\n\n## H2\nBody."
        sections = chunk_text(md, "test", "wikis", "seven.md")
        assert len(sections) == 2
        assert sections[0].title == "H1"
        assert sections[1].title == "H2"

    def test_setext_heading(self):
        md = "H1 Heading\n===\nBody paragraph.\n\nH2 Heading\n---\nMore body."
        sections = chunk_text(md, "test", "wikis", "setext.md")
        assert len(sections) == 2
        assert sections[0].title == "H1 Heading"
        assert sections[0].heading_path == "H1 Heading"
        assert sections[1].title == "H2 Heading"
        assert sections[1].heading_path == "H1 Heading > H2 Heading"

    def test_setext_heading_leading_whitespace(self):
        md = "Top Level\n  ===\nBody.\n\nSecond Level\n  ---\nMore body."
        sections = chunk_text(md, "test", "wikis", "setext.md")
        assert len(sections) == 2
        assert sections[0].title == "Top Level"
        assert sections[0].heading_path == "Top Level"
        assert sections[1].title == "Second Level"
        assert sections[1].heading_path == "Top Level > Second Level"

    def test_crlf_line_endings(self):
        md = "# Heading\r\n\r\nBody paragraph.\r\n\r\n## Subheading\r\nMore content."
        sections = chunk_text(md, "test", "wikis", "crlf.md")
        assert len(sections) == 2
        assert sections[0].title == "Heading"
        assert sections[1].title == "Subheading"
        assert "Body paragraph." in sections[0].body

    def test_frontmatter_preserves_body_content(self):
        md = "---\ntitle: Test\n---\n\nBody text after frontmatter.\n\n# Real Heading\nBody."
        sections = chunk_text(md, "test", "wikis", "fm.md")
        assert len(sections) == 2
        assert sections[0].heading_path == "fm"
        assert "Body text after frontmatter" in sections[0].body
        assert sections[1].title == "Real Heading"


class TestChunkFile:
    def test_file_read(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Hello\nWorld.")
        sections = chunk_file(f, "test", "wikis", "test.md")
        assert len(sections) == 1
        assert sections[0].title == "Hello"
        assert "World" in sections[0].body

    def test_notebook_conversion(self, tmp_path):
        import nbformat as nbf

        nb = nbf.v4.new_notebook()
        nb.cells = [
            nbf.v4.new_markdown_cell("# Notebook Title"),
            nbf.v4.new_code_cell("x = 1"),
            nbf.v4.new_markdown_cell("## Section 2\nContent."),
        ]
        f = tmp_path / "test.ipynb"
        nbf.write(nb, f)
        sections = chunk_file(f, "test", "notebooks", "test.ipynb")
        assert len(sections) == 2
        assert sections[0].title == "Notebook Title"
        assert sections[1].title == "Section 2"

    def test_notebook_raw_cell(self, tmp_path):
        import nbformat as nbf

        nb = nbf.v4.new_notebook()
        nb.cells = [
            nbf.v4.new_raw_cell("raw content here"),
            nbf.v4.new_markdown_cell("# Heading\nBody text."),
        ]
        f = tmp_path / "test.ipynb"
        nbf.write(nb, f)
        sections = chunk_file(f, "test", "notebooks", "test.ipynb")
        assert len(sections) == 2
        assert sections[0].heading_path == "test"
        assert "raw content here" in sections[0].body
        assert sections[1].title == "Heading"
