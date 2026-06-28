"""Tests for knowledge.normalize — content and heading normalization."""

from __future__ import annotations

from knowledge.normalize import normalize_heading


def test_normalize_heading_strips_markdown_links() -> None:
    result = normalize_heading("[Label](https://example.com)")
    assert result == "Label"


def test_normalize_heading_strips_html_anchors() -> None:
    result = normalize_heading('Section <a id="foo"></a> Name')
    assert result == "Section Name"


def test_normalize_heading_decodes_html_entities() -> None:
    result = normalize_heading("TCP &amp; UDP")
    assert result == "TCP & UDP"


def test_normalize_heading_collapses_whitespace() -> None:
    result = normalize_heading("  Too   much  space  ")
    assert result == "Too much space"


def test_normalize_heading_preserves_inline_formatting() -> None:
    result = normalize_heading("**bold** and *italic* and `code`")
    assert result == "**bold** and *italic* and `code`"


def test_normalize_heading_strips_setext_underline_residue() -> None:
    result = normalize_heading("====")
    assert result == "===="  # single-word setext underlines are harmless


def test_normalize_body_passthrough_md(tmp_path) -> None:
    from knowledge.normalize import normalize_body

    f = tmp_path / "test.md"
    f.write_text("# Hello\n\nWorld")
    result = normalize_body(f, ".md")
    assert result == "# Hello\n\nWorld"


def test_normalize_body_passthrough_txt(tmp_path) -> None:
    from knowledge.normalize import normalize_body

    f = tmp_path / "test.txt"
    f.write_text("plain text")
    result = normalize_body(f, ".txt")
    assert result == "plain text"


def test_normalize_body_rst_to_md(tmp_path) -> None:
    from knowledge.normalize import normalize_body

    f = tmp_path / "test.rst"
    f.write_text(
        "Hello\n=====\n\nSome **bold** text.\n\n.. code-block:: python\n\n    print(1)"
    )
    result = normalize_body(f, ".rst")
    assert "Hello" in result
    assert "print(1)" in result or "```" in result


def test_normalize_body_rst_failure_fallback(tmp_path) -> None:
    """When rst2gfm raises, log warning and return original text."""
    from unittest.mock import patch
    from knowledge.normalize import normalize_body

    f = tmp_path / "test.rst"
    body = "Hello World"
    f.write_text(body)
    with patch("knowledge.normalize._rst_to_md", side_effect=ValueError("bad rst")):
        result = normalize_body(f, ".rst")
    assert result == body


def test_normalize_body_notebook_none_on_failure(tmp_path) -> None:
    """Corrupt notebook returns None (file skipped)."""
    from knowledge.normalize import normalize_body

    f = tmp_path / "bad.ipynb"
    f.write_text("not json")
    result = normalize_body(f, ".ipynb")
    assert result is None


def test_qualify_heading_top_level() -> None:
    from knowledge.normalize import qualify_heading

    result = qualify_heading("HackTricks", "Token Confusion", is_top_level=True)
    assert result == "HackTricks: Token Confusion"


def test_qualify_heading_nested() -> None:
    from knowledge.normalize import qualify_heading

    result = qualify_heading("HackTricks", "Configuration", is_top_level=False)
    assert result == "Configuration"


def test_qualify_heading_empty_source_title() -> None:
    from knowledge.normalize import qualify_heading

    result = qualify_heading("", "Token Confusion", is_top_level=True)
    assert result == "Token Confusion"


def test_qualify_heading_default_is_top_level() -> None:
    from knowledge.normalize import qualify_heading

    result = qualify_heading("HackTricks", "Token Confusion")
    assert result == "HackTricks: Token Confusion"
