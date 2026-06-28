"""Content and heading normalization for knowledge-db sources."""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

import nbformat

logger = logging.getLogger(__name__)

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HTML_ANCHOR_RE = re.compile(r"<a\b[^>]*>.*?</a>", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_heading(segment: str) -> str:
    """Clean a single heading segment for display.

    Strips markdown links, HTML anchors, decodes entities, collapses
    whitespace. Preserves inline formatting (**bold**, *italic*, ``code``).

    Args:
        segment: Raw heading text from a document.

    Returns:
        Cleaned heading text suitable for display.
    """
    s = _MD_LINK_RE.sub(r"\1", segment)
    s = _HTML_ANCHOR_RE.sub("", s)
    s = html.unescape(s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def normalize_body(path: Path, file_ext: str) -> str | None:
    """Convert non-markdown file content to clean markdown.

    Args:
        path: Path to the file. Read internally (required for notebook parsing).
        file_ext: File extension with leading dot (e.g. ``.rst``, ``.ipynb``).

    Returns:
        Normalized markdown text, or ``None`` if the file should be skipped
        (corrupt notebook). On other conversion failures, returns original text.
    """
    try:
        match file_ext:
            case ".ipynb":
                return _notebook_to_md(path)
            case ".rst":
                return _rst_to_md(path)
            case _:
                return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.warning("normalize_body failed for %s, using original", path)
        return path.read_text(encoding="utf-8", errors="replace")


def _notebook_to_md(path: Path) -> str | None:
    """Convert Jupyter notebook to markdown text.

    Returns None if the notebook is corrupt/unparseable (file skipped).
    """
    try:
        nb = nbformat.read(path, as_version=4)
    except Exception:
        logger.warning("Failed to parse notebook %s, skipping", path)
        return None

    cells: list[str] = []
    for cell in nb.cells:
        match cell.cell_type:
            case "markdown":
                cells.append(cell.source)
            case "code":
                cells.append(f"```\n{cell.source}\n```")
            case "raw":
                cells.append(cell.source)
    return "\n\n".join(cells)


def _rst_to_md(path: Path) -> str:
    """Convert reStructuredText to markdown via rst2gfm."""
    from rst2gfm import convert_rst_to_md

    raw = path.read_text(encoding="utf-8", errors="replace")
    return convert_rst_to_md(raw)


def qualify_heading(source_title: str, heading: str, is_top_level: bool = True) -> str:
    """Prepend source title to top-level heading.

    Only applies to the first segment in a heading path (``is_top_level=True``).
    Nested segments pass through unchanged.

    Args:
        source_title: Human-readable source title from ``sources.yaml``.
        heading: Cleaned heading segment text.
        is_top_level: Whether this is the first segment in the heading path.

    Returns:
        ``SourceTitle: Heading`` if top-level, or ``Heading`` unchanged.
    """
    if is_top_level and source_title:
        return f"{source_title}: {heading}"
    return heading
