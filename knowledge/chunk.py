"""Heading-aware document chunker: ATX + setext, code-block aware, notebook conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nbformat

HEADING_AWARE_EXTS: set[str] = {".md", ".markdown", ".mdx", ".rst", ".ipynb"}


@dataclass(frozen=True, slots=True)
class Section:
    """A single document section with heading path and body."""

    source: str
    title: str
    category: str
    path: str
    heading_path: str
    body: str


def chunk_file(
    filepath: Path,
    source: str,
    category: str,
    rel_path: str | None = None,
    source_title: str | None = None,
) -> list[Section]:
    """Read a file, normalize body, and split into heading-bounded sections.

    Non-markdown formats (RST, notebooks) are converted to markdown
    via ``normalize_body`` before chunking.

    Args:
        filepath: Path to the file to chunk.
        source: Source name for metadata.
        category: Source category for metadata.
        rel_path: Relative path for metadata. Defaults to str(filepath).
        source_title: Human-readable source title for heading qualification.

    Returns:
        List of Section dataclass instances.
    """
    from knowledge.normalize import normalize_body

    ext = filepath.suffix.lower()
    normalized = normalize_body(filepath, ext)
    if normalized is None:
        return []
    detect_headings = ext in HEADING_AWARE_EXTS
    return chunk_text(
        normalized,
        source,
        category,
        rel_path or str(filepath),
        detect_headings=detect_headings,
        source_title=source_title,
    )


def chunk_text(
    text: str,
    source: str,
    category: str,
    rel_path: str,
    detect_headings: bool = True,
    source_title: str | None = None,
) -> list[Section]:
    """Split a text string into heading-bounded Section instances.

    Scans for ATX (``# heading``) and setext (underlined) headings,
    stripping frontmatter and respecting code-block fences.
    Normalizes each heading segment and qualifies the first segment
    with the source title when provided.

    Args:
        text: Document text content.
        source: Source name for metadata.
        category: Source category for metadata.
        rel_path: Relative file path for metadata.
        detect_headings: Enable ATX and setext heading detection.
            When ``False``, the entire document is returned as a single section.
        source_title: Human-readable source title for heading qualification.

    Returns:
        List of Section dataclass instances. Returns a single Section with
        the full text if no headings are found.
    """
    text = re.sub(r"^\ufeff?---+\s*\n.*?\n---+[ \t]*", "", text, flags=re.DOTALL)
    lines = text.splitlines()
    boundaries = _scan_headings(lines) if detect_headings else []

    stem = Path(rel_path).stem
    fallback_title = stem.replace("-", " ").replace("_", " ").replace(".", " ")

    if not boundaries:
        return [
            Section(
                source=source,
                title=fallback_title,
                category=category,
                path=rel_path,
                heading_path=fallback_title,
                body=text.strip(),
            )
        ]

    sections = []
    heading_path_parts: list[str] = []

    first_hd_idx = boundaries[0][0]
    if first_hd_idx > 0:
        preamble = "\n".join(lines[:first_hd_idx]).strip()
        if preamble:
            sections.append(
                Section(
                    source=source,
                    title=fallback_title,
                    category=category,
                    path=rel_path,
                    heading_path=fallback_title,
                    body=preamble,
                )
            )

    for idx, (_, level, text, body_start) in enumerate(boundaries):
        heading_path_parts = heading_path_parts[: level - 1]
        while len(heading_path_parts) < level:
            heading_path_parts.append("")
        heading_path_parts[level - 1] = text

        from knowledge.normalize import normalize_heading, qualify_heading

        cleaned_parts: list[str] = []
        for i, part in enumerate(heading_path_parts):
            if not part:
                cleaned_parts.append(part)
                continue
            norm = normalize_heading(part)
            if i == 0 and source_title:
                norm = qualify_heading(source_title, norm, is_top_level=True)
            cleaned_parts.append(norm)

        heading_path = " > ".join(p for p in cleaned_parts if p)

        if idx + 1 < len(boundaries):
            body_end = boundaries[idx + 1][0]
        else:
            body_end = len(lines)

        body_lines = lines[body_start:body_end]
        body = "\n".join(body_lines).strip()
        # Empty body is valid — don't fall back to entire document
        sections.append(
            Section(
                source=source,
                title=text,
                category=category,
                path=rel_path,
                heading_path=heading_path,
                body=body,
            )
        )

    return sections


def _scan_headings(
    lines: list[str],
) -> list[tuple[int, int, str, int]]:
    headings: list[tuple[int, int, str, int]] = []
    in_code_block = False
    prev_text: tuple[int, str] | None = None

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith(("```", "~~~")):
            in_code_block = not in_code_block
            prev_text = None
            i += 1
            continue
        if in_code_block:
            prev_text = None
            i += 1
            continue

        atx = re.match(r"^(#{1,6})\s+(.+)$", line)
        if atx:
            level = len(atx.group(1))
            if level <= 6:
                headings.append((i, level, atx.group(2).strip(), i + 1))
                prev_text = None
                i += 1
                continue

        setext = re.match(r"^ {0,3}(={3,}|-{3,})\s*$", line)
        if setext and prev_text:
            prev_idx, prev_line = prev_text
            level = 1 if setext.group(1)[0] == "=" else 2
            headings.append((prev_idx, level, prev_line.strip(), i + 1))
            prev_text = None
            i += 1
            continue

        if line.strip():
            prev_text = (i, line)
        else:
            prev_text = None
        i += 1

    headings.sort(key=lambda x: x[0])
    return headings


def _convert_notebook(filepath: Path) -> str:
    with open(filepath, encoding="utf-8", errors="replace") as f:
        nb = nbformat.read(f, as_version=4)

    cells = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cells.append(cell.source)
        elif cell.cell_type == "code":
            cells.append(f"```\n{cell.source}\n```")
        elif cell.cell_type == "raw":
            cells.append(f"<!-- raw cell -->\n{cell.source}")
    return "\n\n".join(cells)
