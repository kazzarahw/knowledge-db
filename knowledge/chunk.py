from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nbformat


@dataclass(frozen=True, slots=True)
class Section:
    source: str
    title: str
    category: str
    path: str
    heading_path: str
    body: str


def chunk_file(
    filepath: Path, source: str, category: str, rel_path: str | None = None
) -> list[Section]:
    ext = filepath.suffix.lower()
    if ext == ".ipynb":
        text = _convert_notebook(filepath)
    else:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    return chunk_text(text, source, category, rel_path or str(filepath))


def chunk_text(text: str, source: str, category: str, rel_path: str) -> list[Section]:
    text = re.sub(r"^\ufeff?---+\s*\n.*?\n---+\s*", "", text, flags=re.DOTALL)
    lines = text.split("\n")
    boundaries = _scan_headings(lines)

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

    for idx, (line_num, level, text, body_start) in enumerate(boundaries):
        heading_path_parts = heading_path_parts[: level - 1]
        while len(heading_path_parts) < level:
            heading_path_parts.append("")
        heading_path_parts[level - 1] = text

        heading_path = " > ".join(p for p in heading_path_parts if p)

        if idx + 1 < len(boundaries):
            body_end = boundaries[idx + 1][0]
        else:
            body_end = len(lines)

        body_lines = lines[body_start:body_end]
        body = "\n".join(body_lines).strip()

        if body:
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
            level = 1 if line[0] == "=" else 2
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
    with open(filepath, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    cells = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cells.append(cell.source)
        elif cell.cell_type == "code":
            cells.append(f"```\n{cell.source}\n```")
    return "\n\n".join(cells)
