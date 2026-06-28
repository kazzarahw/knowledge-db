"""CLI entry point — argparse definitions and main()."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from knowledge.config import (
    VERSION,
    ConfigError,
    resolve_data_dir,
    resolve_sources_yaml,
)


def _validate_hex_prefix(value: str) -> str:
    """Argparse type validator for hash prefix arguments."""
    if not all(c in "0123456789abcdef" for c in value.lower()):
        raise argparse.ArgumentTypeError("hash prefix must be hex characters only")
    if len(value) < 10:
        raise argparse.ArgumentTypeError(
            "hash prefix must be at least 10 hex characters"
        )
    return value.lower()


def _add_global_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-c", "--config", help="path to config directory")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kdb",
        description="Pentest knowledge database — semantic search over aggregated docs",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="clone/pull source repos")
    _add_global_args(p_fetch)
    p_fetch.add_argument("--only", help="only process this source")

    p_index = sub.add_parser("index", help="(re)build vector index")
    _add_global_args(p_index)
    p_index.add_argument("--force", action="store_true", help="full rebuild")

    p_update = sub.add_parser("update", help="fetch + index (one-shot)")
    _add_global_args(p_update)
    p_update.add_argument("--force", action="store_true", help="full rebuild")

    p_search = sub.add_parser("search", help="search indexed knowledge")
    _add_global_args(p_search)
    p_search.add_argument("query", nargs="?", help="search query")
    p_search.add_argument("--json", action="store_true", help="JSON output")
    p_search.add_argument("--top-k", type=int, default=None, help="number of results")
    p_search.add_argument("--source", help="filter by source name")

    p_list = sub.add_parser("list-sources", help="list all configured sources")
    _add_global_args(p_list)

    p_get = sub.add_parser("get", help="Retrieve a section by content hash prefix")
    _add_global_args(p_get)
    p_get.add_argument(
        "hash_prefix", type=_validate_hex_prefix, help="Hash prefix (min 10 hex chars)"
    )
    p_get.add_argument("--json", action="store_true", help="JSON output")

    return parser


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _format_search_results(results: list[dict]) -> str | None:
    if not results:
        return None

    term_width = shutil.get_terminal_size((80, 20)).columns
    content_width = int(term_width * 0.85)

    hash_w = 12
    dist_w = 9
    remaining = content_width - hash_w - dist_w
    tag_w = int(remaining * 0.35)
    title_w = remaining - tag_w

    lines: list[str] = []

    if term_width < 80:
        lines.append(f"{'Hash':<14} {'Tag':<{tag_w}}")
        lines.append("─" * (14 + tag_w + 2))
        for r in results:
            tag = f"{r['category']}·{r.get('source_title', r['source'])}"
            h = r["content_hash"][:12] if r.get("content_hash") else "?" * 12
            title = _truncate(r["title"], title_w)
            dist = f"{r['distance']:.2f}"
            lines.append(f"{h:<14} {_truncate(tag, tag_w):<{tag_w}}")
            lines.append(f"{'':<14} {title:<{title_w}} {dist:>{dist_w}}")
    else:
        header = (
            f"{'Hash':<{hash_w}} {'Tag':<{tag_w}} "
            f"{'Title':<{title_w}} {'Distance':<{dist_w}}"
        )
        lines.append(header)
        lines.append("─" * len(header))
        for r in results:
            tag = f"{r['category']}·{r.get('source_title', r['source'])}"
            h = r["content_hash"][:12] if r.get("content_hash") else "?" * 12
            title = _truncate(r["title"], title_w)
            dist = f"{r['distance']:.2f}"
            lines.append(
                f"{h:<{hash_w}} {_truncate(tag, tag_w):<{tag_w}} "
                f"{title:<{title_w}} {dist:>{dist_w}}"
            )

    return "\n".join(lines)


def _print_get_result(result: dict) -> None:
    """Print a formatted section result for ``kdb get``."""
    print(f"Hash:\t\t{result['hash']}")
    print(f"Source:\t\t{result['source']}")
    print(f"Title:\t\t{result['title']}")
    print(f"Category:\t{result['category']}")
    print(f"Path:\t\t{result['path']}")
    print(f"Heading:\t{result['heading_path']}")
    print()
    print("--- Content ---")
    print(result["body"])


def main() -> None:
    """CLI entry point: parse args, dispatch to subcommand handlers."""
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = _build_parser()
    args = parser.parse_args()

    try:
        match args.command:
            case "fetch":
                from knowledge.fetch import fetch_sources
                from knowledge.config import ensure_data_dir
                from knowledge.sources import load_sources

                data_dir = ensure_data_dir(resolve_data_dir(args.config))
                sources = load_sources(resolve_sources_yaml(args.config))
                changed = fetch_sources(
                    sources,
                    data_dir,
                    only=args.only,
                    verbose=args.verbose,
                    config_dir=args.config,
                )
                if changed:
                    print(f"Updated: {', '.join(changed)}")
                else:
                    print("All sources up to date.")

            case "index":
                from knowledge.indexer import cmd_index

                cmd_index(
                    config_dir=args.config, force=args.force, verbose=args.verbose
                )

            case "update":
                from knowledge.fetch import fetch_sources
                from knowledge.indexer import cmd_index
                from knowledge.config import ensure_data_dir
                from knowledge.sources import load_sources

                data_dir = ensure_data_dir(resolve_data_dir(args.config))
                sources = load_sources(resolve_sources_yaml(args.config))
                print("Fetching sources...")
                fetch_sources(
                    sources, data_dir, verbose=args.verbose, config_dir=args.config
                )
                print("Indexing...")
                cmd_index(
                    config_dir=args.config, force=args.force, verbose=args.verbose
                )

            case "search":
                from knowledge.config import load_config
                from knowledge.search import cmd_search

                if not args.query:
                    print("Error: search query is required")
                    sys.exit(1)

                cfg = load_config(args.config)
                top_k = (
                    args.top_k if args.top_k is not None else cfg.search.default_top_k
                )
                if top_k < 1:
                    print("Error: --top-k must be >= 1", file=sys.stderr)
                    sys.exit(1)

                results = cmd_search(
                    query=args.query,
                    top_k=top_k,
                    source=args.source,
                    config_dir=args.config,
                )

                if args.json:
                    print(json.dumps(results, indent=2, ensure_ascii=False))
                elif results:
                    output = _format_search_results(results)
                    if output:
                        print(output)
                    else:
                        for r in results:
                            print(f"{r['source']}: {r['title']} ({r['distance']:.2f})")

            case "list-sources":
                from knowledge.sources import load_sources

                data_dir = resolve_data_dir(args.config)
                sources = load_sources(resolve_sources_yaml(args.config))

                header = f"{'Name':<34} {'Title':<50} {'Category':<10} {'Type':<8} {'Status':<12}"
                print(header)
                print("-" * len(header))
                for s in sources:
                    if s.source_type == "local":
                        st = (
                            "available"
                            if s.path and Path(s.path).expanduser().exists()
                            else "missing"
                        )
                    else:
                        st = (
                            "cloned"
                            if (data_dir / "sources" / s.name).exists()
                            else "not-cloned"
                        )
                    print(
                        f"{s.name[:32]:<34} {s.title[:48]:<50} {s.category[:8]:<10} "
                        f"{s.source_type:<8} {st:<12}"
                    )

            case "get":
                from knowledge.getter import cmd_get

                result = cmd_get(args.hash_prefix, config_dir=args.config)
                if result is None:
                    sys.exit(1)
                if args.json:
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    _print_get_result(result)

            case _:
                print(f"Error: unknown command '{args.command}'", file=sys.stderr)
                sys.exit(1)

    except ConfigError as e:
        print(f"Config error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
