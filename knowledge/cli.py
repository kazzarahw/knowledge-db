"""CLI entry point — argparse definitions and main()."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from knowledge.config import VERSION, resolve_data_dir, resolve_sources_yaml
from knowledge.sources import ConfigError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kdb",
        description="Pentest knowledge database — semantic search over aggregated docs",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-c", "--config", help="path to config directory")

    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="clone/pull source repos")
    p_fetch.add_argument("--only", help="only process this source")

    p_index = sub.add_parser("index", help="(re)build vector index")
    p_index.add_argument("--force", action="store_true", help="full rebuild")

    p_update = sub.add_parser("update", help="fetch + index (one-shot)")
    p_update.add_argument("--force", action="store_true", help="full rebuild")

    p_search = sub.add_parser("search", help="search indexed knowledge")
    p_search.add_argument("query", nargs="?", help="search query")
    p_search.add_argument("--json", action="store_true", help="JSON output")
    p_search.add_argument("--top-k", type=int, default=None, help="number of results")
    p_search.add_argument("--source", help="filter by source name")

    sub.add_parser("list-sources", help="list all configured sources")

    return parser


def main() -> None:
    """CLI entry point: parse args, dispatch to subcommand handlers."""
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
                    header = f"{'Source':<16} {'Title':<25} {'Category':<10} {'Heading Path':<17} {'Distance':<8}"
                    print(header)
                    print("-" * len(header))
                    for r in results:
                        print(
                            f"{r['source'][:14]:<16} {r['title'][:23]:<25} "
                            f"{r['category'][:8]:<10} {r['heading_path'][:15]:<17} "
                            f"{r['distance']:<8.4f}"
                        )

            case "list-sources":
                from knowledge.sources import load_sources

                data_dir = resolve_data_dir(args.config)
                sources = load_sources(resolve_sources_yaml(args.config))

                header = f"{'Name':<20} {'Title':<25} {'Category':<10} {'Type':<8} {'Status':<12}"
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
                        f"{s.name[:18]:<20} {s.title[:23]:<25} {s.category[:8]:<10} "
                        f"{s.source_type:<8} {st:<12}"
                    )

            case _:
                print(f"Error: unknown command '{args.command}'", file=sys.stderr)
                sys.exit(1)

    except ConfigError as e:
        print(f"Config error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
