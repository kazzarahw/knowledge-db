"""CLI entry point — argparse definitions and main()."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from knowledge.config import VERSION
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
    p_update.add_argument("--force", action="store_true")

    p_search = sub.add_parser("search", help="search indexed knowledge")
    p_search.add_argument("query", nargs="?", help="search query")
    p_search.add_argument("--json", action="store_true", help="JSON output")
    p_search.add_argument("--top-k", type=int, default=10, help="number of results")
    p_search.add_argument("--source", help="filter by source name")

    sub.add_parser("list-sources", help="list all configured sources")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "search" and args.top_k is not None and args.top_k < 1:
            print("Error: --top-k must be >= 1", file=sys.stderr)
            sys.exit(1)

        if args.command == "fetch":
            from knowledge.fetch import fetch_sources
            from knowledge.config import resolve_data_dir, ensure_data_dir
            from knowledge.sources import load_sources

            data_dir = ensure_data_dir(resolve_data_dir(args.config))
            config_dir = args.config or str(data_dir.parent)
            sp = Path(config_dir) / "sources.yaml"
            if not sp.exists():
                sp = Path.cwd() / "sources.yaml"
            sources = load_sources(sp)
            changed = fetch_sources(
                sources, data_dir, only=args.only, verbose=args.verbose
            )
            if changed:
                print(f"Updated: {', '.join(changed)}")
            else:
                print("All sources up to date.")

        elif args.command == "index":
            from knowledge.indexer import cmd_index

            cmd_index(config_dir=args.config, force=args.force, verbose=args.verbose)

        elif args.command == "update":
            from knowledge.fetch import fetch_sources
            from knowledge.indexer import cmd_index
            from knowledge.config import resolve_data_dir, ensure_data_dir
            from knowledge.sources import load_sources

            data_dir = ensure_data_dir(resolve_data_dir(args.config))
            config_dir = args.config or str(data_dir.parent)
            sp = Path(config_dir) / "sources.yaml"
            if not sp.exists():
                sp = Path.cwd() / "sources.yaml"
            sources = load_sources(sp)
            print("Fetching sources...")
            fetch_sources(sources, data_dir, verbose=args.verbose)
            print("Indexing...")
            cmd_index(config_dir=args.config, force=args.force, verbose=args.verbose)

        elif args.command == "search":
            from knowledge.search import cmd_search

            if not args.query:
                print("Error: search query is required")
                sys.exit(1)

            results = cmd_search(
                query=args.query,
                top_k=args.top_k,
                source=args.source,
                json_output=args.json,
                config_dir=args.config,
            )

            if args.json:
                print(json.dumps(results, indent=2, ensure_ascii=False))
            else:
                if not results:
                    print("No results found.")
                    return
                header = f"{'Source':<20} {'Title':<40} {'Category':<15} {'Heading Path':<45} {'Distance':<8}"
                print(header)
                print("-" * len(header))
                for r in results:
                    print(
                        f"{r['source'][:18]:<20} {r['title'][:38]:<40} "
                        f"{r['category'][:13]:<15} {r['heading_path'][:43]:<45} "
                        f"{r['distance']:.4f}"
                    )

        elif args.command == "list-sources":
            from knowledge.sources import load_sources
            from knowledge.config import resolve_data_dir

            data_dir = resolve_data_dir(args.config)
            config_dir = args.config or str(data_dir.parent)
            sp = Path(config_dir) / "sources.yaml"
            if not sp.exists():
                sp = Path.cwd() / "sources.yaml"
            sources = load_sources(sp)

            header = f"{'Name':<25} {'Title':<40} {'Category':<15} {'Type':<8} {'Status':<12}"
            print(header)
            print("-" * len(header))
            for s in sources:
                if s.type == "local":
                    status = (
                        "available"
                        if s.path and Path(s.path).expanduser().exists()
                        else "missing"
                    )
                else:
                    status = (
                        "cloned"
                        if (data_dir / "sources" / s.name).exists()
                        else "not-cloned"
                    )
                print(
                    f"{s.name:<25} {s.title:<40} {s.category:<15} {s.type:<8} {status:<12}"
                )

    except ConfigError as e:
        print(f"Config error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
