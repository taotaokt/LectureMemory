"""Validate the retrieval benchmark schema and its local source PDFs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from app.database import create_database_engine, create_session_factory
from benchmark import (
    load_benchmark_queries,
    resolve_database_references,
    validate_source_pdfs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "queries.jsonl",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="Also validate labels against an already-ingested SQLite database.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    queries = load_benchmark_queries(args.dataset)
    sources = validate_source_pdfs(queries, args.raw_root)

    print(f"Validated {len(queries)} benchmark queries across {len(sources)} PDFs.")
    for label, values in (
        ("Languages", Counter(query.language for query in queries)),
        ("Query types", Counter(query.query_type for query in queries)),
        ("Difficulties", Counter(query.difficulty for query in queries)),
    ):
        summary = ", ".join(f"{key}={value}" for key, value in sorted(values.items()))
        print(f"{label}: {summary}")
    for source in sources:
        print(
            f"- {source.course}/{source.source_file}: "
            f"{source.page_count} pages, {source.query_count} queries"
        )

    if args.database is not None:
        engine = create_database_engine(args.database)
        factory = create_session_factory(engine)
        try:
            with factory() as session:
                resolved = resolve_database_references(session, queries)
        finally:
            engine.dispose()
        print(f"Resolved {len(resolved)} relevant-page labels against the database.")


if __name__ == "__main__":
    main()
