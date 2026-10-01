"""Generate embeddings and rebuild the unified FAISS search index."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.database import create_database_engine, create_session_factory, session_scope
from app.embeddings import EmbeddingCache, Qwen3VLEmbeddingProvider
from app.services.indexing_service import IndexingProgress, build_search_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, help="Override DATABASE_PATH.")
    parser.add_argument("--embedding-dir", type=Path, help="Override EMBEDDING_DIR.")
    parser.add_argument("--index-dir", type=Path, help="Override INDEX_DIR.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Commit embedding cache metadata every N processed entities.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    database_path = (args.database or settings.database_path)
    embedding_dir = args.embedding_dir or settings.embedding_dir
    index_dir = args.index_dir or settings.index_dir
    if database_path is None or not database_path.expanduser().resolve().is_file():
        raise SystemExit(f"Database does not exist: {database_path}")
    if embedding_dir is None:
        raise SystemExit("EMBEDDING_DIR is not configured")
    if index_dir is None:
        raise SystemExit("INDEX_DIR is not configured")
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive")

    provider = Qwen3VLEmbeddingProvider(
        model_name=settings.model_name,
        device=settings.device,
        dtype=settings.embedding_dtype,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_pixels=settings.embedding_max_pixels,
        query_instruction=settings.embedding_query_instruction,
    )
    cache = EmbeddingCache(embedding_dir)
    engine = create_database_engine(database_path)
    factory = create_session_factory(engine)
    print(
        f"Building {provider.model_name} index from {database_path} "
        f"into {Path(index_dir).expanduser().resolve()}..."
    )
    try:
        with session_scope(factory) as session:
            def report_progress(progress: IndexingProgress) -> None:
                print(
                    f"[{progress.processed}/{progress.total}] "
                    f"{progress.entity_type}:{progress.entity_id} {progress.status}",
                    flush=True,
                )
                if progress.processed % args.checkpoint_every == 0:
                    session.commit()

            summary = build_search_index(
                session,
                provider=provider,
                cache=cache,
                index_dir=index_dir,
                progress_callback=report_progress,
            )
    finally:
        engine.dispose()

    print(
        f"Indexed {summary.indexed_entities}/{summary.total_entities} entities "
        f"({summary.indexed_slides} slides, {summary.indexed_notes} notes)."
    )
    print(
        f"Embeddings: {summary.generated_embeddings} generated, "
        f"{summary.cached_embeddings} cached, {summary.failed_entities} failed."
    )
    for failure in summary.failures:
        print(
            f"- {failure.entity_type}:{failure.entity_id} "
            f"({failure.error_type}): {failure.message}"
        )
    if not summary.complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
