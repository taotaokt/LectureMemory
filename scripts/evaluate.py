"""Evaluate BM25, multimodal embedding retrieval, and Qwen reranking."""

from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from time import perf_counter

from sqlalchemy import select

from app.config import PROJECT_ROOT, get_settings
from app.database import create_database_engine, create_session_factory
from app.embeddings import Qwen3VLEmbeddingProvider
from app.models import Course
from app.retrieval import (
    FaissVectorIndex,
    IndexPersistenceError,
    Qwen3VLReranker,
    search_lecture_memory_by_vector,
)
from app.schemas import SearchResult
from benchmark import load_benchmark_queries, resolve_database_references
from benchmark.bm25 import BM25Index, load_bm25_documents
from benchmark.evaluation import (
    MethodEvaluation,
    PageKey,
    QueryEvaluation,
    build_evaluation_report,
    write_evaluation_report,
)

METHOD_LABELS = {
    "bm25": "BM25",
    "embedding": "Qwen Embedding",
    "reranker": "Qwen + Reranker",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "queries.jsonl",
    )
    parser.add_argument("--database", type=Path, help="Override DATABASE_PATH.")
    parser.add_argument("--index-dir", type=Path, help="Override INDEX_DIR.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "results",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_LABELS),
        default=tuple(METHOD_LABELS),
    )
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--retrieval-top-k", type=int)
    parser.add_argument("--rerank-top-k", type=int)
    parser.add_argument(
        "--reranker-weight",
        type=float,
        help="Reranker contribution from 0 (embedding only) to 1 (reranker only).",
    )
    parser.add_argument("--run-id", help="Optional stable output filename suffix.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    database_path = (args.database or settings.database_path)
    index_dir = args.index_dir or settings.index_dir
    methods = tuple(dict.fromkeys(args.methods))
    if database_path is None or not database_path.expanduser().resolve().is_file():
        raise SystemExit(f"Database does not exist: {database_path}")
    if args.max_k < 10:
        raise SystemExit("--max-k must be at least 10 to report Recall@10")

    uses_embeddings = "embedding" in methods or "reranker" in methods
    retrieval_top_k = args.retrieval_top_k or max(settings.retrieval_top_k, args.max_k)
    rerank_top_k = args.rerank_top_k or max(settings.rerank_top_k, args.max_k)
    reranker_weight = (
        settings.reranker_weight
        if args.reranker_weight is None
        else args.reranker_weight
    )
    if uses_embeddings and retrieval_top_k < args.max_k:
        raise SystemExit("--retrieval-top-k must be at least --max-k")
    if "reranker" in methods and not args.max_k <= rerank_top_k <= retrieval_top_k:
        raise SystemExit(
            "reranking requires --max-k <= --rerank-top-k <= --retrieval-top-k"
        )
    if not 0.0 <= reranker_weight <= 1.0:
        raise SystemExit("--reranker-weight must be between 0 and 1")

    queries = load_benchmark_queries(args.dataset)
    engine = create_database_engine(database_path)
    factory = create_session_factory(engine)
    runs: list[QueryEvaluation] = []
    try:
        with factory() as session:
            resolved = resolve_database_references(session, queries)
            relevant_by_query: dict[str, list[PageKey]] = defaultdict(list)
            for page in resolved:
                relevant_by_query[page.query_id].append(
                    (page.lecture_id, page.page_number)
                )
            course_ids = dict(session.execute(select(Course.code, Course.id)).all())

            bm25 = (
                BM25Index(load_bm25_documents(session)) if "bm25" in methods else None
            )
            provider = None
            vector_index = None
            reranker = None
            if uses_embeddings:
                if index_dir is None:
                    raise SystemExit("INDEX_DIR is not configured")
                try:
                    vector_index = FaissVectorIndex.load(index_dir)
                except IndexPersistenceError as exc:
                    raise SystemExit(f"Search index could not be loaded: {exc}") from exc
                provider = Qwen3VLEmbeddingProvider(
                    model_name=settings.model_name,
                    device=settings.device,
                    dtype=settings.embedding_dtype,
                    dimension=settings.embedding_dimension,
                    batch_size=settings.embedding_batch_size,
                    max_pixels=settings.embedding_max_pixels,
                    query_instruction=settings.embedding_query_instruction,
                )
                if provider.dimension != vector_index.dimension:
                    raise SystemExit(
                        f"Embedding dimension {provider.dimension} does not match "
                        f"index dimension {vector_index.dimension}"
                    )
                if "reranker" in methods:
                    reranker = Qwen3VLReranker(
                        model_name=settings.reranker_model_name,
                        device=settings.device,
                        dtype=settings.reranker_dtype,
                        batch_size=settings.reranker_batch_size,
                        max_length=settings.reranker_max_length,
                        min_pixels=settings.reranker_min_pixels,
                        max_pixels=settings.reranker_max_pixels,
                        instruction=settings.reranker_instruction,
                    )

            query_contexts = []
            for query in queries:
                course_id = course_ids.get(query.course)
                if course_id is None:
                    raise SystemExit(f"Course code is not in the database: {query.course}")
                relevant_pages = tuple(relevant_by_query[query.id])
                query_contexts.append((query, course_id, relevant_pages))

            if bm25 is not None:
                for position, (query, course_id, relevant_pages) in enumerate(
                    query_contexts,
                    start=1,
                ):
                    started = perf_counter()
                    results = bm25.search(
                        query.query,
                        top_k=args.max_k,
                        course_id=course_id,
                    )
                    retrieval_ms = _elapsed_ms(started)
                    runs.append(
                        QueryEvaluation(
                            method=METHOD_LABELS["bm25"],
                            query_id=query.id,
                            ranked_pages=tuple(
                                (result.lecture_id, result.page_number)
                                for result in results
                            ),
                            relevant_pages=relevant_pages,
                            retrieval_ms=retrieval_ms,
                            total_ms=retrieval_ms,
                        )
                    )
                    print(f"BM25 {position}/{len(queries)}: {query.id}")

            if provider is not None and vector_index is not None:
                query_vectors = {}
                embedding_times = {}
                for position, (query, _, _) in enumerate(query_contexts, start=1):
                    started = perf_counter()
                    query_vectors[query.id] = provider.embed_query(query.query)
                    embedding_times[query.id] = _elapsed_ms(started)
                    print(f"Embedded {position}/{len(queries)}: {query.id}")

                provider = None
                _release_model_memory()
                candidates_by_query = {}
                retrieval_times = {}
                for position, (query, course_id, relevant_pages) in enumerate(
                    query_contexts,
                    start=1,
                ):
                    started = perf_counter()
                    candidates = search_lecture_memory_by_vector(
                        session,
                        query_vectors[query.id],
                        index=vector_index,
                        top_k=retrieval_top_k,
                        course_id=course_id,
                    )
                    retrieval_ms = _elapsed_ms(started)
                    candidates_by_query[query.id] = candidates
                    retrieval_times[query.id] = retrieval_ms
                    if "embedding" in methods:
                        runs.append(
                            QueryEvaluation(
                                method=METHOD_LABELS["embedding"],
                                query_id=query.id,
                                ranked_pages=_ranked_page_keys(candidates, args.max_k),
                                relevant_pages=relevant_pages,
                                embedding_ms=embedding_times[query.id],
                                retrieval_ms=retrieval_ms,
                                total_ms=embedding_times[query.id] + retrieval_ms,
                            )
                        )
                    print(f"Retrieved {position}/{len(queries)}: {query.id}")

                if reranker is not None:
                    for position, (query, _, relevant_pages) in enumerate(
                        query_contexts,
                        start=1,
                    ):
                        started = perf_counter()
                        reranked = reranker.rerank(
                            query.query,
                            candidates_by_query[query.id][:rerank_top_k],
                            reranker_weight=reranker_weight,
                        )
                        reranking_ms = _elapsed_ms(started)
                        runs.append(
                            QueryEvaluation(
                                method=METHOD_LABELS["reranker"],
                                query_id=query.id,
                                ranked_pages=_ranked_page_keys(reranked, args.max_k),
                                relevant_pages=relevant_pages,
                                embedding_ms=embedding_times[query.id],
                                retrieval_ms=retrieval_times[query.id],
                                reranking_ms=reranking_ms,
                                total_ms=(
                                    embedding_times[query.id]
                                    + retrieval_times[query.id]
                                    + reranking_ms
                                ),
                            )
                        )
                        print(f"Reranked {position}/{len(queries)}: {query.id}")

    finally:
        engine.dispose()

    report = build_evaluation_report(
        runs,
        dataset=args.dataset,
        configuration={
            "methods": list(methods),
            "max_k": args.max_k,
            "retrieval_top_k": retrieval_top_k,
            "rerank_top_k": rerank_top_k,
            "reranker_weight": reranker_weight if "reranker" in methods else None,
            "embedding_model": settings.model_name if uses_embeddings else None,
            "embedding_dimension": settings.embedding_dimension if uses_embeddings else None,
            "reranker_model": (
                settings.reranker_model_name if "reranker" in methods else None
            ),
            "device": settings.device if uses_embeddings else None,
            "embedding_dtype": settings.embedding_dtype if uses_embeddings else None,
            "reranker_dtype": (
                settings.reranker_dtype if "reranker" in methods else None
            ),
            "reranker_batch_size": (
                settings.reranker_batch_size if "reranker" in methods else None
            ),
            "reranker_max_pixels": (
                settings.reranker_max_pixels if "reranker" in methods else None
            ),
        },
    )
    json_path, csv_path = write_evaluation_report(
        report,
        args.output_dir,
        run_id=args.run_id,
    )
    _print_summary(report.methods)
    print(f"Detailed JSON: {json_path}")
    print(f"Summary CSV:   {csv_path}")


def _ranked_page_keys(
    results: tuple[SearchResult, ...],
    limit: int,
) -> tuple[PageKey, ...]:
    pages: list[PageKey] = []
    seen: set[PageKey] = set()
    for result in results:
        if result.page_number is None:
            continue
        key = (result.lecture_id, result.page_number)
        if key in seen:
            continue
        pages.append(key)
        seen.add(key)
        if len(pages) == limit:
            break
    return tuple(pages)


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000


def _release_model_memory() -> None:
    """Release one Qwen model before loading the next evaluation stage."""
    gc.collect()
    torch = import_module("torch")
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_summary(methods: tuple[MethodEvaluation, ...]) -> None:
    print()
    print("Method                 R@1    R@5   R@10    MRR   Total ms")
    print("-----------------------------------------------------------")
    for method in methods:
        print(
            f"{method.method:<22} "
            f"{method.recall_at_1:>5.3f} "
            f"{method.recall_at_5:>6.3f} "
            f"{method.recall_at_10:>6.3f} "
            f"{method.mrr:>6.3f} "
            f"{method.average_total_ms:>10.1f}"
        )


if __name__ == "__main__":
    main()
