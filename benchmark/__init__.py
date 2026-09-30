"""Retrieval benchmarks for Lecture Memory."""

from benchmark.bm25 import (
    BM25Document,
    BM25Index,
    BM25SearchResult,
    load_bm25_documents,
    tokenize,
)
from benchmark.dataset import (
    BenchmarkDatasetError,
    BenchmarkQuery,
    BenchmarkSource,
    RelevantPage,
    ResolvedRelevantPage,
    load_benchmark_queries,
    resolve_database_references,
    validate_source_pdfs,
)
from benchmark.evaluation import (
    EvaluationReport,
    MethodEvaluation,
    PageKey,
    QueryEvaluation,
    build_evaluation_report,
    write_evaluation_report,
)

__all__ = [
    "BM25Document",
    "BM25Index",
    "BM25SearchResult",
    "BenchmarkDatasetError",
    "BenchmarkQuery",
    "BenchmarkSource",
    "EvaluationReport",
    "MethodEvaluation",
    "PageKey",
    "QueryEvaluation",
    "RelevantPage",
    "ResolvedRelevantPage",
    "build_evaluation_report",
    "load_bm25_documents",
    "load_benchmark_queries",
    "resolve_database_references",
    "tokenize",
    "validate_source_pdfs",
    "write_evaluation_report",
]
