"""Retrieval evaluation metrics and durable report output."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

PageKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """Ranked page output and latency measurements for one query and method."""

    method: str
    query_id: str
    ranked_pages: tuple[PageKey, ...]
    relevant_pages: tuple[PageKey, ...]
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    total_ms: float = 0.0

    def recall_at(self, cutoff: int) -> float:
        """Return the fraction of relevant pages present in the first ``cutoff`` results."""
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if not self.relevant_pages:
            raise ValueError("relevant_pages must not be empty")
        relevant = set(self.relevant_pages)
        retrieved = set(self.ranked_pages[:cutoff])
        return len(relevant & retrieved) / len(relevant)

    @property
    def reciprocal_rank(self) -> float:
        """Return reciprocal rank of the first relevant page, or zero."""
        relevant = set(self.relevant_pages)
        for rank, page in enumerate(self.ranked_pages, start=1):
            if page in relevant:
                return 1.0 / rank
        return 0.0


@dataclass(frozen=True, slots=True)
class MethodEvaluation:
    """Aggregate quality and latency metrics for one retrieval method."""

    method: str
    query_count: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    average_embedding_ms: float
    average_retrieval_ms: float
    average_reranking_ms: float
    average_total_ms: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Complete reproducible benchmark result."""

    generated_at: str
    dataset: str
    methods: tuple[MethodEvaluation, ...]
    queries: tuple[QueryEvaluation, ...]


def build_evaluation_report(
    runs: tuple[QueryEvaluation, ...] | list[QueryEvaluation],
    *,
    dataset: str | Path,
    generated_at: datetime | None = None,
) -> EvaluationReport:
    """Aggregate per-query runs and verify that methods cover the same query set."""
    query_runs = tuple(runs)
    if not query_runs:
        raise ValueError("runs must not be empty")
    identities = [(run.method, run.query_id) for run in query_runs]
    if len(identities) != len(set(identities)):
        raise ValueError("runs must contain one result per method and query ID")
    for run in query_runs:
        if not run.method.strip() or not run.query_id.strip():
            raise ValueError("method and query_id must not be blank")
        if not run.relevant_pages:
            raise ValueError("every run must contain at least one relevant page")
        if min(run.embedding_ms, run.retrieval_ms, run.reranking_ms, run.total_ms) < 0:
            raise ValueError("latencies must not be negative")

    grouped: dict[str, list[QueryEvaluation]] = {}
    for run in query_runs:
        grouped.setdefault(run.method, []).append(run)
    expected_query_ids = {run.query_id for run in query_runs}
    if any(
        {run.query_id for run in method_runs} != expected_query_ids
        for method_runs in grouped.values()
    ):
        raise ValueError("every method must evaluate the same query IDs")

    methods = tuple(
        _summarize_method(method, tuple(method_runs))
        for method, method_runs in grouped.items()
    )
    timestamp = generated_at or datetime.now(UTC)
    return EvaluationReport(
        generated_at=timestamp.isoformat(),
        dataset=str(Path(dataset).expanduser().resolve()),
        methods=methods,
        queries=query_runs,
    )


def write_evaluation_report(
    report: EvaluationReport,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
) -> tuple[Path, Path]:
    """Write detailed JSON and summary CSV reports and return their paths."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not identifier or any(character not in allowed_characters for character in identifier):
        raise ValueError("run_id may contain only letters, numbers, hyphens, and underscores")

    json_path = destination / f"evaluation-{identifier}.json"
    csv_path = destination / f"evaluation-{identifier}.csv"
    json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "method",
                "query_count",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "mrr",
                "average_embedding_ms",
                "average_retrieval_ms",
                "average_reranking_ms",
                "average_total_ms",
            ),
        )
        writer.writeheader()
        for method in report.methods:
            writer.writerow(asdict(method))
    return json_path, csv_path


def _summarize_method(
    method: str,
    runs: tuple[QueryEvaluation, ...],
) -> MethodEvaluation:
    count = len(runs)
    return MethodEvaluation(
        method=method,
        query_count=count,
        recall_at_1=sum(run.recall_at(1) for run in runs) / count,
        recall_at_5=sum(run.recall_at(5) for run in runs) / count,
        recall_at_10=sum(run.recall_at(10) for run in runs) / count,
        mrr=sum(run.reciprocal_rank for run in runs) / count,
        average_embedding_ms=sum(run.embedding_ms for run in runs) / count,
        average_retrieval_ms=sum(run.retrieval_ms for run in runs) / count,
        average_reranking_ms=sum(run.reranking_ms for run in runs) / count,
        average_total_ms=sum(run.total_ms for run in runs) / count,
    )
