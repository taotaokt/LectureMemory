"""Create an idempotent local benchmark database from labeled source PDFs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT, get_settings
from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from app.models import Course, Lecture
from app.repositories.course_repository import create_course
from app.repositories.lecture_repository import create_lecture
from app.schemas import CourseCreate, LectureCreate
from app.services.ingestion_service import IngestionStatus, ingest_lecture_pdf
from benchmark import (
    BenchmarkDatasetError,
    BenchmarkQuery,
    load_benchmark_queries,
    resolve_database_references,
    validate_source_pdfs,
)


@dataclass(frozen=True, slots=True)
class BenchmarkLecture:
    """One unique lecture inferred from stable benchmark labels."""

    course_code: str
    title: str
    source_file: str
    lecture_number: int


@dataclass(frozen=True, slots=True)
class BenchmarkPreparationSummary:
    """Aggregate result of preparing benchmark persistence and rendered pages."""

    database_path: Path
    course_count: int
    lecture_count: int
    page_count: int
    ingested_lectures: int
    skipped_lectures: int


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
        help="Directory containing one subdirectory per course code.",
    )
    parser.add_argument("--database", type=Path, help="Override DATABASE_PATH.")
    parser.add_argument("--rendered-dir", type=Path, help="Override RENDERED_DIR.")
    return parser


def collect_benchmark_lectures(
    queries: tuple[BenchmarkQuery, ...],
) -> tuple[BenchmarkLecture, ...]:
    """Collect unique lectures in dataset order and reject ambiguous sources."""
    source_by_lecture: dict[tuple[str, str], str] = {}
    lecture_by_source: dict[tuple[str, str], str] = {}
    positions_by_course: dict[str, int] = {}
    lectures: list[BenchmarkLecture] = []

    for query in queries:
        lecture_key = (query.course, query.lecture)
        source_key = (query.course, query.source_file)
        known_source = source_by_lecture.get(lecture_key)
        if known_source is not None and known_source != query.source_file:
            raise BenchmarkDatasetError(
                f"Lecture {query.course}/{query.lecture} references both "
                f"{known_source} and {query.source_file}"
            )
        known_lecture = lecture_by_source.get(source_key)
        if known_lecture is not None and known_lecture != query.lecture:
            raise BenchmarkDatasetError(
                f"Source {query.course}/{query.source_file} labels both "
                f"{known_lecture} and {query.lecture}"
            )
        if known_source is not None:
            continue

        source_by_lecture[lecture_key] = query.source_file
        lecture_by_source[source_key] = query.lecture
        lecture_number = positions_by_course.get(query.course, 0) + 1
        positions_by_course[query.course] = lecture_number
        lectures.append(
            BenchmarkLecture(
                course_code=query.course,
                title=query.lecture,
                source_file=query.source_file,
                lecture_number=lecture_number,
            )
        )
    return tuple(lectures)


def prepare_benchmark_database(
    queries: tuple[BenchmarkQuery, ...],
    *,
    database_path: str | Path,
    raw_root: str | Path,
    rendered_root: str | Path,
) -> BenchmarkPreparationSummary:
    """Create missing courses and lectures, ingest PDFs, and validate all labels."""
    source_root = Path(raw_root).expanduser().resolve()
    destination_database = Path(database_path).expanduser().resolve()
    destination_rendered = Path(rendered_root).expanduser().resolve()
    validate_source_pdfs(queries, source_root)
    lectures = collect_benchmark_lectures(queries)

    engine = create_database_engine(destination_database)
    init_database(engine)
    factory = create_session_factory(engine)
    ingested_lectures = 0
    skipped_lectures = 0
    page_count = 0
    try:
        with session_scope(factory) as session:
            for item in lectures:
                course = _get_or_create_course(session, item.course_code)
                lecture = _get_or_create_lecture(session, course, item)
                summary = ingest_lecture_pdf(
                    session,
                    lecture.id,
                    source_root / item.course_code / item.source_file,
                    raw_root=source_root,
                    rendered_root=destination_rendered,
                )
                page_count += summary.total_pages
                ingested_lectures += int(summary.status == IngestionStatus.INGESTED)
                skipped_lectures += int(summary.status == IngestionStatus.SKIPPED)
                print(
                    f"{item.course_code}/{item.title}: {summary.status} "
                    f"({summary.total_pages} pages)",
                    flush=True,
                )
            resolve_database_references(session, queries)
    finally:
        engine.dispose()

    return BenchmarkPreparationSummary(
        database_path=destination_database,
        course_count=len({lecture.course_code for lecture in lectures}),
        lecture_count=len(lectures),
        page_count=page_count,
        ingested_lectures=ingested_lectures,
        skipped_lectures=skipped_lectures,
    )


def _get_or_create_course(session: Session, code: str) -> Course:
    matches = tuple(
        session.scalars(select(Course).where(Course.code == code).order_by(Course.id)).all()
    )
    if len(matches) > 1:
        raise BenchmarkDatasetError(
            f"Benchmark course code {code} is ambiguous; found {len(matches)} courses"
        )
    if matches:
        return matches[0]
    return create_course(
        session,
        CourseCreate(
            name=code,
            code=code,
            description="Local retrieval benchmark course.",
        ),
    )


def _get_or_create_lecture(
    session: Session,
    course: Course,
    item: BenchmarkLecture,
) -> Lecture:
    matches = tuple(
        session.scalars(
            select(Lecture)
            .where(Lecture.course_id == course.id, Lecture.title == item.title)
            .order_by(Lecture.id)
        ).all()
    )
    if len(matches) > 1:
        raise BenchmarkDatasetError(
            f"Benchmark lecture {item.course_code}/{item.title} is ambiguous; "
            f"found {len(matches)} lectures"
        )
    if matches:
        return matches[0]
    return create_lecture(
        session,
        course.id,
        LectureCreate(title=item.title, lecture_number=item.lecture_number),
    )


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    database_path = args.database or settings.database_path
    rendered_root = args.rendered_dir or settings.rendered_dir
    if database_path is None:
        raise SystemExit("DATABASE_PATH is not configured")
    if rendered_root is None:
        raise SystemExit("RENDERED_DIR is not configured")

    queries = load_benchmark_queries(args.dataset)
    summary = prepare_benchmark_database(
        queries,
        database_path=database_path,
        raw_root=args.raw_root,
        rendered_root=rendered_root,
    )
    print(
        f"Prepared {summary.course_count} courses, {summary.lecture_count} lectures, "
        f"and {summary.page_count} pages in {summary.database_path}."
    )
    print(
        f"Lectures: {summary.ingested_lectures} ingested, "
        f"{summary.skipped_lectures} already complete."
    )


if __name__ == "__main__":
    main()
