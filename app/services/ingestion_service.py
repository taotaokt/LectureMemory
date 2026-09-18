"""Transactional orchestration for ingesting one lecture PDF."""

import hashlib
import logging
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile

from sqlalchemy.orm import Session

from app.config import get_settings
from app.ingestion.document_processor import extract_page_texts
from app.ingestion.pdf_renderer import DEFAULT_DPI, render_pdf_pages
from app.models import Lecture, SlidePage
from app.repositories.errors import LectureNotFoundError
from app.repositories.slide_page_repository import create_slide_page, list_lecture_pages
from app.schemas import SlidePageCreate

logger = logging.getLogger(__name__)


class IngestionStatus(StrEnum):
    """Possible successful ingestion outcomes."""

    INGESTED = "ingested"
    SKIPPED = "skipped"


class DuplicateIngestionError(RuntimeError):
    """Raised when a lecture already contains different or incomplete source data."""


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    """Result of ingesting or safely skipping one lecture PDF."""

    course_id: int
    lecture_id: int
    source_pdf: Path
    total_pages: int
    rendered_pages: int
    created_pages: int
    failed_pages: int
    status: IngestionStatus


def ingest_lecture_pdf(
    session: Session,
    lecture_id: int,
    pdf_path: str | Path,
    *,
    raw_root: str | Path | None = None,
    rendered_root: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> IngestionSummary:
    """Copy, render, extract, and persist one lecture PDF without duplication."""
    lecture = session.get(Lecture, lecture_id)
    if lecture is None:
        raise LectureNotFoundError(f"Lecture {lecture_id} does not exist")

    source_path = Path(pdf_path).expanduser().resolve()
    page_texts = extract_page_texts(source_path)
    raw_directory = _resolve_raw_root(raw_root)
    rendered_directory = _resolve_rendered_root(rendered_root)
    stored_source = (
        raw_directory
        / f"course_{lecture.course_id}"
        / f"lecture_{lecture.id}"
        / "source.pdf"
    )
    expected_render_directory = (
        rendered_directory / f"course_{lecture.course_id}" / f"lecture_{lecture.id}"
    )
    existing_pages = list_lecture_pages(session, lecture.id)

    if _has_existing_ingestion_state(lecture, existing_pages, stored_source, expected_render_directory):
        return _resolve_existing_ingestion(
            lecture=lecture,
            existing_pages=existing_pages,
            incoming_source=source_path,
            stored_source=stored_source,
            expected_render_directory=expected_render_directory,
            expected_page_count=len(page_texts),
        )

    stored_source.parent.mkdir(parents=True, exist_ok=True)
    rendered_paths: list[Path] = []
    source_stored = False

    try:
        _copy_source_atomically(source_path, stored_source)
        source_stored = True
        rendered_paths = render_pdf_pages(
            stored_source,
            course_id=lecture.course_id,
            lecture_id=lecture.id,
            output_root=rendered_directory,
            dpi=dpi,
        )
        if len(rendered_paths) != len(page_texts):
            raise RuntimeError(
                "Rendered page count does not match the validated PDF page count"
            )

        with session.begin_nested():
            for page_number, (image_path, text_content) in enumerate(
                zip(rendered_paths, page_texts, strict=True),
                start=1,
            ):
                create_slide_page(
                    session,
                    lecture.id,
                    SlidePageCreate(
                        page_number=page_number,
                        image_path=str(image_path),
                        text_content=text_content,
                    ),
                )
            lecture.source_pdf = str(stored_source)
            session.flush()
    except Exception:
        _clean_partial_files(stored_source if source_stored else None, rendered_paths)
        raise

    summary = IngestionSummary(
        course_id=lecture.course_id,
        lecture_id=lecture.id,
        source_pdf=stored_source,
        total_pages=len(page_texts),
        rendered_pages=len(rendered_paths),
        created_pages=len(rendered_paths),
        failed_pages=0,
        status=IngestionStatus.INGESTED,
    )
    logger.info(
        "Ingested lecture PDF",
        extra={
            "course_id": summary.course_id,
            "lecture_id": summary.lecture_id,
            "pages": summary.total_pages,
            "source_pdf": str(summary.source_pdf),
        },
    )
    return summary


def _resolve_raw_root(raw_root: str | Path | None) -> Path:
    if raw_root is not None:
        return Path(raw_root).expanduser().resolve()
    return (get_settings().data_dir / "raw").resolve()


def _resolve_rendered_root(rendered_root: str | Path | None) -> Path:
    if rendered_root is not None:
        return Path(rendered_root).expanduser().resolve()
    configured_root = get_settings().rendered_dir
    if configured_root is None:
        raise ValueError("RENDERED_DIR must be configured")
    return configured_root


def _has_existing_ingestion_state(
    lecture: Lecture,
    existing_pages: list[SlidePage],
    stored_source: Path,
    render_directory: Path,
) -> bool:
    return bool(
        lecture.source_pdf
        or existing_pages
        or stored_source.exists()
        or any(render_directory.glob("page_*.png"))
    )


def _resolve_existing_ingestion(
    *,
    lecture: Lecture,
    existing_pages: list[SlidePage],
    incoming_source: Path,
    stored_source: Path,
    expected_render_directory: Path,
    expected_page_count: int,
) -> IngestionSummary:
    expected_numbers = list(range(1, expected_page_count + 1))
    actual_numbers = [page.page_number for page in existing_pages]
    expected_paths = [
        expected_render_directory / f"page_{page_number:04d}.png"
        for page_number in expected_numbers
    ]
    stored_lecture_path = Path(lecture.source_pdf).resolve() if lecture.source_pdf else None

    is_complete_duplicate = (
        stored_lecture_path == stored_source
        and stored_source.is_file()
        and _file_sha256(incoming_source) == _file_sha256(stored_source)
        and actual_numbers == expected_numbers
        and [Path(page.image_path).resolve() for page in existing_pages] == expected_paths
        and all(path.is_file() for path in expected_paths)
    )
    if not is_complete_duplicate:
        raise DuplicateIngestionError(
            f"Lecture {lecture.id} already has different or incomplete ingestion data"
        )

    return IngestionSummary(
        course_id=lecture.course_id,
        lecture_id=lecture.id,
        source_pdf=stored_source,
        total_pages=expected_page_count,
        rendered_pages=len(existing_pages),
        created_pages=0,
        failed_pages=0,
        status=IngestionStatus.SKIPPED,
    )


def _copy_source_atomically(source_path: Path, stored_source: Path) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            prefix=".source-",
            suffix=".pdf",
            dir=stored_source.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        shutil.copy2(source_path, temporary_path)
        temporary_path.replace(stored_source)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_partial_files(stored_source: Path | None, rendered_paths: list[Path]) -> None:
    if stored_source is not None:
        stored_source.unlink(missing_ok=True)
    for rendered_path in rendered_paths:
        rendered_path.unlink(missing_ok=True)
