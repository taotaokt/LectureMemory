"""Create an idempotent public demo workspace with synthetic lecture material."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from sqlalchemy import select

from app.concepts import RuleBasedConceptExtractor
from app.config import PROJECT_ROOT, Settings
from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from app.models import Course, Lecture, Note
from app.repositories.course_repository import create_course
from app.repositories.lecture_repository import create_lecture
from app.repositories.note_repository import create_note
from app.repositories.slide_page_repository import list_lecture_pages
from app.schemas import CourseCreate, LectureCreate, NoteCreate
from app.services.concept_service import extract_lecture_concepts
from app.services.ingestion_service import ingest_lecture_pdf

DEMO_NOTES = (
    (
        "A learning rate that is too large can overshoot the minimum, while a very small "
        "rate converges slowly.",
        1,
    ),
    (
        "L2 regularization discourages large weights and often improves generalization.",
        2,
    ),
)


@dataclass(frozen=True, slots=True)
class DemoPreparationSummary:
    """Stable identifiers and counts for the prepared demo workspace."""

    data_dir: Path
    course_id: int
    lecture_id: int
    slide_count: int
    note_count: int
    concept_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "demo",
        help="Isolated output directory for the public demo workspace.",
    )
    return parser


def prepare_demo_workspace(data_dir: str | Path) -> DemoPreparationSummary:
    """Create or validate the synthetic course, lecture, slides, notes, and concepts."""
    root = Path(data_dir).expanduser().resolve()
    settings = Settings(_env_file=None, DATA_DIR=root)
    source_pdf = root / "source" / "gradient-descent-demo.pdf"
    _write_demo_pdf(source_pdf)

    engine = create_database_engine(settings.database_path)
    init_database(engine)
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as session:
            course = session.scalar(select(Course).where(Course.code == "ML101"))
            if course is None:
                course = create_course(
                    session,
                    CourseCreate(
                        name="Machine Learning Foundations",
                        code="ML101",
                        description=("A synthetic public demo showing multimodal lecture search."),
                    ),
                )

            lecture = session.scalar(
                select(Lecture).where(
                    Lecture.course_id == course.id,
                    Lecture.lecture_number == 1,
                )
            )
            if lecture is None:
                lecture = create_lecture(
                    session,
                    course.id,
                    LectureCreate(
                        title="Gradient Descent and Regularization",
                        lecture_number=1,
                    ),
                )

            ingest_lecture_pdf(
                session,
                lecture.id,
                source_pdf,
                raw_root=root / "raw",
                rendered_root=settings.rendered_dir,
                dpi=120,
            )
            pages = list_lecture_pages(session, lecture.id)
            notes = list(
                session.scalars(
                    select(Note).where(Note.lecture_id == lecture.id).order_by(Note.id)
                ).all()
            )
            existing_note_contents = {note.content for note in notes}
            for content, page_offset in DEMO_NOTES:
                if content in existing_note_contents:
                    continue
                create_note(
                    session,
                    lecture.id,
                    NoteCreate(
                        content=content,
                        page_id=pages[page_offset].id,
                    ),
                )

            concept_summary = extract_lecture_concepts(
                session,
                lecture.id,
                extractor=RuleBasedConceptExtractor(max_concepts=8),
            )
            persisted_notes = tuple(
                session.scalars(select(Note).where(Note.lecture_id == lecture.id)).all()
            )
            return DemoPreparationSummary(
                data_dir=root,
                course_id=course.id,
                lecture_id=lecture.id,
                slide_count=len(pages),
                note_count=len(persisted_notes),
                concept_count=len(concept_summary.extracted_concepts),
            )
    finally:
        engine.dispose()


def _write_demo_pdf(destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    try:
        _add_title_slide(
            document,
            title="Gradient Descent",
            subtitle="Following the slope toward a minimum",
            body=(
                "Update rule: theta <- theta - eta * gradient J(theta)\n\n"
                "The gradient gives the direction of steepest increase.\n"
                "Moving in the opposite direction reduces the objective."
            ),
            accent=(0.13, 0.55, 0.47),
        )
        _add_title_slide(
            document,
            title="Choosing the Learning Rate",
            subtitle="Fast progress without overshooting",
            body=(
                "Too small: stable but slow convergence\n\n"
                "Too large: oscillation or divergence\n\n"
                "A schedule can reduce the learning rate as training approaches the minimum."
            ),
            accent=(0.20, 0.42, 0.72),
        )
        _add_title_slide(
            document,
            title="L2 Regularization",
            subtitle="Prefer simpler parameter values",
            body=(
                "Regularized objective: J(theta) + lambda * ||theta||^2\n\n"
                "The penalty discourages very large weights and can improve "
                "generalization on unseen data."
            ),
            accent=(0.57, 0.32, 0.72),
        )
        temporary = destination.with_suffix(".tmp.pdf")
        document.save(temporary)
        temporary.replace(destination)
    finally:
        document.close()


def _add_title_slide(
    document: pymupdf.Document,
    *,
    title: str,
    subtitle: str,
    body: str,
    accent: tuple[float, float, float],
) -> None:
    page = document.new_page(width=960, height=540)
    page.draw_rect(page.rect, color=(0.95, 0.97, 0.99), fill=(0.95, 0.97, 0.99))
    page.draw_rect(pymupdf.Rect(0, 0, 28, 540), color=accent, fill=accent)
    page.insert_text((72, 104), title, fontsize=34, color=(0.08, 0.12, 0.18))
    page.insert_text((74, 143), subtitle, fontsize=17, color=accent)
    page.draw_line((72, 172), (888, 172), color=(0.78, 0.82, 0.87), width=1)
    page.insert_textbox(
        pymupdf.Rect(74, 210, 860, 440),
        body,
        fontsize=20,
        lineheight=1.35,
        color=(0.15, 0.19, 0.25),
    )


def main() -> int:
    args = build_parser().parse_args()
    summary = prepare_demo_workspace(args.data_dir)
    print(
        f"Prepared demo course {summary.course_id}, lecture {summary.lecture_id}: "
        f"{summary.slide_count} slides, {summary.note_count} notes, "
        f"{summary.concept_count} concepts."
    )
    print(f"Demo data directory: {summary.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
