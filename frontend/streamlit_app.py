"""Streamlit entry point for the Lecture Memory interface."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.database import create_database_engine, create_session_factory, init_database
from app.embeddings import EmbeddingError
from app.ingestion.document_processor import PDFProcessingError
from app.ingestion.pdf_renderer import PDFRenderError
from app.repositories.errors import (
    CourseNotFoundError,
    LectureNotFoundError,
    NoteNotFoundError,
    NotePageMismatchError,
    SlidePageNotFoundError,
)
from app.retrieval import RerankerError, RetrievalConfigurationError, VectorIndexError
from app.schemas import (
    CourseWorkspace,
    LectureSummary,
    LectureWorkspace,
    NoteDisplay,
    SearchResult,
    SlidePageDisplay,
)
from app.services.course_service import create_course_from_input, list_course_summaries
from app.services.ingestion_service import DuplicateIngestionError
from app.services.lecture_detail_service import (
    create_lecture_note,
    get_lecture_workspace,
    update_lecture_note,
)
from app.services.lecture_service import (
    create_course_lecture,
    get_course_workspace,
    ingest_uploaded_lecture_pdf,
)
from app.services.search_runtime_service import (
    SearchRuntime,
    SearchRuntimeUnavailableError,
    build_search_runtime,
    search_course_memory,
)

APP_TITLE = "Lecture Memory"


@st.cache_resource
def _build_session_factory(database_path: str) -> sessionmaker[Session]:
    """Create one reusable database runtime for the configured local path."""
    engine = create_database_engine(Path(database_path))
    init_database(engine)
    return create_session_factory(engine)


@st.cache_resource
def _build_search_runtime(settings_payload: str) -> SearchRuntime:
    """Build and cache heavyweight search resources for one configuration."""
    return build_search_runtime(Settings.model_validate_json(settings_payload))


def render_home(session_factory: sessionmaker[Session]) -> None:
    """Render the Task 7.1 course home screen."""
    st.title(APP_TITLE)
    st.caption("Your searchable memory for lecture slides and personal notes.")
    if notice := st.session_state.pop("home_notice", None):
        st.success(notice)

    with st.expander("Create course"):
        _render_create_course_form(session_factory)

    with session_factory() as session:
        courses = list_course_summaries(session)

    st.subheader("Courses")
    if not courses:
        st.info("No courses yet. Create your first course to get started.")
        return

    for course in courses:
        with st.container(border=True):
            details, action = st.columns([4, 1])
            with details:
                st.markdown(f"### {course.name}")
                st.caption(course.code)
                lecture_label = "lecture" if course.lecture_count == 1 else "lectures"
                st.write(f"{course.lecture_count} {lecture_label}")
                if course.description:
                    st.write(course.description)
            with action:
                if st.button(
                    "Open",
                    key=f"open-course-{course.id}",
                    width="stretch",
                ):
                    st.query_params["course_id"] = str(course.id)
                    st.rerun()


def _render_create_course_form(session_factory: sessionmaker[Session]) -> None:
    with st.form("create-course", clear_on_submit=True):
        name = st.text_input("Course name", placeholder="Design and Analysis of Algorithms")
        code = st.text_input("Course code", placeholder="CS344")
        description = st.text_area(
            "Description (optional)",
            placeholder="What this course covers",
        )
        submitted = st.form_submit_button("Create course", width="stretch")

    if not submitted:
        return

    try:
        with session_factory.begin() as session:
            course = create_course_from_input(
                session,
                name=name,
                code=code,
                description=description,
            )
    except ValidationError:
        st.error("Course name and code are required.")
        return

    st.session_state["home_notice"] = f"Created {course.code}: {course.name}"
    st.rerun()


def render_course_page(
    session_factory: sessionmaker[Session],
    course_id: int,
    settings: Settings,
) -> None:
    """Render the Task 7.2 course workspace."""
    try:
        with session_factory() as session:
            workspace = get_course_workspace(session, course_id)
    except CourseNotFoundError:
        st.error(f"Course {course_id} does not exist.")
        if st.button("Back to courses", key="missing-course-back"):
            _return_home()
        return

    header, navigation = st.columns([5, 1])
    with header:
        st.title(workspace.course.name)
        st.caption(workspace.course.code)
    with navigation:
        if st.button("Back to courses", width="stretch"):
            _return_home()

    if workspace.course.description:
        st.write(workspace.course.description)
    if notice := st.session_state.pop(f"course_notice_{course_id}", None):
        st.success(notice)

    with st.expander("Create lecture"):
        _render_create_lecture_form(session_factory, workspace)
    with st.expander("Upload lecture PDF"):
        _render_pdf_upload_form(session_factory, workspace, settings)

    _render_course_search(session_factory, workspace, settings)
    st.divider()

    search_query = st.text_input(
        "Filter lecture list",
        placeholder="Filter lectures by title or number",
        key=f"lecture-filter-{course_id}",
    )
    lectures = _filter_lectures(workspace.lectures, search_query)

    st.subheader("Lectures")
    if not workspace.lectures:
        st.info("No lectures yet. Create the first lecture for this course.")
        return
    if not lectures:
        st.info("No lectures match this search.")
        return

    for lecture in lectures:
        with st.container(border=True):
            title, status = st.columns([4, 1])
            with title:
                st.markdown(f"### {lecture.lecture_number}. {lecture.title}")
                st.caption(
                    lecture.lecture_date.isoformat()
                    if lecture.lecture_date is not None
                    else "Date not set"
                )
                slide_label = "slide" if lecture.slide_count == 1 else "slides"
                note_label = "note" if lecture.note_count == 1 else "notes"
                st.write(
                    f"{lecture.slide_count} {slide_label} · "
                    f"{lecture.note_count} {note_label}"
                )
            with status:
                st.caption("PDF ready" if lecture.source_pdf else "No PDF uploaded")
                if st.button(
                    "Open lecture",
                    key=f"open-lecture-{lecture.id}",
                    width="stretch",
                ):
                    st.query_params["course_id"] = str(course_id)
                    st.query_params["lecture_id"] = str(lecture.id)
                    st.rerun()


def render_lecture_page(
    session_factory: sessionmaker[Session],
    lecture_id: int,
) -> None:
    """Render the Task 7.3 lecture detail workspace."""
    try:
        with session_factory() as session:
            workspace = get_lecture_workspace(session, lecture_id)
    except LectureNotFoundError:
        st.error(f"Lecture {lecture_id} does not exist.")
        if st.button("Back to courses", key="missing-lecture-back"):
            _return_home()
        return

    heading, navigation = st.columns([5, 1])
    with heading:
        st.title(workspace.lecture.title)
        date_label = (
            workspace.lecture.lecture_date.isoformat()
            if workspace.lecture.lecture_date is not None
            else "Date not set"
        )
        st.caption(
            f"{workspace.course.code} · Lecture "
            f"{workspace.lecture.lecture_number} · {date_label}"
        )
    with navigation:
        if st.button("Back to course", width="stretch"):
            _return_course(workspace.course.id)

    if notice := st.session_state.pop(f"lecture_notice_{lecture_id}", None):
        st.success(notice)

    _render_concepts(workspace)
    _render_slides(workspace)

    st.subheader("Notes")
    with st.expander("Add note"):
        _render_add_note_form(session_factory, workspace)
    _render_notes(session_factory, workspace)


def _render_concepts(workspace: LectureWorkspace) -> None:
    st.subheader("Concepts")
    if not workspace.concepts:
        st.caption("No concepts extracted for this lecture yet.")
        return

    for concept in workspace.concepts:
        source = concept.source.replace("_", " ").title()
        st.markdown(f"- **{concept.name}** · {concept.confidence:.0%} · {source}")


def _render_slides(workspace: LectureWorkspace) -> None:
    st.subheader("Slides")
    if not workspace.slides:
        st.info("No slides available. Upload a PDF from the course workspace.")
        return

    slide_by_id = {slide.id: slide for slide in workspace.slides}
    selected_id = st.selectbox(
        "Slide",
        options=list(slide_by_id),
        format_func=lambda identifier: f"Page {slide_by_id[identifier].page_number}",
        key=f"lecture-slide-{workspace.lecture.id}",
    )
    selected = slide_by_id[selected_id]
    preview, text = st.columns([3, 2])
    with preview:
        image_path = Path(selected.image_path)
        if image_path.is_file():
            st.image(
                str(image_path),
                caption=f"Page {selected.page_number}",
                width="stretch",
            )
        else:
            st.warning(f"Slide preview is missing: {selected.image_path}")
    with text:
        st.markdown(f"#### Page {selected.page_number}")
        if selected.text_content:
            st.write(selected.text_content)
        else:
            st.caption("No text was extracted from this page.")


def _render_add_note_form(
    session_factory: sessionmaker[Session],
    workspace: LectureWorkspace,
) -> None:
    page_by_id = {slide.id: slide for slide in workspace.slides}
    page_options: list[int | None] = [None, *page_by_id]
    with st.form(f"add-note-{workspace.lecture.id}", clear_on_submit=True):
        content = st.text_area(
            "New note",
            placeholder="Write an explanation, reminder, or question",
        )
        page_id = st.selectbox(
            "Attach to slide",
            options=page_options,
            format_func=lambda identifier: _page_option_label(page_by_id, identifier),
        )
        submitted = st.form_submit_button("Add note", width="stretch")

    if not submitted:
        return
    try:
        with session_factory.begin() as session:
            note = create_lecture_note(
                session,
                workspace.lecture.id,
                content=content,
                page_id=page_id,
            )
    except (LectureNotFoundError, SlidePageNotFoundError, NotePageMismatchError, ValidationError) as exc:
        st.error(_validation_message(exc, fallback="A note cannot be empty."))
        return

    st.session_state[f"lecture_notice_{workspace.lecture.id}"] = (
        f"Added note {note.id}."
    )
    st.rerun()


def _render_notes(
    session_factory: sessionmaker[Session],
    workspace: LectureWorkspace,
) -> None:
    if not workspace.notes:
        st.caption("No notes yet. Add the first note for this lecture.")
        return

    page_by_id = {slide.id: slide for slide in workspace.slides}
    for note in workspace.notes:
        with st.container(border=True):
            st.write(note.content)
            association = (
                f"Slide {note.page_number}"
                if note.page_number is not None
                else "General lecture note"
            )
            st.caption(association)
            with st.expander("Edit note"):
                _render_edit_note_form(session_factory, workspace, note, page_by_id)


def _render_edit_note_form(
    session_factory: sessionmaker[Session],
    workspace: LectureWorkspace,
    note: NoteDisplay,
    page_by_id: dict[int, SlidePageDisplay],
) -> None:
    page_options: list[int | None] = [None, *page_by_id]
    selected_index = page_options.index(note.page_id) if note.page_id in page_options else 0
    with st.form(f"edit-note-{note.id}"):
        content = st.text_area("Note content", value=note.content, key=f"note-content-{note.id}")
        page_id = st.selectbox(
            "Attached slide",
            options=page_options,
            index=selected_index,
            format_func=lambda identifier: _page_option_label(page_by_id, identifier),
            key=f"note-page-{note.id}",
        )
        submitted = st.form_submit_button("Save changes", width="stretch")

    if not submitted:
        return
    try:
        with session_factory.begin() as session:
            update_lecture_note(
                session,
                workspace.lecture.id,
                note.id,
                content=content,
                page_id=page_id,
            )
    except (
        LectureNotFoundError,
        NoteNotFoundError,
        SlidePageNotFoundError,
        NotePageMismatchError,
        ValidationError,
    ) as exc:
        st.error(_validation_message(exc, fallback="A note cannot be empty."))
        return

    st.session_state[f"lecture_notice_{workspace.lecture.id}"] = (
        f"Updated note {note.id}."
    )
    st.rerun()


def _page_option_label(
    page_by_id: dict[int, SlidePageDisplay],
    page_id: int | None,
) -> str:
    if page_id is None:
        return "General lecture note"
    page = page_by_id[page_id]
    return f"Slide {page.page_number}"


def _render_create_lecture_form(
    session_factory: sessionmaker[Session],
    workspace: CourseWorkspace,
) -> None:
    lectures = workspace.lectures
    next_number = max((lecture.lecture_number for lecture in lectures), default=0) + 1
    with st.form(f"create-lecture-{workspace.course.id}", clear_on_submit=True):
        title = st.text_input("Lecture title", placeholder="Divide and Conquer")
        lecture_number = st.number_input(
            "Lecture number",
            min_value=1,
            value=next_number,
            step=1,
        )
        lecture_date = st.date_input("Lecture date (optional)", value=None)
        submitted = st.form_submit_button("Create lecture", width="stretch")

    if not submitted:
        return
    try:
        with session_factory.begin() as session:
            lecture = create_course_lecture(
                session,
                workspace.course.id,
                title=title,
                lecture_number=int(lecture_number),
                lecture_date=lecture_date,
            )
    except (CourseNotFoundError, ValidationError) as exc:
        st.error(_validation_message(exc, fallback="A lecture title is required."))
        return

    st.session_state[f"course_notice_{workspace.course.id}"] = (
        f"Created lecture {lecture.lecture_number}: {lecture.title}"
    )
    st.rerun()


def _render_course_search(
    session_factory: sessionmaker[Session],
    workspace: CourseWorkspace,
    settings: Settings,
) -> None:
    course_id = workspace.course.id
    results_key = f"course_search_results_{course_id}"
    query_key = f"course_search_query_{course_id}"
    error_key = f"course_search_error_{course_id}"

    st.subheader("Search course memory")
    with st.form(f"search-course-{course_id}"):
        query = st.text_input(
            "Natural-language query",
            placeholder="Where did we discuss three recursive multiplication calls?",
        )
        submitted = st.form_submit_button("Search", width="stretch")

    if submitted:
        cleaned_query = query.strip()
        st.session_state.pop(results_key, None)
        st.session_state.pop(error_key, None)
        if not cleaned_query:
            st.session_state[error_key] = "Enter a question or description to search."
        else:
            with st.spinner("Searching slides and notes…"):
                try:
                    results = _execute_course_search(
                        session_factory,
                        settings,
                        course_id=course_id,
                        query=cleaned_query,
                    )
                except (
                    SearchRuntimeUnavailableError,
                    EmbeddingError,
                    RerankerError,
                    RetrievalConfigurationError,
                    VectorIndexError,
                    ValueError,
                ) as exc:
                    st.session_state[error_key] = _search_error_message(exc)
                else:
                    st.session_state[results_key] = results
                    st.session_state[query_key] = cleaned_query

    if error := st.session_state.get(error_key):
        st.error(error)

    stored_results = st.session_state.get(results_key)
    if stored_results is None:
        st.caption(
            "Search uses the persisted multimodal index and reranks matches within this course."
        )
        return
    if not stored_results:
        st.info(f"No indexed slides or notes matched “{st.session_state[query_key]}”.")
        return

    st.markdown(f"#### Results for “{st.session_state[query_key]}”")
    for result in stored_results:
        _render_search_result(result)


def _execute_course_search(
    session_factory: sessionmaker[Session],
    settings: Settings,
    *,
    course_id: int,
    query: str,
) -> tuple[SearchResult, ...]:
    runtime = _build_search_runtime(settings.model_dump_json())
    with session_factory() as session:
        return search_course_memory(
            session,
            query,
            course_id=course_id,
            runtime=runtime,
            settings=settings,
        )


def _search_error_message(error: Exception) -> str:
    if isinstance(error, SearchRuntimeUnavailableError):
        return str(error)
    if isinstance(error, (EmbeddingError, RerankerError)):
        return (
            "The search models could not complete this query. Check the optional Qwen "
            "installation, model access, and device configuration."
        )
    return f"Search configuration is invalid: {error}"


def _render_search_result(result: SearchResult) -> None:
    result_label = "Slide" if result.result_type == "slide" else "Note"
    page_label = f" · Page {result.page_number}" if result.page_number is not None else ""
    with st.container(border=True):
        st.markdown(f"### {result.rank}. {result_label}")
        st.caption(
            f"{result.course_name} ({result.course_code}) · Lecture "
            f"{result.lecture_number}: {result.lecture_title}{page_label}"
        )
        preview, details = st.columns([2, 3])
        with preview:
            preview_path = Path(result.preview_path) if result.preview_path else None
            if preview_path is not None and preview_path.is_file():
                st.image(
                    str(preview_path),
                    caption=f"Page {result.page_number}",
                    width="stretch",
                )
            else:
                st.caption("Slide preview is not available for this result.")
        with details:
            similarity, reranker = st.columns(2)
            similarity.metric("Embedding similarity", f"{result.raw_similarity:.3f}")
            reranker.metric(
                "Reranker score",
                f"{result.reranker_score:.3f}"
                if result.reranker_score is not None
                else "Not scored",
            )
            st.markdown("**Text preview**")
            st.write(result.text_preview or "No text preview available.")
            if result.related_notes:
                st.markdown("**Related notes**")
                for note in result.related_notes:
                    st.write(f"- {note}")
            if result.concepts:
                st.caption(f"Concepts: {', '.join(result.concepts)}")
            if st.button(
                "Open lecture",
                key=f"open-search-result-{result.result_type}-{result.entity_id}",
            ):
                st.query_params["course_id"] = str(result.course_id)
                st.query_params["lecture_id"] = str(result.lecture_id)
                st.rerun()


def _render_pdf_upload_form(
    session_factory: sessionmaker[Session],
    workspace: CourseWorkspace,
    settings: Settings,
) -> None:
    if not workspace.lectures:
        st.info("Create a lecture before uploading its PDF.")
        return

    lecture_by_id = {lecture.id: lecture for lecture in workspace.lectures}
    with st.form(f"upload-pdf-{workspace.course.id}", clear_on_submit=True):
        lecture_id = st.selectbox(
            "Lecture",
            options=list(lecture_by_id),
            format_func=lambda identifier: (
                f"{lecture_by_id[identifier].lecture_number}. "
                f"{lecture_by_id[identifier].title}"
            ),
        )
        uploaded_file = st.file_uploader("PDF file", type=["pdf"])
        submitted = st.form_submit_button("Upload PDF", width="stretch")

    if not submitted:
        return
    if uploaded_file is None:
        st.error("Choose a PDF file to upload.")
        return
    if settings.rendered_dir is None:
        st.error("RENDERED_DIR is not configured.")
        return

    try:
        with session_factory.begin() as session:
            summary = ingest_uploaded_lecture_pdf(
                session,
                lecture_id,
                filename=uploaded_file.name,
                content=uploaded_file.getvalue(),
                raw_root=settings.data_dir / "raw",
                rendered_root=settings.rendered_dir,
            )
    except (DuplicateIngestionError, PDFProcessingError, PDFRenderError, ValueError, OSError) as exc:
        st.error(f"Could not upload PDF: {exc}")
        return

    st.session_state[f"course_notice_{workspace.course.id}"] = (
        f"Uploaded {summary.total_pages} pages for "
        f"{lecture_by_id[lecture_id].title}."
    )
    st.rerun()


def _filter_lectures(
    lectures: tuple[LectureSummary, ...],
    query: str,
) -> tuple[LectureSummary, ...]:
    cleaned_query = query.strip().casefold()
    if not cleaned_query:
        return lectures
    return tuple(
        lecture
        for lecture in lectures
        if cleaned_query in lecture.title.casefold()
        or cleaned_query in str(lecture.lecture_number)
    )


def _validation_message(error: Exception, *, fallback: str) -> str:
    if isinstance(error, ValidationError):
        return fallback
    return str(error)


def _return_home() -> None:
    st.query_params.clear()
    st.rerun()


def _return_course(course_id: int) -> None:
    st.query_params.clear()
    st.query_params["course_id"] = str(course_id)
    st.rerun()


def _id_from_query(name: str) -> int | None:
    value = st.query_params.get(name)
    if value is None:
        return None
    try:
        identifier = int(value)
    except (TypeError, ValueError):
        return None
    return identifier if identifier > 0 else None


def main() -> None:
    """Configure and run the Streamlit application."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    settings = get_settings()
    if settings.database_path is None:
        st.error("DATABASE_PATH is not configured.")
        return
    session_factory = _build_session_factory(str(settings.database_path))
    lecture_id = _id_from_query("lecture_id")
    course_id = _id_from_query("course_id")
    if lecture_id is not None:
        render_lecture_page(session_factory, lecture_id)
    elif course_id is not None:
        render_course_page(session_factory, course_id, settings)
    else:
        render_home(session_factory)


if __name__ == "__main__":
    main()
