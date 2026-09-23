"""Streamlit entry point for the Lecture Memory interface."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.database import create_database_engine, create_session_factory, init_database
from app.services.course_service import create_course_from_input, list_course_summaries

APP_TITLE = "Lecture Memory"


@st.cache_resource
def _build_session_factory(database_path: str) -> sessionmaker[Session]:
    """Create one reusable database runtime for the configured local path."""
    engine = create_database_engine(Path(database_path))
    init_database(engine)
    return create_session_factory(engine)


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
                    use_container_width=True,
                ):
                    st.session_state["selected_course_id"] = course.id
                    st.query_params["course_id"] = str(course.id)

            if st.session_state.get("selected_course_id") == course.id:
                st.success(f"{course.code} is selected.")


def _render_create_course_form(session_factory: sessionmaker[Session]) -> None:
    with st.form("create-course", clear_on_submit=True):
        name = st.text_input("Course name", placeholder="Design and Analysis of Algorithms")
        code = st.text_input("Course code", placeholder="CS344")
        description = st.text_area(
            "Description (optional)",
            placeholder="What this course covers",
        )
        submitted = st.form_submit_button("Create course", use_container_width=True)

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
    render_home(_build_session_factory(str(settings.database_path)))


if __name__ == "__main__":
    main()
