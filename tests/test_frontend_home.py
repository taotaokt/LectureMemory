"""Streamlit page tests for the Lecture Memory home screen."""

import base64
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from app.config import get_settings
from app.database import create_database_engine, create_session_factory, session_scope
from app.models import Lecture
from app.repositories.concept_repository import (
    assign_lecture_concept,
    get_or_create_concept,
)
from app.repositories.note_repository import create_note
from app.repositories.slide_page_repository import create_slide_page
from app.schemas import NoteCreate, SearchResult, SlidePageCreate

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


@pytest.fixture(autouse=True)
def clear_configuration_cache() -> Iterator[None]:
    """Prevent temporary AppTest database settings from leaking to other tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def build_app(monkeypatch, database_path: Path) -> AppTest:
    """Create an isolated AppTest bound to a temporary database."""
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("INDEX_DIR", str(database_path.parent / f"{database_path.stem}-index"))
    get_settings.cache_clear()
    return AppTest.from_file(str(APP_PATH), default_timeout=10)


def create_and_open_course(app: AppTest) -> None:
    """Create the standard UI fixture and navigate to its course workspace."""
    app.text_input[0].input("Algorithms")
    app.text_input[1].input("CS344")
    app.text_area[0].input("Algorithm design techniques")
    app.button[0].click()
    app.run()

    open_buttons = [button for button in app.button if button.label == "Open"]
    assert len(open_buttons) == 1
    open_buttons[0].click()
    app.run()


def create_and_open_lecture(app: AppTest, title: str = "Divide and Conquer") -> None:
    """Create one lecture from the active course and open its detail page."""
    lecture_title = [item for item in app.text_input if item.label == "Lecture title"]
    assert len(lecture_title) == 1
    lecture_title[0].input(title)
    create_buttons = [button for button in app.button if button.label == "Create lecture"]
    assert len(create_buttons) == 1
    create_buttons[0].click()
    app.run()

    open_buttons = [button for button in app.button if button.label == "Open lecture"]
    assert len(open_buttons) == 1
    open_buttons[0].click()
    app.run()


def test_home_screen_shows_empty_state_and_create_form(monkeypatch, tmp_path: Path) -> None:
    app = build_app(monkeypatch, tmp_path / "empty-home.db").run()

    assert not app.exception
    assert app.title[0].value == "Lecture Memory"
    assert app.subheader[0].value == "Courses"
    assert app.expander[0].label == "Create course"
    assert app.info[0].value == "No courses yet. Create your first course to get started."
    assert [item.label for item in app.text_input] == ["Course name", "Course code"]
    assert app.text_area[0].label == "Description (optional)"


def test_home_screen_can_create_and_open_course(monkeypatch, tmp_path: Path) -> None:
    app = build_app(monkeypatch, tmp_path / "interactive-home.db").run()

    app.text_input[0].input("Algorithms")
    app.text_input[1].input("CS344")
    app.text_area[0].input("Algorithm design techniques")
    app.button[0].click()
    app.run()

    assert not app.exception
    assert app.success[0].value == "Created CS344: Algorithms"
    assert any(item.value == "### Algorithms" for item in app.markdown)
    assert any(item.value == "CS344" for item in app.caption)
    assert any(item.value == "0 lectures" for item in app.markdown)

    open_buttons = [button for button in app.button if button.label == "Open"]
    assert len(open_buttons) == 1
    open_buttons[0].click()
    app.run()

    assert not app.exception
    assert app.title[0].value == "Algorithms"
    assert any(item.value == "CS344" for item in app.caption)
    assert [item.label for item in app.expander] == [
        "Create lecture",
        "Upload lecture PDF",
    ]
    assert any(item.label == "Natural-language query" for item in app.text_input)
    assert any(item.label == "Filter lecture list" for item in app.text_input)
    assert any(
        item.value == "No lectures yet. Create the first lecture for this course."
        for item in app.info
    )

    back_button = [button for button in app.button if button.label == "Back to courses"]
    assert len(back_button) == 1
    back_button[0].click()
    app.run()

    assert app.title[0].value == "Lecture Memory"


def test_course_page_can_create_and_filter_lectures(monkeypatch, tmp_path: Path) -> None:
    app = build_app(monkeypatch, tmp_path / "course-workspace.db").run()
    create_and_open_course(app)

    lecture_title = [item for item in app.text_input if item.label == "Lecture title"]
    assert len(lecture_title) == 1
    lecture_title[0].input("Divide and Conquer")
    create_buttons = [button for button in app.button if button.label == "Create lecture"]
    assert len(create_buttons) == 1
    create_buttons[0].click()
    app.run()

    assert not app.exception
    assert app.success[0].value == "Created lecture 1: Divide and Conquer"
    assert any(item.value == "### 1. Divide and Conquer" for item in app.markdown)
    assert any(item.value == "0 slides · 0 notes" for item in app.markdown)
    assert app.file_uploader[0].label == "PDF file"

    search_input = [item for item in app.text_input if item.label == "Filter lecture list"]
    assert len(search_input) == 1
    search_input[0].input("not present")
    app.run()

    assert app.info[0].value == "No lectures match this search."


def test_home_screen_validates_required_course_fields(monkeypatch, tmp_path: Path) -> None:
    app = build_app(monkeypatch, tmp_path / "invalid-home.db").run()

    app.text_input[0].input(" ")
    app.text_input[1].input("CS344")
    app.button[0].click()
    app.run()

    assert not app.exception
    assert app.error[0].value == "Course name and code are required."


def test_lecture_page_can_add_edit_and_navigate_from_notes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app = build_app(monkeypatch, tmp_path / "lecture-notes.db").run()
    create_and_open_course(app)
    create_and_open_lecture(app)

    assert not app.exception
    assert app.title[0].value == "Divide and Conquer"
    assert [item.value for item in app.subheader] == ["Concepts", "Slides", "Notes"]
    assert any(
        item.value == "No slides available. Upload a PDF from the course workspace."
        for item in app.info
    )
    assert any(item.label == "Add note" for item in app.expander)

    new_note = [item for item in app.text_area if item.label == "New note"]
    assert len(new_note) == 1
    new_note[0].input("Remember the recursion-tree intuition")
    add_buttons = [button for button in app.button if button.label == "Add note"]
    assert len(add_buttons) == 1
    add_buttons[0].click()
    app.run()

    assert not app.exception
    assert app.success[0].value == "Added note 1."
    assert any(
        item.value == "Remember the recursion-tree intuition" for item in app.markdown
    )

    note_content = [item for item in app.text_area if item.label == "Note content"]
    assert len(note_content) == 1
    note_content[0].input("Connect the recursion tree to the Master Theorem")
    save_buttons = [button for button in app.button if button.label == "Save changes"]
    assert len(save_buttons) == 1
    save_buttons[0].click()
    app.run()

    assert not app.exception
    assert app.success[0].value == "Updated note 1."
    assert any(
        item.value == "Connect the recursion tree to the Master Theorem"
        for item in app.markdown
    )

    back_buttons = [button for button in app.button if button.label == "Back to course"]
    assert len(back_buttons) == 1
    back_buttons[0].click()
    app.run()
    assert app.title[0].value == "Algorithms"


def test_lecture_page_displays_slide_concept_and_attached_note(
    monkeypatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "lecture-content.db"
    app = build_app(monkeypatch, database_path).run()
    create_and_open_course(app)

    lecture_title = [item for item in app.text_input if item.label == "Lecture title"]
    lecture_title[0].input("Divide and Conquer")
    [button for button in app.button if button.label == "Create lecture"][0].click()
    app.run()

    image_path = tmp_path / "page-1.png"
    image_path.write_bytes(ONE_PIXEL_PNG)
    engine = create_database_engine(database_path)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        lecture = session.scalar(select(Lecture))
        assert lecture is not None
        page = create_slide_page(
            session,
            lecture.id,
            SlidePageCreate(
                page_number=1,
                image_path=str(image_path),
                text_content="Split the problem into smaller subproblems.",
            ),
        )
        create_note(
            session,
            lecture.id,
            NoteCreate(content="Check the base case", page_id=page.id),
        )
        concept, _ = get_or_create_concept(
            session,
            name="Master Theorem",
            normalized_name="master theorem",
        )
        assign_lecture_concept(
            session,
            lecture_id=lecture.id,
            concept_id=concept.id,
            confidence=0.9,
            source="auto_extracted",
        )
    engine.dispose()

    open_buttons = [button for button in app.button if button.label == "Open lecture"]
    open_buttons[0].click()
    app.run()

    assert not app.exception
    assert any("Master Theorem" in item.value for item in app.markdown)
    assert any(
        item.value == "Split the problem into smaller subproblems."
        for item in app.markdown
    )
    assert any(item.value == "Check the base case" for item in app.markdown)
    slide_select = [item for item in app.selectbox if item.label == "Slide"]
    assert len(slide_select) == 1
    assert slide_select[0].value == 1


def test_course_search_displays_ranked_result_and_opens_lecture(
    monkeypatch,
    tmp_path: Path,
) -> None:
    preview_path = tmp_path / "search-preview.png"
    preview_path.write_bytes(ONE_PIXEL_PNG)
    search_calls: list[tuple[int, str, float]] = []

    def fake_search(
        session_factory,
        settings,
        *,
        course_id: int,
        query: str,
        reranker_weight: float,
    ) -> tuple[SearchResult, ...]:
        search_calls.append((course_id, query, reranker_weight))
        return (
            SearchResult(
                result_type="slide",
                entity_id=1,
                rank=1,
                course_id=course_id,
                course_name="Algorithms",
                course_code="CS344",
                lecture_id=1,
                lecture_title="Divide and Conquer",
                lecture_number=1,
                page_number=17,
                preview_path=str(preview_path),
                raw_similarity=0.81234,
                reranker_score=0.93456,
                text_preview="Three recursive calls reduce the multiplication work.",
                related_notes=("Compare with ordinary multiplication.",),
                concepts=("Karatsuba Multiplication", "Master Theorem"),
            ),
        )

    monkeypatch.setattr("frontend.streamlit_app._execute_course_search", fake_search)
    app = build_app(monkeypatch, tmp_path / "search-ui.db").run()
    create_and_open_course(app)

    lecture_title = [item for item in app.text_input if item.label == "Lecture title"]
    lecture_title[0].input("Divide and Conquer")
    [button for button in app.button if button.label == "Create lecture"][0].click()
    app.run()

    search_input = [item for item in app.text_input if item.label == "Natural-language query"]
    assert len(search_input) == 1
    search_input[0].input("why is Karatsuba faster?")
    assert app.toggle[0].label == "Use multimodal reranking"
    assert app.toggle[0].value is False
    app.toggle[0].set_value(True)
    search_buttons = [button for button in app.button if button.label == "Search"]
    assert len(search_buttons) == 1
    search_buttons[0].click()
    app.run()

    assert not app.exception
    assert search_calls == [(1, "why is Karatsuba faster?", 0.6)]
    assert any(item.value == "### 1. Slide" for item in app.markdown)
    assert any(
        item.value == "Three recursive calls reduce the multiplication work."
        for item in app.markdown
    )
    assert any("Compare with ordinary multiplication" in item.value for item in app.markdown)
    assert any("Karatsuba Multiplication" in item.value for item in app.caption)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics == {
        "Embedding similarity": "0.812",
        "Reranker score": "0.935",
    }

    result_buttons = [button for button in app.button if button.label == "Open lecture"]
    assert len(result_buttons) == 2
    result_buttons[0].click()
    app.run()
    assert app.title[0].value == "Divide and Conquer"


def test_course_search_reports_missing_index(monkeypatch, tmp_path: Path) -> None:
    app = build_app(monkeypatch, tmp_path / "missing-search-index.db").run()
    create_and_open_course(app)

    search_input = [item for item in app.text_input if item.label == "Natural-language query"]
    search_input[0].input("convex sets")
    [button for button in app.button if button.label == "Search"][0].click()
    app.run()

    assert not app.exception
    assert app.error[0].value == (
        "No usable search index was found. Generate and save the lecture index first."
    )
