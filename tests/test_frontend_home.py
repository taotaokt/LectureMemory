"""Streamlit page tests for the Lecture Memory home screen."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.config import get_settings

APP_PATH = Path(__file__).resolve().parents[1] / "frontend/streamlit_app.py"


@pytest.fixture(autouse=True)
def clear_configuration_cache() -> Iterator[None]:
    """Prevent temporary AppTest database settings from leaking to other tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def build_app(monkeypatch, database_path: Path) -> AppTest:
    """Create an isolated AppTest bound to a temporary database."""
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    get_settings.cache_clear()
    return AppTest.from_file(str(APP_PATH), default_timeout=10)


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
    assert app.session_state["selected_course_id"] == 1
    assert any(item.value == "CS344 is selected." for item in app.success)


def test_home_screen_validates_required_course_fields(monkeypatch, tmp_path: Path) -> None:
    app = build_app(monkeypatch, tmp_path / "invalid-home.db").run()

    app.text_input[0].input(" ")
    app.text_input[1].input("CS344")
    app.button[0].click()
    app.run()

    assert not app.exception
    assert app.error[0].value == "Course name and code are required."
