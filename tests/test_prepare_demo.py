"""Tests for the public synthetic demo workspace generator."""

from pathlib import Path

import pymupdf

from scripts.prepare_demo import prepare_demo_workspace


def test_prepare_demo_workspace_is_complete_and_idempotent(tmp_path: Path) -> None:
    first = prepare_demo_workspace(tmp_path / "demo")
    second = prepare_demo_workspace(tmp_path / "demo")

    assert first == second
    assert first.course_id == 1
    assert first.lecture_id == 1
    assert first.slide_count == 3
    assert first.note_count == 2
    assert first.concept_count > 0

    source_pdf = first.data_dir / "source" / "gradient-descent-demo.pdf"
    with pymupdf.open(source_pdf) as document:
        assert len(document) == 3
        assert "theta <- theta" in document[0].get_text()
