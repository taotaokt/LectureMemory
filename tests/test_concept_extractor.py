"""Tests for deterministic concept extraction and normalization."""

import pytest

from app.concepts import (
    ExtractedConcept,
    RuleBasedConceptExtractor,
    normalize_concept_name,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("  Divide   and Conquer  ", "divide and conquer"),
        ("B–Trees", "b trees"),
        ("PCA", "pca"),
        ("C++ Programming", "c++ programming"),
    ],
)
def test_normalize_concept_name_collapses_display_variants(
    name: str,
    expected: str,
) -> None:
    assert normalize_concept_name(name) == expected


def test_rule_based_extractor_finds_major_lecture_concepts() -> None:
    extractor = RuleBasedConceptExtractor(max_concepts=10)

    concepts = extractor.extract(
        lecture_title="Divide and Conquer",
        slide_texts=(
            "Karatsuba Multiplication\nKaratsuba reduces recursive calls.",
            "Recurrence Relation\nKaratsuba Multiplication uses three subproblems.",
            "Master Theorem",
        ),
        notes=("Review the Recurrence Relation and Master Theorem.",),
    )

    assert [concept.name for concept in concepts] == [
        "Divide and Conquer",
        "Karatsuba Multiplication",
        "Master Theorem",
        "Recurrence Relation",
    ]
    assert concepts[0].confidence == 1.0
    assert all(0 <= concept.confidence <= 1 for concept in concepts)
    assert len({concept.normalized_name for concept in concepts}) == len(concepts)


def test_rule_based_extractor_ignores_generic_title_and_honors_limit() -> None:
    concepts = RuleBasedConceptExtractor(max_concepts=2).extract(
        lecture_title="Lecture 3",
        slide_texts=("Principal Component Analysis\nSupport Vector Machine",),
        notes=(),
    )

    assert len(concepts) == 2
    assert "lecture 3" not in {concept.normalized_name for concept in concepts}


@pytest.mark.parametrize("max_concepts", [0, -1, True])
def test_rule_based_extractor_rejects_invalid_limit(max_concepts: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RuleBasedConceptExtractor(max_concepts=max_concepts)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "confidence", "error_type"),
    [
        ("", 0.5, ValueError),
        ("Concept", -0.1, ValueError),
        ("Concept", 1.1, ValueError),
        ("Concept", float("nan"), ValueError),
        ("Concept", True, TypeError),
    ],
)
def test_extracted_concept_validates_contract(
    name: str,
    confidence: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        ExtractedConcept(name, confidence)  # type: ignore[arg-type]


def test_extractor_rejects_invalid_source_inputs() -> None:
    extractor = RuleBasedConceptExtractor()

    with pytest.raises(ValueError, match="lecture_title"):
        extractor.extract(lecture_title=" ", slide_texts=(), notes=())
    with pytest.raises(TypeError, match="slide_texts"):
        extractor.extract(lecture_title="Valid", slide_texts="not a list", notes=())
    with pytest.raises(TypeError, match="notes"):
        extractor.extract(
            lecture_title="Valid",
            slide_texts=(),
            notes=("valid", 1),  # type: ignore[arg-type]
        )
