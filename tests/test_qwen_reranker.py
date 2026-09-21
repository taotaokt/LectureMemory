"""Unit tests for the optional Qwen3-VL reranker adapter."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.retrieval import Qwen3VLReranker, QwenRerankerDependencyError, QwenRerankerError
from app.retrieval.qwen_reranker import (
    DEFAULT_RERANKER_INSTRUCTION,
    DEFAULT_RERANKER_MODEL_NAME,
    _resolve_device,
    _resolve_dtype,
)
from app.schemas import SearchResult


class FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[dict[str, Any]] = []

    def score(self, **kwargs: Any) -> list[float]:
        self.calls.append(kwargs)
        return self.scores


class FailingModel:
    def score(self, **kwargs: Any) -> list[float]:
        raise RuntimeError("synthetic inference failure")


def make_result(
    entity_id: int,
    *,
    result_type: str,
    preview_path: str | None,
    text_preview: str | None,
) -> SearchResult:
    return SearchResult(
        result_type=result_type,
        entity_id=entity_id,
        rank=entity_id,
        course_id=1,
        course_name="Algorithms",
        course_code="CS344",
        lecture_id=3,
        lecture_title="Divide and Conquer",
        lecture_number=3,
        page_number=entity_id if preview_path else None,
        preview_path=preview_path,
        raw_similarity=0.9 - entity_id * 0.1,
        text_preview=text_preview,
    )


def install_fake_model(
    monkeypatch,
    model: Any,
    *,
    device: str = "mps",
) -> list[int]:
    load_count = [0]

    def build(reranker: Qwen3VLReranker) -> Any:
        load_count[0] += 1
        reranker._resolved_device = device
        return model

    monkeypatch.setattr(Qwen3VLReranker, "_build_model", build)
    return load_count


def make_torch(*, cuda: bool, mps: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        float16="float16-value",
        bfloat16="bfloat16-value",
        float32="float32-value",
    )


def test_model_loads_lazily_and_scores_slide_images_and_note_text(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "slide.png"
    image.write_bytes(b"synthetic image")
    model = FakeModel([0.2, 0.9])
    load_count = install_fake_model(monkeypatch, model)
    reranker = Qwen3VLReranker(batch_size=2)
    candidates = (
        make_result(
            1,
            result_type="slide",
            preview_path=str(image),
            text_preview="Three recursive calls",
        ),
        make_result(
            2,
            result_type="note",
            preview_path=str(image),
            text_preview="Karatsuba note",
        ),
    )

    assert reranker.is_loaded is False
    assert reranker.resolved_device is None

    results = reranker.rerank("why is Karatsuba faster?", candidates)

    assert reranker.is_loaded is True
    assert reranker.resolved_device == "mps"
    assert load_count == [1]
    assert [result.entity_id for result in results] == [2, 1]
    assert model.calls == [
        {
            "query": "why is Karatsuba faster?",
            "documents": [
                {"text": "Three recursive calls", "image": str(image.resolve())},
                {"text": "Karatsuba note"},
            ],
            "instruction": DEFAULT_RERANKER_INSTRUCTION,
            "batch_size": 2,
        }
    ]


def test_model_is_reused_across_calls(monkeypatch) -> None:
    model = FakeModel([0.5])
    load_count = install_fake_model(monkeypatch, model, device="cpu")
    reranker = Qwen3VLReranker()
    candidates = [
        make_result(1, result_type="note", preview_path=None, text_preview="A note")
    ]

    reranker.rerank("first", candidates)
    reranker.rerank("second", candidates)

    assert load_count == [1]
    assert len(model.calls) == 2


def test_empty_candidates_do_not_load_model(monkeypatch) -> None:
    load_count = install_fake_model(monkeypatch, FakeModel([]))
    reranker = Qwen3VLReranker()

    assert reranker.rerank("valid query", []) == ()
    assert reranker.is_loaded is False
    assert load_count == [0]


def test_missing_slide_image_falls_back_to_available_text(monkeypatch, tmp_path: Path) -> None:
    model = FakeModel([0.5])
    install_fake_model(monkeypatch, model)
    reranker = Qwen3VLReranker()
    candidate = make_result(
        1,
        result_type="slide",
        preview_path=str(tmp_path / "missing.png"),
        text_preview="Extracted slide text",
    )

    reranker.rerank("query", [candidate])

    assert model.calls[0]["documents"] == [{"text": "Extracted slide text"}]


def test_candidate_without_media_uses_domain_context_fallback(monkeypatch) -> None:
    model = FakeModel([0.5])
    install_fake_model(monkeypatch, model)
    reranker = Qwen3VLReranker()
    candidate = make_result(
        1,
        result_type="slide",
        preview_path=None,
        text_preview=None,
    )

    reranker.rerank("query", [candidate])

    assert model.calls[0]["documents"] == [
        {"text": "CS344: Divide and Conquer"}
    ]


def test_inference_errors_include_model_candidate_count_and_device(monkeypatch) -> None:
    install_fake_model(monkeypatch, FailingModel(), device="mps")
    reranker = Qwen3VLReranker()
    candidate = make_result(
        1,
        result_type="note",
        preview_path=None,
        text_preview="A note",
    )

    with pytest.raises(
        QwenRerankerError,
        match="failed to rerank 1 candidates on mps: synthetic inference failure",
    ):
        reranker.rerank("query", [candidate])


def test_missing_optional_dependencies_have_install_instructions(monkeypatch) -> None:
    def missing_import(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("app.retrieval.qwen_reranker.import_module", missing_import)
    reranker = Qwen3VLReranker()
    candidate = make_result(
        1,
        result_type="note",
        preview_path=None,
        text_preview="A note",
    )

    with pytest.raises(QwenRerankerDependencyError, match=r"pip install -e '\.\[qwen\]'"):
        reranker.rerank("query", [candidate])


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_selection(cuda: bool, mps: bool, expected: str) -> None:
    assert _resolve_device(make_torch(cuda=cuda, mps=mps), "auto") == expected


@pytest.mark.parametrize(("device", "message"), [("cuda", "CUDA"), ("mps", "MPS")])
def test_unavailable_explicit_device_is_rejected(device: str, message: str) -> None:
    with pytest.raises(QwenRerankerError, match=message):
        _resolve_device(make_torch(cuda=False, mps=False), device)


def test_auto_dtype_uses_float16_for_accelerators_and_float32_for_cpu() -> None:
    torch = make_torch(cuda=False, mps=True)

    assert _resolve_dtype(torch, "auto", "mps") == "float16-value"
    assert _resolve_dtype(torch, "auto", "cuda") == "float16-value"
    assert _resolve_dtype(torch, "auto", "cpu") == "float32-value"
    assert _resolve_dtype(torch, "bfloat16", "cpu") == "bfloat16-value"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_name": "  "}, "model_name"),
        ({"device": "metal"}, "device"),
        ({"dtype": "int8"}, "dtype"),
        ({"batch_size": 0}, "batch_size"),
        ({"max_length": 0}, "max_length"),
        ({"min_pixels": 0}, "min_pixels"),
        ({"min_pixels": 8192, "max_pixels": 4096}, "max_pixels"),
        ({"instruction": "  "}, "instruction"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Qwen3VLReranker(**kwargs)


def test_official_2b_model_is_the_default() -> None:
    assert Qwen3VLReranker().model_name == DEFAULT_RERANKER_MODEL_NAME
