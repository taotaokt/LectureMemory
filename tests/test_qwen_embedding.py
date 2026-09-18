"""Unit tests for the optional Qwen3-VL embedding provider."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.embeddings.base import EmbeddingError
from app.embeddings.qwen import (
    DEFAULT_QUERY_INSTRUCTION,
    Qwen3VLEmbeddingProvider,
    QwenDependencyError,
    _resolve_device,
    _resolve_dtype,
)


class FakeModel:
    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def encode(self, inputs: list[str], **kwargs: Any) -> np.ndarray:
        self.calls.append((inputs, kwargs))
        vectors = np.zeros((len(inputs), self.dimension), dtype=np.float32)
        for index, value in enumerate(inputs):
            vectors[index, 0] = len(value) + 1
            vectors[index, 1] = index + 1
        return vectors


class FailingModel:
    def encode(self, inputs: list[str], **kwargs: Any) -> np.ndarray:
        raise RuntimeError("synthetic inference failure")


def install_fake_model(monkeypatch, model: Any, *, device: str = "mps") -> list[int]:
    load_count = [0]

    def build(provider: Qwen3VLEmbeddingProvider) -> Any:
        load_count[0] += 1
        provider._resolved_device = device
        return model

    monkeypatch.setattr(Qwen3VLEmbeddingProvider, "_build_model", build)
    return load_count


def make_torch(*, cuda: bool, mps: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        float16="float16-value",
        bfloat16="bfloat16-value",
        float32="float32-value",
    )


def test_model_loads_lazily_once_and_batches_all_modalities(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = FakeModel()
    load_count = install_fake_model(monkeypatch, model)
    provider = Qwen3VLEmbeddingProvider(dimension=64, batch_size=2)
    image = tmp_path / "slide.png"
    image.write_bytes(b"synthetic image")

    assert provider.is_loaded is False
    assert provider.resolved_device is None

    text_vectors = provider.embed_texts(["convex sets", "Karatsuba"])
    image_vectors = provider.embed_images([image])
    query_vectors = provider.embed_queries(["where was PCA discussed?"])

    assert provider.is_loaded is True
    assert provider.resolved_device == "mps"
    assert load_count == [1]
    assert text_vectors.shape == (2, 64)
    assert image_vectors.shape == (1, 64)
    assert query_vectors.shape == (1, 64)
    assert np.linalg.norm(text_vectors, axis=1) == pytest.approx([1.0, 1.0])
    assert model.calls[0][0] == ["convex sets", "Karatsuba"]
    assert model.calls[0][1]["batch_size"] == 2
    assert model.calls[0][1]["prompt"] is None
    assert model.calls[1][0] == [str(image.resolve())]
    assert model.calls[2][1]["prompt"] == DEFAULT_QUERY_INSTRUCTION


def test_empty_batches_do_not_load_model(monkeypatch) -> None:
    model = FakeModel()
    load_count = install_fake_model(monkeypatch, model)
    provider = Qwen3VLEmbeddingProvider(dimension=64)

    assert provider.embed_texts([]).shape == (0, 64)
    assert provider.embed_images([]).shape == (0, 64)
    assert provider.embed_queries([]).shape == (0, 64)
    assert load_count == [0]
    assert provider.is_loaded is False


def test_single_item_methods_use_model_and_return_normalized_vectors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = FakeModel()
    install_fake_model(monkeypatch, model, device="cpu")
    provider = Qwen3VLEmbeddingProvider(dimension=64)
    image = tmp_path / "slide.png"
    image.write_bytes(b"image")

    vectors = [
        provider.embed_text("lecture note"),
        provider.embed_image(image),
        provider.embed_query("find this concept"),
    ]

    assert all(vector.shape == (64,) for vector in vectors)
    assert all(vector.dtype == np.float32 for vector in vectors)
    assert all(np.linalg.norm(vector) == pytest.approx(1.0) for vector in vectors)
    assert model.calls[-1][1]["prompt"] == DEFAULT_QUERY_INSTRUCTION


def test_inference_errors_include_model_input_kind_and_device(monkeypatch) -> None:
    install_fake_model(monkeypatch, FailingModel(), device="mps")
    provider = Qwen3VLEmbeddingProvider(dimension=64)

    with pytest.raises(
        EmbeddingError,
        match="failed to embed query input on mps: synthetic inference failure",
    ):
        provider.embed_query("find PCA")


def test_missing_optional_dependencies_have_install_instructions(monkeypatch) -> None:
    def missing_import(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("app.embeddings.qwen.import_module", missing_import)
    provider = Qwen3VLEmbeddingProvider()

    with pytest.raises(QwenDependencyError, match=r"pip install -e '\.\[qwen\]'"):
        provider.embed_text("convexity")


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_selection(cuda: bool, mps: bool, expected: str) -> None:
    assert _resolve_device(make_torch(cuda=cuda, mps=mps), "auto") == expected


@pytest.mark.parametrize(("device", "message"), [("cuda", "CUDA"), ("mps", "MPS")])
def test_unavailable_explicit_device_is_rejected(device: str, message: str) -> None:
    with pytest.raises(EmbeddingError, match=message):
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
        ({"max_pixels": 4095}, "max_pixels"),
        ({"dimension": 32}, "dimension"),
        ({"dimension": 4096}, "maximum dimension"),
        ({"query_instruction": "  "}, "query_instruction"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Qwen3VLEmbeddingProvider(**kwargs)


def test_unknown_model_requires_explicit_dimension() -> None:
    with pytest.raises(ValueError, match="dimension is required"):
        Qwen3VLEmbeddingProvider(model_name="local/custom-model")
