"""Contract tests for the model-independent embedding provider."""

from pathlib import Path

import numpy as np
import pytest

from app.embeddings.base import EmbeddingProvider, InvalidEmbeddingError, RawEmbedding


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Small deterministic provider used to exercise the base contract."""

    @property
    def model_name(self) -> str:
        return "deterministic-test-model"

    @property
    def dimension(self) -> int:
        return 3

    def _embed_text(self, text: str) -> RawEmbedding:
        return [len(text), 2.0, 1.0]

    def _embed_image(self, image_path: Path) -> RawEmbedding:
        return [image_path.stat().st_size, 1.0, 2.0]

    def _embed_query(self, query: str) -> RawEmbedding:
        return [1.0, len(query), 2.0]


class InvalidVectorProvider(DeterministicEmbeddingProvider):
    def __init__(self, raw_vector: RawEmbedding) -> None:
        self.raw_vector = raw_vector

    def _embed_text(self, text: str) -> RawEmbedding:
        return self.raw_vector


class InvalidDimensionProvider(DeterministicEmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 0


def test_abstract_provider_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        EmbeddingProvider()


@pytest.mark.parametrize("method_name", ["embed_text", "embed_query"])
def test_text_and_query_embeddings_are_float32_and_normalized(method_name: str) -> None:
    provider = DeterministicEmbeddingProvider()

    vector = getattr(provider, method_name)("  convex sets  ")

    assert vector.shape == (3,)
    assert vector.dtype == np.float32
    assert vector.flags.c_contiguous
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_image_embedding_validates_path_and_normalizes(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake image bytes")
    provider = DeterministicEmbeddingProvider()

    vector = provider.embed_image(image_path)

    assert vector.shape == (3,)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_default_batch_methods_stack_vectors(tmp_path: Path) -> None:
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    first_image.write_bytes(b"first")
    second_image.write_bytes(b"second image")
    provider = DeterministicEmbeddingProvider()

    text_batch = provider.embed_texts(["first", "second"])
    image_batch = provider.embed_images([first_image, second_image])

    assert text_batch.shape == (2, 3)
    assert image_batch.shape == (2, 3)
    assert text_batch.dtype == np.float32
    assert image_batch.dtype == np.float32
    assert np.linalg.norm(text_batch, axis=1) == pytest.approx([1.0, 1.0])
    assert np.linalg.norm(image_batch, axis=1) == pytest.approx([1.0, 1.0])


def test_empty_batches_have_stable_shape() -> None:
    provider = DeterministicEmbeddingProvider()

    assert provider.embed_texts([]).shape == (0, 3)
    assert provider.embed_images([]).shape == (0, 3)


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_text_and_queries_are_rejected(value: str) -> None:
    provider = DeterministicEmbeddingProvider()

    with pytest.raises(ValueError, match="must not be blank"):
        provider.embed_text(value)
    with pytest.raises(ValueError, match="must not be blank"):
        provider.embed_query(value)


def test_missing_image_is_rejected(tmp_path: Path) -> None:
    provider = DeterministicEmbeddingProvider()

    with pytest.raises(FileNotFoundError, match="Image does not exist"):
        provider.embed_image(tmp_path / "missing.png")


@pytest.mark.parametrize(
    "raw_vector",
    [
        [1.0, 2.0],
        [0.0, 0.0, 0.0],
        [1.0, np.nan, 2.0],
        [1.0, np.inf, 2.0],
        [[1.0, 2.0, 3.0]],
    ],
)
def test_invalid_provider_vectors_are_rejected(raw_vector: RawEmbedding) -> None:
    provider = InvalidVectorProvider(raw_vector)

    with pytest.raises(InvalidEmbeddingError):
        provider.embed_text("valid input")


def test_invalid_declared_dimension_is_rejected() -> None:
    provider = InvalidDimensionProvider()

    with pytest.raises(InvalidEmbeddingError, match="invalid embedding dimension"):
        provider.embed_text("valid input")
