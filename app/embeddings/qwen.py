"""Qwen3-VL embedding provider with optional, lazily loaded dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from app.embeddings.base import (
    EmbeddingBatch,
    EmbeddingError,
    EmbeddingProvider,
    InvalidEmbeddingError,
    RawEmbedding,
    _validate_image_path,
    _validate_text,
)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_QUERY_INSTRUCTION = (
    "Retrieve the lecture slide image or note most relevant to the user's query."
)
KNOWN_MODEL_DIMENSIONS = {
    "Qwen/Qwen3-VL-Embedding-2B": 2048,
    "Qwen/Qwen3-VL-Embedding-8B": 4096,
}
SUPPORTED_DEVICES = {"auto", "cpu", "cuda", "mps"}
SUPPORTED_DTYPES = {"auto", "float16", "bfloat16", "float32"}
MINIMUM_DIMENSION = 64


class QwenDependencyError(EmbeddingError):
    """Raised when optional Qwen runtime dependencies are unavailable."""


class Qwen3VLEmbeddingProvider(EmbeddingProvider):
    """Generate shared text, image, and query embeddings with Qwen3-VL.

    The large ML stack and model weights are loaded only on the first inference.
    The provider follows the official Transformers implementation while adding
    explicit Apple MPS support and a stable application-facing contract.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        dtype: str = "auto",
        dimension: int | None = None,
        batch_size: int = 1,
        max_pixels: int = 512 * 32 * 32,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        cleaned_model_name = model_name.strip()
        if not cleaned_model_name:
            raise ValueError("model_name must not be blank")
        if device not in SUPPORTED_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(SUPPORTED_DEVICES)}; got {device!r}"
            )
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(SUPPORTED_DTYPES)}; got {dtype!r}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if (
            not isinstance(max_pixels, int)
            or isinstance(max_pixels, bool)
            or max_pixels < 4 * 32 * 32
        ):
            raise ValueError("max_pixels must be an integer of at least 4096")

        maximum_dimension = KNOWN_MODEL_DIMENSIONS.get(cleaned_model_name)
        if dimension is None:
            if maximum_dimension is None:
                raise ValueError(
                    "dimension is required for an unknown or local Qwen model"
                )
            dimension = maximum_dimension
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < MINIMUM_DIMENSION
        ):
            raise ValueError(f"dimension must be an integer of at least {MINIMUM_DIMENSION}")
        if maximum_dimension is not None and dimension > maximum_dimension:
            raise ValueError(
                f"dimension {dimension} exceeds {cleaned_model_name}'s maximum "
                f"dimension of {maximum_dimension}"
            )

        self._model_name = cleaned_model_name
        self._requested_device = device
        self._dtype_name = dtype
        self._dimension = dimension
        self._batch_size = batch_size
        self._max_pixels = max_pixels
        self._query_instruction = _validate_text(
            query_instruction,
            field_name="query_instruction",
        )
        self._model: Any | None = None
        self._resolved_device: str | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        """Return whether the heavyweight model has been initialized."""
        return self._model is not None

    @property
    def resolved_device(self) -> str | None:
        """Return the selected device after loading, or ``None`` before loading."""
        return self._resolved_device

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        cleaned_texts = [_validate_text(text, field_name="text") for text in texts]
        return self._embed_batch(cleaned_texts, input_kind="text")

    def embed_images(self, image_paths: Sequence[str | Path]) -> EmbeddingBatch:
        resolved_paths = [_validate_image_path(path) for path in image_paths]
        return self._embed_batch(
            [str(path) for path in resolved_paths],
            input_kind="image",
        )

    def embed_queries(self, queries: Sequence[str]) -> EmbeddingBatch:
        """Embed retrieval queries together with the retrieval instruction."""
        cleaned_queries = [_validate_text(query, field_name="query") for query in queries]
        return self._embed_batch(
            cleaned_queries,
            prompt=self._query_instruction,
            input_kind="query",
        )

    def warmup(self) -> None:
        """Load the model and execute a small text inference eagerly."""
        self.embed_query("lecture retrieval warmup")

    def _embed_text(self, text: str) -> RawEmbedding:
        return self._embed_batch([text], input_kind="text")[0]

    def _embed_image(self, image_path: Path) -> RawEmbedding:
        return self._embed_batch([str(image_path)], input_kind="image")[0]

    def _embed_query(self, query: str) -> RawEmbedding:
        return self._embed_batch(
            [query],
            prompt=self._query_instruction,
            input_kind="query",
        )[0]

    def _embed_batch(
        self,
        inputs: Sequence[str],
        *,
        prompt: str | None = None,
        input_kind: str,
    ) -> EmbeddingBatch:
        if not inputs:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = self._get_model()
        try:
            with self._inference_lock:
                raw_batch = model.encode(
                    list(inputs),
                    prompt=prompt,
                    batch_size=self._batch_size,
                    input_kind=input_kind,
                )
        except Exception as exc:
            device = self._resolved_device or self._requested_device
            raise EmbeddingError(
                f"{self.model_name} failed to embed {input_kind} input on {device}: {exc}"
            ) from exc

        array = np.asarray(raw_batch)
        if array.ndim == 1 and len(inputs) == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] != len(inputs):
            raise InvalidEmbeddingError(
                f"{self.model_name} returned batch shape {array.shape}; "
                f"expected ({len(inputs)}, {self.dimension})"
            )
        return self._stack_vectors([self._prepare_vector(vector) for vector in array])

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is None:
                self._model = self._build_model()
        return self._model

    def _build_model(self) -> Any:
        torch, modeling, processing, process_vision_info = _import_qwen_dependencies()
        device = _resolve_device(torch, self._requested_device)
        torch_dtype = _resolve_dtype(torch, self._dtype_name, device)
        self._resolved_device = device

        try:
            return _TransformersQwenRuntime(
                model_name=self.model_name,
                device=device,
                dtype=torch_dtype,
                dimension=self.dimension,
                max_pixels=self._max_pixels,
                torch=torch,
                modeling=modeling,
                processor_type=processing.Qwen3VLProcessor,
                process_vision_info=process_vision_info,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load {self.model_name} on {device}. Verify the model path, "
                "available memory, and Hugging Face access. "
                f"Original error: {exc}"
            ) from exc


class _TransformersQwenRuntime:
    """Small text-and-image runtime based on Qwen's official implementation."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        dtype: Any,
        dimension: int,
        max_pixels: int,
        torch: Any,
        modeling: Any,
        processor_type: Any,
        process_vision_info: Any,
    ) -> None:
        self._torch = torch
        self._device = device
        self._dimension = dimension
        self._max_pixels = max_pixels
        self._process_vision_info = process_vision_info

        model_type = _create_embedding_model_type(modeling)
        self._model = model_type.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        self._model.eval()
        self._processor = processor_type.from_pretrained(
            model_name,
            padding_side="right",
        )

    def encode(
        self,
        inputs: list[str],
        *,
        prompt: str | None,
        batch_size: int,
        input_kind: str,
    ) -> np.ndarray:
        batches = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            batches.append(self._encode_batch(batch, prompt=prompt, input_kind=input_kind))
        return np.concatenate(batches, axis=0)

    def _encode_batch(
        self,
        inputs: list[str],
        *,
        prompt: str | None,
        input_kind: str,
    ) -> np.ndarray:
        instruction = prompt or "Represent the user's input."
        conversations = [
            self._format_conversation(value, input_kind=input_kind, instruction=instruction)
            for value in inputs
        ]
        text = self._processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        images, video_inputs, video_kwargs = self._process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos = None
            video_metadata = None

        model_inputs = self._processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=8192,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        model_inputs = {key: value.to(self._device) for key, value in model_inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**model_inputs)
            embeddings = self._pool_last_token(
                outputs.last_hidden_state,
                model_inputs["attention_mask"],
            )
            embeddings = embeddings[:, : self._dimension]
            embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings.float().cpu().numpy()

    def _format_conversation(
        self,
        value: str,
        *,
        input_kind: str,
        instruction: str,
    ) -> list[dict[str, Any]]:
        if input_kind == "image":
            content = [
                {
                    "type": "image",
                    "image": f"file://{value}",
                    "min_pixels": 4 * 32 * 32,
                    "max_pixels": self._max_pixels,
                }
            ]
        else:
            content = [{"type": "text", "text": value}]
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

    def _pool_last_token(self, hidden_state: Any, attention_mask: Any) -> Any:
        last_positions = attention_mask.shape[1] - attention_mask.flip(dims=[1]).argmax(dim=1) - 1
        rows = self._torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[rows, last_positions]


def _create_embedding_model_type(modeling: Any) -> Any:
    class Qwen3VLForEmbedding(modeling.Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping = {}
        accepts_loss_kwargs = False

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.model = modeling.Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self) -> Any:
            return self.model.get_input_embeddings()

        def set_input_embeddings(self, value: Any) -> None:
            self.model.set_input_embeddings(value)

        def forward(self, **inputs: Any) -> Any:
            return self.model(**inputs)

    return Qwen3VLForEmbedding


def _import_qwen_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        torch = import_module("torch")
        modeling = import_module("transformers.models.qwen3_vl.modeling_qwen3_vl")
        processing = import_module("transformers.models.qwen3_vl.processing_qwen3_vl")
        vision_process = import_module("qwen_vl_utils.vision_process")
    except (ImportError, ModuleNotFoundError) as exc:
        raise QwenDependencyError(
            "Qwen embedding dependencies are not installed. "
            "Install them with: pip install -e '.[qwen]'"
        ) from exc
    return torch, modeling, processing, vision_process.process_vision_info


def _resolve_device(torch: Any, requested_device: str) -> str:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise EmbeddingError("CUDA was requested but is not available")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise EmbeddingError("MPS was requested but is not available")
    return requested_device


def _resolve_dtype(torch: Any, dtype_name: str, device: str) -> Any:
    if dtype_name == "auto":
        dtype_name = "float32" if device == "cpu" else "float16"
    return getattr(torch, dtype_name)
