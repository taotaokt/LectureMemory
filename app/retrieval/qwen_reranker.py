"""Lazy Qwen3-VL multimodal reranker adapter."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any

from app.retrieval.reranker import RawRerankerScores, Reranker, RerankerError
from app.schemas import SearchResult

DEFAULT_RERANKER_MODEL_NAME = "Qwen/Qwen3-VL-Reranker-2B"
DEFAULT_RERANKER_INSTRUCTION = (
    "Retrieve the lecture slide image or note most relevant to the user's query."
)
SUPPORTED_DEVICES = {"auto", "cpu", "cuda", "mps"}
SUPPORTED_DTYPES = {"auto", "float16", "bfloat16", "float32"}


class QwenRerankerError(RerankerError):
    """Raised when Qwen reranker loading or inference fails."""


class QwenRerankerDependencyError(QwenRerankerError):
    """Raised when the optional Qwen runtime is unavailable."""


class Qwen3VLReranker(Reranker):
    """Score slide images and note text with Qwen3-VL-Reranker."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
        max_length: int = 10_240,
        min_pixels: int = 4 * 32 * 32,
        max_pixels: int = 512 * 32 * 32,
        instruction: str = DEFAULT_RERANKER_INSTRUCTION,
    ) -> None:
        cleaned_model_name = _validate_text(model_name, field_name="model_name")
        if device not in SUPPORTED_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(SUPPORTED_DEVICES)}; got {device!r}"
            )
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(SUPPORTED_DTYPES)}; got {dtype!r}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if (
            not isinstance(min_pixels, int)
            or isinstance(min_pixels, bool)
            or min_pixels <= 0
        ):
            raise ValueError("min_pixels must be a positive integer")
        if (
            not isinstance(max_pixels, int)
            or isinstance(max_pixels, bool)
            or max_pixels < min_pixels
        ):
            raise ValueError("max_pixels must be an integer greater than or equal to min_pixels")

        self._model_name = cleaned_model_name
        self._requested_device = device
        self._dtype_name = dtype
        self._batch_size = batch_size
        self._max_length = max_length
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels
        self._instruction = _validate_text(instruction, field_name="instruction")
        self._model: Any | None = None
        self._resolved_device: str | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """Return whether the heavyweight model has been initialized."""
        return self._model is not None

    @property
    def resolved_device(self) -> str | None:
        """Return the selected device after loading, or ``None`` before loading."""
        return self._resolved_device

    def _score(
        self,
        query: str,
        candidates: tuple[SearchResult, ...],
    ) -> RawRerankerScores:
        documents = [_candidate_document(candidate) for candidate in candidates]
        model = self._get_model()
        try:
            with self._inference_lock:
                return model.score(
                    query=query,
                    documents=documents,
                    instruction=self._instruction,
                    batch_size=self._batch_size,
                )
        except Exception as exc:
            device = self._resolved_device or self._requested_device
            raise QwenRerankerError(
                f"{self.model_name} failed to rerank {len(candidates)} candidates "
                f"on {device}: {exc}"
            ) from exc

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
        dtype = _resolve_dtype(torch, self._dtype_name, device)
        self._resolved_device = device
        try:
            return _TransformersQwenRerankerRuntime(
                model_name=self.model_name,
                device=device,
                dtype=dtype,
                max_length=self._max_length,
                min_pixels=self._min_pixels,
                max_pixels=self._max_pixels,
                torch=torch,
                model_type=modeling.Qwen3VLForConditionalGeneration,
                processor_type=processing.Qwen3VLProcessor,
                process_vision_info=process_vision_info,
            )
        except Exception as exc:
            raise QwenRerankerError(
                f"Could not load {self.model_name} on {device}. Verify the model path, "
                "available memory, and Hugging Face access. "
                f"Original error: {exc}"
            ) from exc


class _TransformersQwenRerankerRuntime:
    """Minimal Transformers runtime following Qwen's official yes/no scorer."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        dtype: Any,
        max_length: int,
        min_pixels: int,
        max_pixels: int,
        torch: Any,
        model_type: Any,
        processor_type: Any,
        process_vision_info: Any,
    ) -> None:
        self._torch = torch
        self._device = device
        self._max_length = max_length
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels
        self._process_vision_info = process_vision_info

        language_model = model_type.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        language_model.eval()
        self._model = language_model.model
        self._model.eval()
        self._processor = processor_type.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        vocabulary = self._processor.tokenizer.get_vocab()
        yes_id = vocabulary["yes"]
        no_id = vocabulary["no"]
        self._score_weight = (
            language_model.lm_head.weight[yes_id] - language_model.lm_head.weight[no_id]
        ).detach()

    def score(
        self,
        *,
        query: str,
        documents: list[dict[str, str]],
        instruction: str,
        batch_size: int,
    ) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            scores.extend(
                self._score_batch(
                    query=query,
                    documents=batch,
                    instruction=instruction,
                )
            )
        return scores

    def _score_batch(
        self,
        *,
        query: str,
        documents: list[dict[str, str]],
        instruction: str,
    ) -> list[float]:
        conversations = [
            self._format_conversation(
                query=query,
                document=document,
                instruction=instruction,
            )
            for document in documents
        ]
        text = self._processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
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
            max_length=self._max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        model_inputs = {key: value.to(self._device) for key, value in model_inputs.items()}
        with self._torch.no_grad():
            hidden_state = self._model(**model_inputs).last_hidden_state[:, -1]
            logits = hidden_state @ self._score_weight
            probabilities = self._torch.sigmoid(logits)
        return probabilities.float().cpu().tolist()

    def _format_conversation(
        self,
        *,
        query: str,
        document: dict[str, str],
        instruction: str,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"<Instruct>: {instruction}"},
            {"type": "text", "text": f"<Query>: {query}"},
            {"type": "text", "text": "\n<Document>:"},
        ]
        if image := document.get("image"):
            content.append(
                {
                    "type": "image",
                    "image": f"file://{image}",
                    "min_pixels": self._min_pixels,
                    "max_pixels": self._max_pixels,
                }
            )
        if text := document.get("text"):
            content.append({"type": "text", "text": text})
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Judge whether the Document meets the requirements based on "
                            "the Query and the Instruct provided. The answer can only be "
                            '"yes" or "no".'
                        ),
                    }
                ],
            },
            {"role": "user", "content": content},
        ]


def _candidate_document(candidate: SearchResult) -> dict[str, str]:
    document: dict[str, str] = {}
    if candidate.text_preview:
        document["text"] = candidate.text_preview
    if candidate.result_type == "slide" and candidate.preview_path:
        image_path = Path(candidate.preview_path).expanduser().resolve()
        if image_path.is_file():
            document["image"] = str(image_path)
    if not document:
        page_context = (
            f", page {candidate.page_number}" if candidate.page_number is not None else ""
        )
        document["text"] = f"{candidate.course_code}: {candidate.lecture_title}{page_context}"
    return document


def _import_qwen_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        torch = import_module("torch")
        modeling = import_module("transformers.models.qwen3_vl.modeling_qwen3_vl")
        processing = import_module("transformers.models.qwen3_vl.processing_qwen3_vl")
        vision_process = import_module("qwen_vl_utils.vision_process")
    except (ImportError, ModuleNotFoundError) as exc:
        raise QwenRerankerDependencyError(
            "Qwen reranker dependencies are not installed. "
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
        raise QwenRerankerError("CUDA was requested but is not available")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise QwenRerankerError("MPS was requested but is not available")
    return requested_device


def _resolve_dtype(torch: Any, dtype_name: str, device: str) -> Any:
    if dtype_name == "auto":
        dtype_name = "float32" if device == "cpu" else "float16"
    return getattr(torch, dtype_name)


def _validate_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned_value = value.strip()
    if not cleaned_value:
        raise ValueError(f"{field_name} must not be blank")
    return cleaned_value
