from __future__ import annotations

import io
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from PIL import Image
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_PORT = 8000
DEFAULT_VLLM_VERSION = "nightly"
PROMPT_OCR_DEFAULT = "<image>\nFree OCR."
PROMPT_OCR_TABLE = "<image>\nOCR the table in this image and output in Markdown format."
PROMPT_EXPLAIN = "<image>\nDescribe this image in detail."
ENV_FILE_PATH = Path(__file__).with_name(".env")
ImageBytes = bytes


@dataclass(frozen=True)
class PromptTask:
    prompt: str
    temperature: float


@dataclass(frozen=True)
class Settings:
    model_id: str = DEFAULT_MODEL_ID
    max_tokens: int = DEFAULT_MAX_TOKENS
    port: int = DEFAULT_PORT
    vllm_version: str = DEFAULT_VLLM_VERSION


class TextResponse(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    model: str
    max_tokens: int
    vllm_version: str


def load_env_file(path: Path = ENV_FILE_PATH) -> None:
    """Populate os.environ from a simple .env file if present."""

    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def _parse_int(value: str | None, fallback: int) -> int:
    try:
        return int(value) if value is not None else fallback
    except ValueError:
        return fallback


def load_settings() -> Settings:
    load_env_file()
    return Settings(
        model_id=os.getenv("MODEL_ID", DEFAULT_MODEL_ID),
        max_tokens=_parse_int(os.getenv("MAX_TOKENS"), DEFAULT_MAX_TOKENS),
        port=_parse_int(os.getenv("PORT"), DEFAULT_PORT),
        vllm_version=os.getenv("VLLM_VERSION", DEFAULT_VLLM_VERSION),
    )


SETTINGS = load_settings()


class OCREngineService:
    """Service wrapper around vLLM's AsyncLLMEngine for OCR and vision prompts."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        engine_args = AsyncEngineArgs(
            model=model_id,
            trust_remote_code=True,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            enforce_eager=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.max_tokens = max_tokens

    @staticmethod
    def _prepare_inputs(image_bytes: ImageBytes, prompt_text: str) -> Dict[str, Any]:
        if not image_bytes:
            raise ValueError("Image data is empty.")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ValueError("Failed to decode image bytes.") from exc

        return {"prompt": prompt_text, "multi_modal_data": {"image": image}}

    async def generate(
        self, image_bytes: ImageBytes, prompt: str, temperature: float = 0.0
    ) -> str:
        inputs = self._prepare_inputs(image_bytes, prompt)
        sampling_params = SamplingParams(max_tokens=self.max_tokens, temperature=temperature)

        results = await self.engine.generate(
            prompts=[inputs], sampling_params=sampling_params
        )
        if not results or not results[0].outputs:
            raise RuntimeError("No generation output returned from vLLM.")

        return results[0].outputs[0].text.strip()


def create_app(settings: Settings = SETTINGS) -> FastAPI:
    app = FastAPI(title="DeepSeek-OCR Service", version="1.0.0")
    router = APIRouter()

    tasks = {
        "full": PromptTask(prompt=PROMPT_OCR_DEFAULT, temperature=0.0),
        "table": PromptTask(prompt=PROMPT_OCR_TABLE, temperature=0.0),
        "custom": PromptTask(prompt=PROMPT_OCR_DEFAULT, temperature=0.1),
        "explain": PromptTask(prompt=PROMPT_EXPLAIN, temperature=0.6),
    }

    @lru_cache(maxsize=1)
    def get_ocr_service() -> OCREngineService:
        return OCREngineService(model_id=settings.model_id, max_tokens=settings.max_tokens)

    def _validate_upload(upload: UploadFile) -> None:
        if upload.content_type and not upload.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Uploaded file must be an image.",
            )

    async def _read_image_bytes(upload: UploadFile) -> ImageBytes:
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return data

    async def _run_inference(
        upload: UploadFile, task: PromptTask, service: OCREngineService
    ) -> TextResponse:
        _validate_upload(upload)
        image_bytes = await _read_image_bytes(upload)

        try:
            text = await service.generate(image_bytes, task.prompt, task.temperature)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - unexpected failures
            raise HTTPException(status_code=500, detail="Inference failed.") from exc

        return TextResponse(text=text)

    @router.post("/ocr/full", response_model=TextResponse)
    async def ocr_full(
        file: UploadFile = File(...),
        service: OCREngineService = Depends(get_ocr_service),
    ) -> TextResponse:
        """Run general OCR with a deterministic prompt."""
        return await _run_inference(file, tasks["full"], service=service)

    @router.post("/ocr/table", response_model=TextResponse)
    async def ocr_table(
        file: UploadFile = File(...),
        service: OCREngineService = Depends(get_ocr_service),
    ) -> TextResponse:
        """Run table-aware OCR and return Markdown."""
        return await _run_inference(file, tasks["table"], service=service)

    @router.post("/ocr/custom", response_model=TextResponse)
    async def ocr_custom(
        file: UploadFile = File(...),
        prompt: str = Form(...),
        service: OCREngineService = Depends(get_ocr_service),
    ) -> TextResponse:
        """Run OCR with a user-supplied prompt."""
        custom_task = PromptTask(prompt=prompt, temperature=tasks["custom"].temperature)
        return await _run_inference(file, custom_task, service=service)

    @router.post("/explain", response_model=TextResponse)
    async def explain(
        file: UploadFile = File(...),
        service: OCREngineService = Depends(get_ocr_service),
    ) -> TextResponse:
        """Generate a descriptive explanation of the image."""
        return await _run_inference(file, tasks["explain"], service=service)

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=settings.model_id,
            max_tokens=settings.max_tokens,
            vllm_version=settings.vllm_version,
        )

    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=SETTINGS.port, reload=False)
