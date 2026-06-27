"""OpenAI-compatible local image-generation server."""

from __future__ import annotations

import base64
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from diffusion.core import generate as generate_module
from diffusion.core import registry
from diffusion.core.models import DeviceInfo, ModelFamily, Task
from diffusion.utils.errors import DiffusionError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from PIL.Image import Image


class ImageGenerationRequest(BaseModel):
    """Request body for OpenAI-compatible image generation."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    size: str = "512x512"
    response_format: str = "b64_json"
    n: int = Field(default=1, ge=1)
    quality: str | None = None
    style: str | None = None
    user: str | None = None
    steps: int = Field(default=25, ge=1)
    seed: int | None = None
    negative_prompt: str | None = None
    guidance_scale: float | None = None


class _OpenAIAPIError(Exception):
    """Expected API error rendered with OpenAI's top-level ``error`` shape."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        type_: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.type = type_
        self.param = param
        self.code = code


@dataclass(frozen=True)
class _ServerConfig:
    model_id: str
    repo_id: str
    device: str | None
    dtype: str | None
    low_mem: bool
    force_size: bool
    sampler: str | None


@dataclass
class _ServerState:
    config: _ServerConfig
    started_at: int = field(default_factory=lambda: int(time.time()))
    lock: Any = field(default_factory=threading.Lock)
    pipe: Any | None = None
    family: ModelFamily | None = None
    plan: DeviceInfo | None = None

    def load(self) -> None:
        """Load and optimize the configured text-to-image pipeline once."""
        self.pipe, self.family, self.plan = generate_module.load_pipeline(
            self.config.repo_id,
            task=Task.TEXT2IMG,
            controlnet_repo=None,
            device_override=self.config.device,
            dtype_override=self.config.dtype,
            low_mem=self.config.low_mem,
            sampler=self.config.sampler,
        )

    @property
    def ready(self) -> bool:
        """Return whether the pipeline is available for requests."""
        return self.pipe is not None and self.family is not None and self.plan is not None

    def generate(self, request: ImageGenerationRequest) -> Image:
        """Run one serialized image-generation request."""
        if request.model != self.config.model_id:
            raise _OpenAIAPIError(
                f"Model '{request.model}' is not served by this process.",
                param="model",
            )
        if request.response_format != "b64_json":
            raise _OpenAIAPIError(
                "Only response_format='b64_json' is supported.",
                param="response_format",
            )
        if request.n != 1:
            raise _OpenAIAPIError("Only n=1 is supported.", param="n")

        width, height = _parse_size(request.size)
        pipe = self.pipe
        family = self.family
        plan = self.plan
        if pipe is None or family is None or plan is None:
            raise _OpenAIAPIError(
                "Model is not loaded yet.",
                status_code=503,
                type_="server_error",
            )

        with self.lock:
            return generate_module.run_inference(
                pipe,
                family,
                plan,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                steps=request.steps,
                width=width,
                height=height,
                seed=request.seed,
                low_mem=self.config.low_mem,
                force_size=self.config.force_size,
                guidance_scale=request.guidance_scale,
            )


def create_app(
    model_id: str,
    *,
    device: str | None = None,
    dtype: str | None = None,
    low_mem: bool = False,
    force_size: bool = False,
    sampler: str | None = None,
) -> FastAPI:
    """Create a FastAPI app that serves one resolved diffusion model."""
    config = _ServerConfig(
        model_id=model_id,
        repo_id=registry.resolve_repo(model_id),
        device=device,
        dtype=dtype,
        low_mem=low_mem,
        force_size=force_size,
        sampler=sampler,
    )
    state = _ServerState(config=config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        state.load()
        yield

    app = FastAPI(title="diffusion", version="0.1.0", lifespan=lifespan)
    app.state.diffusion = state

    @app.exception_handler(_OpenAIAPIError)
    async def api_error_handler(_request: Request, exc: _OpenAIAPIError) -> JSONResponse:
        return _openai_error_response(
            exc.message,
            status_code=exc.status_code,
            type_=exc.type,
            param=exc.param,
            code=exc.code,
        )

    @app.exception_handler(DiffusionError)
    async def diffusion_error_handler(_request: Request, exc: DiffusionError) -> JSONResponse:
        message = exc.message if exc.hint is None else f"{exc.message} {exc.hint}"
        return _openai_error_response(message, status_code=400)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        message, param = _validation_message(exc)
        return _openai_error_response(message, status_code=422, param=param)

    @app.get("/health")
    def health() -> dict[str, object]:
        server = _get_state(app)
        return {
            "status": "ok" if server.ready else "loading",
            "ready": server.ready,
            "model": server.config.model_id,
            "repo": server.config.repo_id,
            "device": server.plan.device if server.plan is not None else None,
            "dtype": server.plan.dtype if server.plan is not None else None,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, object]:
        server = _get_state(app)
        return {
            "object": "list",
            "data": [
                {
                    "id": server.config.model_id,
                    "object": "model",
                    "created": server.started_at,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/images/generations")
    def create_image(request: ImageGenerationRequest) -> dict[str, object]:
        server = _get_state(app)
        image = server.generate(request)
        return {
            "created": int(time.time()),
            "data": [{"b64_json": _image_to_base64_png(image)}],
        }

    return app


def _get_state(app: FastAPI) -> _ServerState:
    """Return the server state attached to the app."""
    return app.state.diffusion


def _openai_error_response(
    message: str,
    *,
    status_code: int,
    type_: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": type_,
                "param": param,
                "code": code,
            }
        },
    )


def _validation_message(exc: RequestValidationError) -> tuple[str, str | None]:
    errors = exc.errors()
    if not errors:
        return "Invalid request body.", None
    first = errors[0]
    loc = first.get("loc", ())
    path = [str(part) for part in loc if part != "body"]
    param = ".".join(path) if path else None
    message = str(first.get("msg", "Invalid request body."))
    if param is not None:
        message = f"{param}: {message}"
    return message, param


def _parse_size(size: str) -> tuple[int, int]:
    parts = size.lower().split("x", maxsplit=1)
    if len(parts) != 2:
        raise _OpenAIAPIError(
            "size must use WIDTHxHEIGHT format, for example 512x512.",
            param="size",
        )
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise _OpenAIAPIError(
            "size must use integer dimensions, for example 512x512.",
            param="size",
        ) from exc
    if width <= 0 or height <= 0:
        raise _OpenAIAPIError("size dimensions must be positive.", param="size")
    return width, height


def _image_to_base64_png(image: Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
