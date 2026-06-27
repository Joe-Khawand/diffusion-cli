"""Tests for the OpenAI-compatible image server."""

from __future__ import annotations

import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from diffusion.core import registry
from diffusion.core.models import DeviceInfo, Task
from diffusion.server import create_app


@pytest.fixture
def server(mocker):
    """Build a server app with all heavy inference boundaries mocked."""
    image = Image.new("RGB", (2, 2), color=(20, 40, 60))
    pipe = object()
    family = registry.by_class_name("StableDiffusionXLPipeline")
    assert family is not None
    plan = DeviceInfo(device="cpu", dtype="float32")

    load = mocker.patch("diffusion.core.generate.load_pipeline", return_value=(pipe, family, plan))
    infer = mocker.patch("diffusion.core.generate.run_inference", return_value=image)
    app = create_app("sdxl", device="cpu", dtype=None, low_mem=False, force_size=False)

    return SimpleNamespace(app=app, image=image, load=load, infer=infer, pipe=pipe, family=family)


def test_health_reports_ready_server(server) -> None:
    with TestClient(server.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ready": True,
        "model": "sdxl",
        "repo": "stabilityai/sdxl-turbo",
        "device": "cpu",
        "dtype": "float32",
    }
    server.load.assert_called_once_with(
        "stabilityai/sdxl-turbo",
        task=Task.TEXT2IMG,
        controlnet_repo=None,
        device_override="cpu",
        dtype_override=None,
        low_mem=False,
        sampler=None,
    )


def test_models_returns_served_model_id(server) -> None:
    with TestClient(server.app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert [model["id"] for model in payload["data"]] == ["sdxl"]


def test_images_generations_returns_base64_png(server) -> None:
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/images/generations",
            json={"model": "sdxl", "prompt": "a robot in a forest"},
        )

    assert response.status_code == 200
    data = response.json()["data"]
    raw = base64.b64decode(data[0]["b64_json"])
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"model": "other", "prompt": "x"}, "model"),
        ({"model": "sdxl", "prompt": "x", "response_format": "url"}, "response_format"),
        ({"model": "sdxl", "prompt": "x", "n": 2}, "n"),
    ],
)
def test_images_generations_rejects_unsupported_request_values(server, body, param) -> None:
    with TestClient(server.app) as client:
        response = client.post("/v1/images/generations", json=body)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == param
    server.infer.assert_not_called()


def test_images_generations_returns_openai_shaped_validation_error(server) -> None:
    with TestClient(server.app) as client:
        response = client.post("/v1/images/generations", json={"model": "sdxl"})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "prompt"
    server.infer.assert_not_called()


def test_images_generations_forwards_supported_parameters(server) -> None:
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/images/generations",
            json={
                "model": "sdxl",
                "prompt": "a glass house",
                "size": "640x384",
                "steps": 7,
                "seed": 123,
                "negative_prompt": "blurry",
                "guidance_scale": 6.5,
                "quality": "hd",
                "style": "vivid",
                "user": "local-test",
            },
        )

    assert response.status_code == 200
    server.infer.assert_called_once_with(
        server.pipe,
        server.family,
        DeviceInfo(device="cpu", dtype="float32"),
        prompt="a glass house",
        negative_prompt="blurry",
        steps=7,
        width=640,
        height=384,
        seed=123,
        low_mem=False,
        force_size=False,
        guidance_scale=6.5,
    )


def test_images_generations_serializes_inference(mocker) -> None:
    image = Image.new("RGB", (1, 1), color=(80, 90, 100))
    pipe = object()
    family = registry.by_class_name("StableDiffusionXLPipeline")
    assert family is not None
    plan = DeviceInfo(device="cpu", dtype="float32")
    mocker.patch("diffusion.core.generate.load_pipeline", return_value=(pipe, family, plan))

    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    def slow_inference(*args, **kwargs):
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            return image
        finally:
            with counter_lock:
                active -= 1

    infer = mocker.patch("diffusion.core.generate.run_inference", side_effect=slow_inference)
    app = create_app("sdxl")

    def post_image(client: TestClient):
        return client.post(
            "/v1/images/generations",
            json={"model": "sdxl", "prompt": "a robot"},
        )

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _index: post_image(client), range(2)))

    assert [response.status_code for response in responses] == [200, 200]
    assert infer.call_count == 2
    assert max_active == 1
