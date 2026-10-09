import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app.config import config
from app.services.tuvi import images


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (768, 1344), (220, 195, 160)).save(buffer, "PNG")
    return buffer.getvalue()


class _FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _image_response():
    part = SimpleNamespace(inline_data=SimpleNamespace(data=_png_bytes()))
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


def test_generate_image_saves_png_and_uses_style(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "tuvi", {"image_model": "gemini-test"})
    models = _FakeModels([RuntimeError("busy"), _image_response()])
    client = SimpleNamespace(models=models)

    output = images.generate_image("a scholar holding a lantern", str(tmp_path / "a.png"), client=client)

    with Image.open(output) as image:
        assert image.size == (768, 1344)
    model, prompt, generation_config = models.calls[-1]
    assert model == "gemini-test"
    assert "a scholar holding a lantern" in prompt
    assert "no text" in prompt.lower()
    assert generation_config.image_config.aspect_ratio == "9:16"


def test_generate_image_raises_when_no_image(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "tuvi", {})
    empty = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))])
    client = SimpleNamespace(models=_FakeModels([empty, empty]))

    with pytest.raises(images.ImageGenerationError):
        images.generate_image("x", str(tmp_path / "a.png"), client=client)


def test_is_enabled_reads_tuvi_then_app_key(monkeypatch):
    monkeypatch.delenv("TUVI_GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(config, "app", {"gemini_api_key": ""})
    monkeypatch.setattr(config, "tuvi", {"image_provider": "gemini", "gemini_api_key": ""})
    assert not images.is_enabled()

    monkeypatch.setattr(config, "app", {"gemini_api_key": "app-key"})
    assert images.is_enabled()

    monkeypatch.setattr(config, "tuvi", {"image_provider": "none", "gemini_api_key": "k"})
    assert not images.is_enabled()


def test_generate_images_collects_failures(tmp_path, monkeypatch):
    def fake_generate(subject, output_path, client=None):
        if subject == "bad":
            raise images.ImageGenerationError("quota")
        return output_path

    monkeypatch.setattr(images, "generate_image", fake_generate)
    errors = images.generate_images([("ok", str(tmp_path / "1.png")), ("bad", str(tmp_path / "2.png"))])

    assert errors == {str(tmp_path / "2.png"): "quota"}
