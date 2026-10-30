import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app.config import config
from app.services.tuvi import images


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "_cache_dir", lambda: str(tmp_path / "cache"))


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


def _fake_mflux(monkeypatch, calls, fail=False):
    """Stand in for the mflux CLI: writes a 576x1024 PNG to the --output path."""
    monkeypatch.setattr(images, "local_command", lambda: "/bin/mflux-generate")

    def fake_run(args, **kwargs):
        calls.append(args)
        if fail:
            return SimpleNamespace(returncode=1, stdout="", stderr="out of memory")
        Image.new("RGB", (576, 1024), (200, 180, 150)).save(args[args.index("--output") + 1], "PNG")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(images.subprocess, "run", fake_run)


def test_provider_chain_parses_and_filters(monkeypatch):
    for raw, expected in [
        ("gemini", ["gemini"]),
        ("local, gemini", ["local", "gemini"]),
        ("local+gemini+local", ["local", "gemini"]),
        (["gemini", "bogus"], ["gemini"]),
        ("none", []),
    ]:
        monkeypatch.setattr(config, "tuvi", {"image_provider": raw})
        assert images.provider_chain() == expected


def test_local_provider_enabled_without_gemini_key(monkeypatch):
    monkeypatch.delenv("TUVI_GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(config, "app", {"gemini_api_key": ""})
    monkeypatch.setattr(config, "tuvi", {"image_provider": "local"})
    monkeypatch.setattr(images, "local_command", lambda: None)
    assert not images.is_enabled()
    assert "mflux" in images.missing_provider_hint()

    monkeypatch.setattr(images, "local_command", lambda: "/bin/mflux-generate")
    assert images.is_enabled()


def test_local_image_runs_mflux_and_normalizes(tmp_path, monkeypatch):
    calls = []
    _fake_mflux(monkeypatch, calls)
    monkeypatch.setattr(config, "tuvi", {"image_provider": "local", "local_steps": 3, "local_lora": "style.safetensors"})

    output = images.generate_image("a scholar with a lantern", str(tmp_path / "out" / "a.png"))

    with Image.open(output) as image:
        assert image.size == (576, 1024)
    args = calls[0]
    assert args[args.index("--steps") + 1] == "3"
    assert args[args.index("--width") + 1] == "576"
    assert args[args.index("--lora-paths") + 1] == "style.safetensors"
    assert "a scholar with a lantern" in args[args.index("--prompt") + 1]
    assert images._seed_for("x") == images._seed_for(" x ")


def test_cache_serves_repeat_prompt_without_regenerating(tmp_path, monkeypatch):
    calls = []
    _fake_mflux(monkeypatch, calls)
    monkeypatch.setattr(config, "tuvi", {"image_provider": "local"})

    images.generate_image("same scene", str(tmp_path / "a.png"))
    images.generate_image("same scene", str(tmp_path / "b.png"))

    assert len(calls) == 1
    assert (tmp_path / "b.png").is_file()

    monkeypatch.setattr(config, "tuvi", {"image_provider": "local", "image_cache": False})
    images.generate_image("same scene", str(tmp_path / "c.png"))
    assert len(calls) == 2


def test_local_failure_falls_back_to_gemini(tmp_path, monkeypatch):
    calls = []
    _fake_mflux(monkeypatch, calls, fail=True)
    monkeypatch.setattr(config, "tuvi", {"image_provider": "local,gemini", "gemini_api_key": "k"})
    client = SimpleNamespace(models=_FakeModels([_image_response()]))

    output = images.generate_image("scene", str(tmp_path / "a.png"), client=client)

    assert len(calls) == 1
    with Image.open(output) as image:
        assert image.size == (768, 1344)


def test_all_providers_failing_reports_each_error(tmp_path, monkeypatch):
    _fake_mflux(monkeypatch, [], fail=True)
    monkeypatch.setattr(config, "tuvi", {"image_provider": "local"})

    with pytest.raises(images.ImageGenerationError, match="local: mflux exited 1: out of memory"):
        images.generate_image("scene", str(tmp_path / "a.png"))


def test_local_timeout_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "local_command", lambda: "/bin/mflux-generate")
    monkeypatch.setattr(config, "tuvi", {"image_provider": "local"})

    def boom(args, **kwargs):
        raise images.subprocess.TimeoutExpired(args, 5)

    monkeypatch.setattr(images.subprocess, "run", boom)
    with pytest.raises(images.ImageGenerationError, match="timed out"):
        images.generate_image("scene", str(tmp_path / "a.png"))


def test_local_chain_generates_sequentially(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(config, "tuvi", {"image_provider": "local"})
    monkeypatch.setattr(images, "local_command", lambda: "/bin/mflux-generate")
    monkeypatch.setattr(images, "generate_image", lambda subject, path, client=None: path)
    real_pool = images.ThreadPoolExecutor

    def spy_pool(max_workers):
        seen["workers"] = max_workers
        return real_pool(max_workers=max_workers)

    monkeypatch.setattr(images, "ThreadPoolExecutor", spy_pool)
    images.generate_images([("a", str(tmp_path / "1.png")), ("b", str(tmp_path / "2.png"))])
    assert seen["workers"] == 1


def test_local_prompt_keeps_original_when_everything_is_a_negation(monkeypatch):
    monkeypatch.setattr(config, "tuvi", {})
    assert images.build_local_prompt("no people").startswith("no people.")
