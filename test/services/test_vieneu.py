"""VieNeu-TTS 自托管服务的协议适配回归，不依赖外部服务。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from app.config import config
from app.services import voice


@pytest.fixture
def vieneu_config(monkeypatch):
    settings = {"base_url": "http://localhost:8000/v1/", "api_key": "test-key",
                "model_id": "vieneu-v3-turbo", "voices": []}
    monkeypatch.setattr(config, "vieneu", settings)
    return settings


def _pcm_response(seconds: float, status_code: int = 200):
    samples = int(voice.VIENEU_SAMPLE_RATE * seconds)
    return SimpleNamespace(status_code=status_code, content=b"\x00\x10" * samples, text="")


def test_voice_list_reads_server_catalog(monkeypatch, vieneu_config):
    payload = {"object": "list", "data": [
        {"id": "Mai Anh", "gender": "female"}, {"id": "Minh Quân Pro"}, {"name": "invalid"},
    ]}
    get = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: payload))
    monkeypatch.setattr(voice.requests, "get", get)

    assert voice.get_vieneu_voices() == ["vieneu:Mai Anh", "vieneu:Minh Quân Pro"]
    get.assert_called_once_with("http://localhost:8000/v1/voices",
                                headers={"Authorization": "Bearer test-key"}, timeout=5)


def test_pinned_voices_do_not_request_server(monkeypatch, vieneu_config):
    vieneu_config["voices"] = "Mai Anh, vieneu:Ngọc Huyền"
    get = Mock()
    monkeypatch.setattr(voice.requests, "get", get)

    assert voice.get_vieneu_voices(fallback=False) == ["vieneu:Mai Anh", "vieneu:Ngọc Huyền"]
    get.assert_not_called()


@pytest.mark.parametrize("failure", [
    requests.ConnectionError("offline"),
    SimpleNamespace(status_code=503, json=lambda: {}),
])
def test_unreachable_server_falls_back_to_builtin_presets(monkeypatch, vieneu_config, failure):
    get = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
    monkeypatch.setattr(voice.requests, "get", get)

    assert voice.get_vieneu_voices(fallback=False) == []
    voices = voice.get_vieneu_voices()
    assert voices[0] == f"vieneu:{voice.VIENEU_DEFAULT_VOICE}"
    assert len(voices) == len(voice.VIENEU_PRESET_VOICES)


def test_vieneu_voice_is_routed_and_not_treated_as_edge_tts(monkeypatch):
    calls = []
    monkeypatch.setattr(voice, "vieneu_tts", lambda *args: calls.append(args) or "ok")

    assert not voice.is_azure_v1_voice("vieneu:Mai Anh")
    assert voice.tts("Xin chào", "vieneu:Mai Anh", 1.2, "out.mp3", 0.8) == "ok"
    assert calls == [("Xin chào", "Mai Anh", "out.mp3", 1.2, 0.8)]


def test_synthesis_requests_pcm_and_writes_mp3(monkeypatch, vieneu_config, tmp_path):
    post = Mock(return_value=_pcm_response(1.0))
    monkeypatch.setattr(voice.requests, "post", post)
    output = tmp_path / "nested" / "voice.mp3"

    sub_maker = voice.vieneu_tts("Xin chào các bạn.", "Mai Anh", str(output))

    assert sub_maker is not None
    assert output.read_bytes()[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
    args, kwargs = post.call_args
    assert args == ("http://localhost:8000/v1/audio/speech",)
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"] == {
        "model": "vieneu-v3-turbo", "input": "Xin chào các bạn.", "voice": "Mai Anh",
        "response_format": "pcm", "sample_rate": voice.VIENEU_SAMPLE_RATE,
    }
    assert sub_maker.offset[-1][1] == pytest.approx(1.0 * 10**7, rel=0.1)


def test_sampling_overrides_are_sent_and_invalid_ones_ignored(monkeypatch, vieneu_config, tmp_path):
    vieneu_config.update({"temperature": "0.3", "top_k": 40, "top_p": 0, "repetition_penalty": 5, "bogus": 1})
    post = Mock(return_value=_pcm_response(1.0))
    monkeypatch.setattr(voice.requests, "post", post)

    voice.vieneu_tts("Xin chào các bạn.", "Mai Anh", str(tmp_path / "voice.mp3"))

    payload = post.call_args.kwargs["json"]
    assert payload["temperature"] == 0.3
    assert payload["top_k"] == 40
    assert "top_p" not in payload and "repetition_penalty" not in payload and "bogus" not in payload


@pytest.mark.parametrize("value", ["", None, "abc", float("nan"), -0.1, 2.5])
def test_unusable_temperature_falls_back_to_server_default(monkeypatch, vieneu_config, value):
    vieneu_config["temperature"] = value
    assert voice._vieneu_sampling_options() == {}


def test_voice_rate_is_applied_locally(monkeypatch, vieneu_config, tmp_path):
    monkeypatch.setattr(voice.requests, "post", Mock(return_value=_pcm_response(2.0)))

    sub_maker = voice.vieneu_tts("Xin chào.", "Mai Anh", str(tmp_path / "v.mp3"), voice_rate=2.0)

    assert sub_maker.offset[-1][1] == pytest.approx(1.0 * 10**7, rel=0.1)


def test_missing_section_uses_default_local_server(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "vieneu", {})
    post = Mock(return_value=_pcm_response(0.5))
    monkeypatch.setattr(voice.requests, "post", post)

    assert voice.vieneu_tts("Xin chào.", "Mai Anh", str(tmp_path / "v.mp3")) is not None
    assert post.call_args.args == (f"{voice.VIENEU_DEFAULT_BASE_URL}/audio/speech",)
    assert post.call_args.kwargs["json"]["model"] == voice.VIENEU_DEFAULT_MODEL
    assert "Authorization" not in post.call_args.kwargs["headers"]


def test_unknown_voice_is_not_retried(monkeypatch, vieneu_config, tmp_path):
    post = Mock(return_value=SimpleNamespace(status_code=400, content=b"", text="unknown voice"))
    monkeypatch.setattr(voice.requests, "post", post)
    output = tmp_path / "v.mp3"

    assert voice.vieneu_tts("Xin chào.", "Không có", str(output)) is None
    assert post.call_count == 1
    assert not output.exists()


@pytest.mark.parametrize("text", ["", "   ", "!!! ..."])
def test_unspeakable_text_does_not_make_requests(monkeypatch, vieneu_config, tmp_path, text):
    post = Mock()
    monkeypatch.setattr(voice.requests, "post", post)
    assert voice.vieneu_tts(text, "Mai Anh", str(tmp_path / "v.mp3")) is None
    post.assert_not_called()
