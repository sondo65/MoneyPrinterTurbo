import json

import pytest

from app.config import config
from app.services.tuvi import content
from app.services.tuvi.sources import SourceItem


def _payload(scene_count=4, **overrides):
    data = {
        "title": "Tứ Tượng là gì?",
        "hook": "Bạn có biết bốn thần thú canh giữ trời đất?",
        "scenes": [
            {
                "heading": f"Cảnh {index}",
                "text": "Nội dung hiển thị",
                "narration": f"Lời đọc cảnh {index} #tuvi **đậm**",
                "zodiac": "THIN" if index == 1 else "rong",
                "image_prompt": f"scholar {index}",
            }
            for index in range(scene_count)
        ],
        "cta": "Tải app Tử Vi AI",
        "caption": "Khám phá Tứ Tượng",
        "hashtags": ["tử vi", "#phongthuy", ""],
    }
    data.update(overrides)
    return data


@pytest.fixture
def tuvi_config(monkeypatch):
    monkeypatch.setattr(
        config, "tuvi", {"scene_count": 4, "app_name": "Tử Vi AI", "cta_text": "Tải app Tử Vi AI"}
    )


def test_parse_content_cleans_scenes_and_hashtags():
    result = content.parse_content(_payload())

    assert len(result.scenes) == 4
    assert result.scenes[0].narration == "Lời đọc cảnh 0 đậm"
    assert result.scenes[1].zodiac == "thin"
    assert result.scenes[2].zodiac == ""
    assert result.scenes[3].image_prompt == "scholar 3"
    assert result.hashtags == ["#tuvi", "#tửvi", "#phongthuy"]
    assert result.narration.count("\n") == 3
    assert result.post_text.startswith("Tứ Tượng là gì?")


def test_parse_content_gives_every_scene_an_image_prompt():
    data = _payload()
    data["scenes"][0]["image_prompt"] = ""
    data["scenes"][2]["visual"] = "ring"

    scenes = content.parse_content(data).scenes

    assert scenes[0].image_prompt == content.DEFAULT_IMAGE_PROMPT
    assert scenes[2].image_prompt == "scholar 2"
    assert not hasattr(scenes[2], "visual")


def test_parse_content_rejects_too_few_scenes():
    with pytest.raises(content.ContentError):
        content.parse_content(_payload(scene_count=2))


def test_parse_content_caps_scene_count():
    assert len(content.parse_content(_payload(scene_count=12)).scenes) == content.MAX_SCENES


def test_extract_json_handles_fences_and_prose():
    data = _payload()
    fenced = f"Đây là kịch bản:\n```json\n{json.dumps(data, ensure_ascii=False)}\n```"
    assert content.extract_json(fenced)["title"] == data["title"]
    prose = f"Kết quả: {json.dumps(data, ensure_ascii=False)} hết."
    assert content.extract_json(prose)["title"] == data["title"]
    with pytest.raises(content.ContentError):
        content.extract_json("không có json")


def test_generate_content_retries_invalid_responses(tuvi_config):
    responses = iter(["xin lỗi", json.dumps(_payload(), ensure_ascii=False)])
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return next(responses)

    result = content.generate_content(
        SourceItem(title="Tứ Tượng", text="Tư liệu"), "Thứ Sáu, ngày 09/10/2026", fake_generate
    )

    assert result.title == "Tứ Tượng là gì?"
    assert len(prompts) == 2
    assert "đúng 4 cảnh" in prompts[0]
    assert "Tư liệu" in prompts[0]


def test_generate_content_stops_on_provider_error(tuvi_config):
    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return "Error: deepseek: api_key is not set"

    with pytest.raises(content.ContentError, match="api_key"):
        content.generate_content(SourceItem(title="x", text=""), "hôm nay", fake_generate)
    assert len(calls) == 1
