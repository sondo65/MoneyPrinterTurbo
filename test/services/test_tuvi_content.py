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
    responses = iter(
        ["DÀN Ý: câu hỏi trung tâm", "xin lỗi", json.dumps(_payload(), ensure_ascii=False), "không phải JSON"]
    )
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return next(responses)

    result = content.generate_content(
        SourceItem(title="Tứ Tượng", text="Tư liệu"), "Thứ Sáu, ngày 09/10/2026", fake_generate
    )

    assert result.title == "Tứ Tượng là gì?"
    assert len(prompts) == 4  # outline, invalid script, valid script, illustrator (unusable reply)
    assert "DÀN Ý" in prompts[0] and "đúng 4 cảnh" not in prompts[0]
    assert "đúng 4 cảnh" in prompts[1] and "DÀN Ý ĐÃ DUYỆT" in prompts[1]
    assert "DÀN Ý: câu hỏi trung tâm" in prompts[2]
    assert "Tư liệu" in prompts[1]
    assert result.outline == "DÀN Ý: câu hỏi trung tâm"
    assert result.to_dict()["outline"] == "DÀN Ý: câu hỏi trung tâm"


def test_generate_content_stops_on_provider_error(tuvi_config):
    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return "Error: deepseek: api_key is not set"

    with pytest.raises(content.ContentError, match="api_key"):
        content.generate_content(SourceItem(title="x", text=""), "hôm nay", fake_generate)
    assert len(calls) == 2  # the outline step is optional; the script step surfaces the error


def test_outline_failure_is_not_fatal(tuvi_config):
    def fake_generate(prompt):
        if "DÀN Ý cho chủ đề" in prompt or "lên DÀN Ý" in prompt:
            raise RuntimeError("rate limited")
        return json.dumps(_payload(), ensure_ascii=False)

    result = content.generate_content(SourceItem(title="Tứ Tượng", text=""), "hôm nay", fake_generate)

    assert result.outline == ""
    assert len(result.scenes) == 4


def test_script_prompt_without_outline_has_no_outline_block():
    item = SourceItem(title="Tứ Tượng", text="x")
    assert "DÀN Ý ĐÃ DUYỆT" not in content.build_prompt(item, "hôm nay", 6, "Tử Vi AI", "Tải app")
    assert "DÀN Ý ĐÃ DUYỆT" in content.build_prompt(item, "hôm nay", 6, "Tử Vi AI", "Tải app", "ý chính")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Mệnh là gốc; Thân là ngọn — hai cung bổ sung.", "Mệnh là gốc, Thân là ngọn, hai cung bổ sung."),
        ("Lưu ý: lúc 6:30 (sáng) hãy dậy, , sớm,.", "Lưu ý, lúc 6:30 sáng hãy dậy, sớm."),
        ("Tâm-an, tâm bình. Vì sao vậy?", "Tâm-an, tâm bình. Vì sao vậy?"),
    ],
)
def test_clean_narration_folds_odd_pause_marks_into_commas(raw, expected):
    assert content._clean_narration(raw) == expected


def test_prompt_asks_for_varied_rhythm_without_unhelpful_marks():
    prompt = content.build_prompt(
        SourceItem(title="Tứ Tượng", text="x"), "30/10/2026", 6, "Tử Vi AI", "Tải app"
    )
    assert "rất ngắn" in prompt and "tối đa 1 lần" in prompt
    assert "không tạo nhịp tốt hơn" in prompt


def test_prompt_prioritises_a_single_through_line_and_depth():
    prompt = content.build_prompt(
        SourceItem(title="Tứ Tượng", text="x"), "30/10/2026", 6, "Tử Vi AI", "Tải app"
    )
    assert "MỘT MẠCH DUY NHẤT" in prompt and "CHIỀU SÂU" in prompt and "ĐỘ CHÍNH XÁC" in prompt
    assert "quá 3 câu liên tiếp" in prompt and "tối đa 3 lần" in prompt


def _scenes_response(count, overrides=None):
    items = [
        {"scene": i, "key_visual": f"vật {i}", "image_prompt": f"a lantern number {i}"} for i in range(1, count + 1)
    ]
    for index, patch in (overrides or {}).items():
        items[index].update(patch)
    return "Đây là kết quả:\n```json\n" + json.dumps(items) + "\n```"


def test_illustrator_prompts_replace_script_prompts_and_cover_every_scene():
    video = content.parse_content(_payload())
    prompt = content.build_illustrator_prompt(video)
    assert all(f"Cảnh {i}:" in prompt for i in range(1, 5))
    assert "lá số" in prompt and "Ảnh không được có chữ" in prompt and "KHÔNG viết câu phủ định" in prompt and "LỚN và rõ" in prompt and "Đừng vẽ điện thoại" in prompt and "không hạn chế số cảnh có người" in prompt and "bóng lưng" not in prompt

    assert content.apply_image_prompts(video, _scenes_response(4))
    assert [s.image_prompt for s in video.scenes] == [f"a lantern number {i}" for i in range(1, 5)]
    assert video.scenes[2].key_visual == "vật 3"


@pytest.mark.parametrize(
    "response",
    [
        "không phải JSON",
        "Error: deepseek: timeout",
        None,
        _scenes_response(3),
        _scenes_response(4, {2: {"image_prompt": ""}}),
        "[1, 2, 3, 4]",
    ],
)
def test_unusable_illustrator_reply_keeps_script_prompts(response):
    video = content.parse_content(_payload())
    before = [s.image_prompt for s in video.scenes]

    assert not content.apply_image_prompts(video, response)
    assert [s.image_prompt for s in video.scenes] == before


def test_generate_content_runs_illustrator_pass_and_survives_its_failure(tuvi_config):
    def fake_generate(prompt):
        if "đạo diễn hình ảnh" in prompt:
            return _scenes_response(4)
        if "lên DÀN Ý" in prompt:
            return "dàn ý"
        return json.dumps(_payload(), ensure_ascii=False)

    result = content.generate_content(SourceItem(title="Tứ Tượng", text=""), "hôm nay", fake_generate)
    assert result.scenes[0].image_prompt == "a lantern number 1"

    def failing(prompt):
        if "đạo diễn hình ảnh" in prompt:
            raise RuntimeError("boom")
        return fake_generate(prompt)

    result = content.generate_content(SourceItem(title="Tứ Tượng", text=""), "hôm nay", failing)
    assert result.scenes[0].image_prompt == "scholar 0"
