from datetime import datetime

import pytest

from app.config import config
from app.services.tuvi import captions, pipeline, render
from app.services.tuvi.content import DEFAULT_IMAGE_PROMPT, Scene, VideoContent


def _content():
    return VideoContent(
        title="Tứ Tượng",
        hook="",
        scenes=[
            Scene("a", "", "Bạn có biết.", image_prompt="scholar"),
            Scene("b", "", "Vòng tử vi.", image_prompt="lamp"),
            Scene("c", "", "Tải app ngay."),
        ],
        cta="Tải app",
        caption="",
    )


def test_scene_durations_add_gap_and_tail():
    audios = [render.SceneAudio("a", 3.0, 0.1, 2.9), render.SceneAudio("b", 2.0, 0.1, 1.9)]

    assert render.scene_durations(audios) == [
        3.0 + render.SCENE_GAP,
        2.0 + render.SCENE_GAP + render.TAIL_SECONDS,
    ]


def test_build_captions_offsets_scenes_at_centre(monkeypatch):
    content = _content()
    audios = [
        render.SceneAudio("1", 2.0, 0.2, 1.8),
        render.SceneAudio("2", 2.0, 0.1, 1.9),
        render.SceneAudio("3", 2.0, 0.0, 2.0),
    ]
    durations = render.scene_durations(audios)

    ass = pipeline.build_captions(content, audios, durations)
    lines = [line for line in ass.splitlines() if ",Caption," in line]

    assert lines[0].startswith("Dialogue: 1,0:00:00.20,")
    second_scene_start = durations[0] + 0.1
    assert any(line.startswith(f"Dialogue: 1,{captions._ass_time(second_scene_start)},") for line in lines)
    assert all(f"\\pos(540,{captions.CAPTION_Y})" in line for line in lines)


def test_generate_illustrations_covers_every_scene(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.images, "is_enabled", lambda: True)
    seen = []

    def fake_generate_images(jobs):
        seen.extend(jobs)
        return {}

    monkeypatch.setattr(pipeline.images, "generate_images", fake_generate_images)
    result = pipeline.generate_illustrations(_content(), str(tmp_path))

    assert [subject for subject, _ in seen] == ["scholar", "lamp", DEFAULT_IMAGE_PROMPT]
    assert list(result) == [0, 1, 2]


def test_generate_illustrations_requires_key(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.images, "is_enabled", lambda: False)
    with pytest.raises(pipeline.PipelineError, match="Gemini"):
        pipeline.generate_illustrations(_content(), str(tmp_path))


def test_failed_illustration_cancels_video(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.images, "is_enabled", lambda: True)
    monkeypatch.setattr(
        pipeline.images,
        "generate_images",
        lambda jobs: {jobs[0][1]: "402 RESOURCE_EXHAUSTED. Your prepayment credits are depleted."},
    )
    monkeypatch.setattr(pipeline, "synthesize_scenes", lambda *a: pytest.fail("TTS must not run"))

    with pytest.raises(pipeline.PipelineError, match="hết credit"):
        pipeline.produce_video(_content(), str(tmp_path), "vieneu:Giọng kể chuyện")


def test_check_prerequisites_requires_llm_key(monkeypatch):
    monkeypatch.setattr(config, "app", {"llm_provider": "deepseek", "deepseek_api_key": ""})

    with pytest.raises(pipeline.PipelineError, match="deepseek_api_key"):
        pipeline.check_prerequisites("vieneu:Giọng kể chuyện")


def test_check_prerequisites_requires_vieneu_server(monkeypatch):
    monkeypatch.setattr(config, "app", {"llm_provider": "deepseek", "deepseek_api_key": "k"})
    monkeypatch.setattr(pipeline.voice, "get_vieneu_voices", lambda fallback=True: [])

    with pytest.raises(pipeline.PipelineError, match="VieNeu"):
        pipeline.check_prerequisites("vieneu:Giọng kể chuyện")


def test_split_sentences_keeps_punctuation():
    text = "Theo tử vi cổ, cung này quan trọng. Bạn có tin không? Hãy thử!  ."
    assert pipeline.split_sentences(text) == [
        "Theo tử vi cổ, cung này quan trọng.",
        "Bạn có tin không?",
        "Hãy thử!",
    ]


def test_join_sentences_inserts_pauses(tmp_path):
    from pydub.generators import Sine

    parts = []
    for index, text in enumerate(["Câu một.", "Câu hai?", "Câu ba."]):
        path = str(tmp_path / f"s{index}.wav")
        tone = Sine(220).to_audio_segment(duration=800).set_frame_rate(48000).set_channels(1)
        (render.AudioSegment.silent(300, 48000) + tone + render.AudioSegment.silent(300, 48000)).export(path, format="wav")
        parts.append((text, path))

    audio = render.join_sentences(parts, str(tmp_path / "scene.wav"))

    assert [s.text for s in audio.sentences] == ["Câu một.", "Câu hai?", "Câu ba."]
    first, second, third = audio.sentences
    assert second.start - first.end == pytest.approx(render.SENTENCE_PAUSES["."], abs=0.02)
    assert third.start - second.end == pytest.approx(render.SENTENCE_PAUSES["?"], abs=0.02)
    assert first.start == 0
    assert audio.duration == pytest.approx(third.end, abs=0.01)


def test_build_captions_follows_sentence_spans():
    content = _content()
    audios = [
        render.SceneAudio("1", 4.0, 0.0, 4.0, [render.SentenceSpan("Bạn có biết.", 0.0, 1.0),
                                               render.SentenceSpan("Vòng tử vi.", 3.0, 4.0)]),
        render.SceneAudio("2", 1.0, 0.0, 1.0),
        render.SceneAudio("3", 1.0, 0.0, 1.0),
    ]
    ass = pipeline.build_captions(content, audios, render.scene_durations(audios))

    assert "Dialogue: 1,0:00:03.00," in ass


def test_today_label_is_vietnamese():
    assert pipeline.today_label(datetime(2026, 10, 9)) == "Thứ Sáu, ngày 09/10/2026"
