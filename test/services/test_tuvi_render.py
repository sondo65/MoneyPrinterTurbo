import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from app.services.tuvi import render


def _tone(dbfs: float, ms: int = 600) -> AudioSegment:
    tone = Sine(220).to_audio_segment(duration=ms).set_frame_rate(48000)
    return tone.apply_gain(dbfs - tone.dBFS)


def test_match_level_pulls_quiet_and_loud_clips_to_target():
    for source in (-28.0, -12.0):
        matched = render._match_level(_tone(source))
        assert matched.dBFS == pytest.approx(render.SENTENCE_TARGET_DBFS, abs=0.3)


def test_match_level_caps_gain_and_leaves_silence_alone():
    barely_there = render._match_level(_tone(-50.0))
    assert barely_there.dBFS == pytest.approx(-50.0 + render.SENTENCE_MAX_GAIN_DB, abs=0.3)

    silence = AudioSegment.silent(duration=300, frame_rate=48000)
    assert render._match_level(silence) is silence


def test_match_level_never_clips_peaky_clips():
    peaky = _tone(-30.0, 300) + AudioSegment.silent(duration=2000, frame_rate=48000)
    peaky = peaky.overlay(_tone(-3.0, 40), position=100)
    assert render._match_level(peaky).max_dBFS <= render.SENTENCE_PEAK_CEILING_DBFS + 0.1


def test_join_sentences_evens_out_levels(tmp_path):
    paths = []
    for index, level in enumerate((-30.0, -14.0)):
        path = tmp_path / f"s{index}.wav"
        _tone(level).export(path, format="wav")
        paths.append(("Câu một." if index == 0 else "Câu hai.", str(path)))

    result = render.join_sentences(paths, str(tmp_path / "scene.wav"))

    track = AudioSegment.from_wav(result.path)
    first = track[int(result.sentences[0].start * 1000) + 100: int(result.sentences[0].end * 1000) - 100]
    second = track[int(result.sentences[1].start * 1000) + 100: int(result.sentences[1].end * 1000) - 100]
    assert abs(first.dBFS - second.dBFS) < 1.0


def test_camera_move_varies_between_scenes_and_pushes_in_on_last(monkeypatch):
    monkeypatch.setitem(render.config.tuvi, "scene_motion", True)
    moves = [render.camera_move(i, 8) for i in range(8)]
    assert all(a != b for a, b in zip(moves, moves[1:]))
    assert moves[-1] == "zoom_in"

    monkeypatch.setitem(render.config.tuvi, "scene_motion", False)
    assert {render.camera_move(i, 8) for i in range(8)} == {"zoom_in"}


def test_camera_expressions_cover_every_move():
    for move in render.CAMERA_MOVES:
        z, x, y = render.camera_expressions(move, 90)
        assert z and x and y
        assert "on/90" in z + x + y


def test_scene_clip_renders_for_each_move(tmp_path):
    from PIL import Image

    image = tmp_path / "card.png"
    Image.new("RGB", (1080, 1920), (200, 170, 140)).save(image)
    for move in sorted(set(render.CAMERA_MOVES)):
        clip = render.render_scene_clip(str(image), 0.5, str(tmp_path / f"{move}.mp4"), move=move)
        assert (tmp_path / f"{move}.mp4").stat().st_size > 0
        assert clip.endswith(f"{move}.mp4")
