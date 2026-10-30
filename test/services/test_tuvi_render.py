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
