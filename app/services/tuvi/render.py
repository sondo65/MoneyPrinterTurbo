"""Assemble scene frames, per-scene narration and karaoke captions into the final MP4."""

import glob
import os
import random
import shutil
import subprocess
import wave
from dataclasses import dataclass, field

from loguru import logger
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from app.config import config
from app.utils import utils

FPS = 30
SCENE_GAP = 0.35
TAIL_SECONDS = 1.2
ZOOM_PER_SCENE = 0.05
PAPER_HEX = "0xDDC3A3"
CAPTION_FONTS = ("NotoSerif-Bold.ttf",)


class RenderError(RuntimeError):
    pass


SENTENCE_PAUSES = {".": 0.5, "!": 0.6, "?": 0.65}
DEFAULT_SENTENCE_PAUSE = 0.45
SENTENCE_EDGE_MS = 40
# Each TTS sentence comes out at a slightly different level; matching them removes the
# loud/quiet jumps that give the AI voice away. The gain is capped so a near-silent or
# failed clip is not boosted into noise, and never pushed into clipping.
SENTENCE_TARGET_DBFS = -20.0
SENTENCE_MAX_GAIN_DB = 9.0
SENTENCE_PEAK_CEILING_DBFS = -1.0


@dataclass
class SentenceSpan:
    text: str
    start: float
    end: float


@dataclass
class SceneAudio:
    path: str
    duration: float
    speech_start: float
    speech_end: float
    sentences: list[SentenceSpan] = field(default_factory=list)


def ffmpeg_exe() -> str:
    configured = config.app.get("ffmpeg_path", "")
    if configured and os.path.isfile(configured):
        return configured
    env = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if env and os.path.isfile(env):
        return env
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: list[str], cwd: str | None = None) -> None:
    command = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RenderError(f"ffmpeg failed: {result.stderr.strip()[-800:]}")


def load_audio(path: str) -> AudioSegment:
    """Decode via the bundled ffmpeg into WAV: pydub's mp3 path needs a system ffprobe."""
    wav_path = os.path.splitext(path)[0] + ".decoded.wav"
    if not os.path.isfile(wav_path) or os.path.getmtime(wav_path) < os.path.getmtime(path):
        _run(["-i", path, "-ac", "1", "-ar", "48000", "-sample_fmt", "s16", wav_path])
    with wave.open(wav_path, "rb") as wav:
        return AudioSegment(
            data=wav.readframes(wav.getnframes()),
            sample_width=wav.getsampwidth(),
            frame_rate=wav.getframerate(),
            channels=wav.getnchannels(),
        )


def analyse_scene_audio(path: str) -> SceneAudio:
    audio = load_audio(path)
    duration = len(audio) / 1000
    spans = detect_nonsilent(audio, min_silence_len=120, silence_thresh=audio.dBFS - 18)
    if spans:
        start, end = spans[0][0] / 1000, spans[-1][1] / 1000
    else:
        start, end = 0.0, duration
    return SceneAudio(path=path, duration=duration, speech_start=start, speech_end=end)


def sentence_pause(text: str) -> float:
    return SENTENCE_PAUSES.get(text.rstrip()[-1:], DEFAULT_SENTENCE_PAUSE)


def _trim(audio: AudioSegment) -> AudioSegment:
    spans = detect_nonsilent(audio, min_silence_len=80, silence_thresh=audio.dBFS - 22)
    if not spans:
        return audio
    start = max(0, spans[0][0] - SENTENCE_EDGE_MS)
    end = min(len(audio), spans[-1][1] + SENTENCE_EDGE_MS)
    return audio[start:end]


def _match_level(clip: AudioSegment) -> AudioSegment:
    if clip.dBFS == float("-inf"):
        return clip
    gain = SENTENCE_TARGET_DBFS - clip.dBFS
    gain = min(gain, SENTENCE_PEAK_CEILING_DBFS - clip.max_dBFS)
    return clip.apply_gain(max(-SENTENCE_MAX_GAIN_DB, min(SENTENCE_MAX_GAIN_DB, gain)))


def join_sentences(parts: list[tuple[str, str]], output_path: str) -> SceneAudio:
    """Concatenate per-sentence TTS clips with deliberate pauses between them.

    ``parts`` is ``[(sentence_text, audio_path), ...]``; the TTS engine barely pauses at full
    stops, so each clip is trimmed and followed by a pause chosen from its final punctuation.
    """
    track = AudioSegment.silent(duration=0, frame_rate=48000)
    spans = []
    for index, (text, path) in enumerate(parts):
        clip = _match_level(_trim(load_audio(path)))
        start = len(track) / 1000
        track += clip
        spans.append(SentenceSpan(text=text, start=start, end=len(track) / 1000))
        if index < len(parts) - 1:
            track += AudioSegment.silent(duration=round(sentence_pause(text) * 1000), frame_rate=48000)
    track.export(output_path, format="wav")
    return SceneAudio(
        path=output_path,
        duration=len(track) / 1000,
        speech_start=spans[0].start if spans else 0.0,
        speech_end=spans[-1].end if spans else len(track) / 1000,
        sentences=spans,
    )


def scene_durations(audios: list[SceneAudio]) -> list[float]:
    durations = [audio.duration + SCENE_GAP for audio in audios]
    if durations:
        durations[-1] += TAIL_SECONDS
    return durations


def build_narration(audios: list[SceneAudio], durations: list[float], output_path: str) -> str:
    track = AudioSegment.silent(duration=0, frame_rate=48000)
    for audio, duration in zip(audios, durations):
        segment = load_audio(audio.path)
        pad = max(0, round(duration * 1000) - len(segment))
        track += segment + AudioSegment.silent(duration=pad, frame_rate=48000)
    track.export(output_path, format="wav")
    return output_path


def render_scene_clip(image_path: str, duration: float, output_path: str) -> str:
    frames = max(1, round(duration * FPS))
    zoom = (
        f"zoompan=z='1+{ZOOM_PER_SCENE}*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d=1:s=1080x1920:fps={FPS}"
    )
    fade_out_start = max(0.0, frames / FPS - 0.25)
    video_filter = (
        f"scale=2160:3840,{zoom},"
        f"fade=t=in:st=0:d=0.3:color={PAPER_HEX},"
        f"fade=t=out:st={fade_out_start:.2f}:d=0.25:color={PAPER_HEX},format=yuv420p"
    )
    _run([
        "-loop", "1", "-framerate", str(FPS), "-i", image_path,
        "-vf", video_filter, "-frames:v", str(frames),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an", output_path,
    ])
    return output_path


def pick_bgm() -> str:
    configured = str(config.tuvi.get("bgm_file", "") or "")
    if configured and os.path.isfile(configured):
        return configured
    songs = sorted(glob.glob(os.path.join(utils.song_dir(), "*.mp3")))
    return random.choice(songs) if songs else ""


def render_video(
    work_dir: str,
    card_paths: list[str],
    durations: list[float],
    narration_path: str,
    ass_text: str,
    output_path: str,
    progress=None,
) -> str:
    progress = progress or (lambda message: None)
    clips_dir = os.path.join(work_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    clip_paths = []
    for index, (card, duration) in enumerate(zip(card_paths, durations), start=1):
        progress(f"Đang dựng cảnh {index}/{len(card_paths)}...")
        clip_paths.append(render_scene_clip(card, duration, os.path.join(clips_dir, f"clip-{index:02d}.mp4")))

    with open(os.path.join(work_dir, "clips.txt"), "w", encoding="utf-8") as fp:
        fp.writelines(f"file 'clips/{os.path.basename(path)}'\n" for path in clip_paths)
    with open(os.path.join(work_dir, "captions.ass"), "w", encoding="utf-8") as fp:
        fp.write(ass_text)

    fonts_dir = os.path.join(work_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    for name in CAPTION_FONTS:
        shutil.copy2(os.path.join(utils.font_dir(), name), fonts_dir)

    total = sum(durations)
    bgm = pick_bgm()
    bgm_volume = float(config.tuvi.get("bgm_volume", 0.15))
    inputs = ["-f", "concat", "-safe", "0", "-i", "clips.txt", "-i", os.path.basename(narration_path)]
    if bgm and bgm_volume > 0:
        inputs += ["-stream_loop", "-1", "-i", bgm]
        audio_filter = (
            f"[2:a]volume={bgm_volume},afade=t=out:st={max(0.0, total - 2):.2f}:d=2[bgm];"
            "[1:a][bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
    else:
        audio_filter = "[1:a]anull[aout]"

    progress("Đang ghép chữ, giọng đọc và nhạc nền...")
    _run(
        [
            *inputs,
            "-filter_complex", f"[0:v]subtitles=captions.ass:fontsdir=fonts[vout];{audio_filter}",
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-t", f"{total:.2f}", "-movflags", "+faststart",
            os.path.abspath(output_path),
        ],
        cwd=work_dir,
    )
    logger.success(f"tuvi video rendered: {output_path} ({total:.1f}s)")
    return output_path
