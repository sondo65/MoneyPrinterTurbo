import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger

from app.config import config
from app.models.llm_provider import get_llm_provider
from app.services import voice
from app.services.tuvi import captions, cards, content, images, render, sources
from app.utils import utils

_generation_lock = threading.Lock()

WEEKDAYS = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật")


class PipelineError(RuntimeError):
    pass


@dataclass
class DailyVideoResult:
    task_id: str
    video_path: str
    title: str
    post_text: str
    source_title: str
    source_url: str
    source_item: sources.SourceItem


def _now() -> datetime:
    return datetime.now(ZoneInfo(config.tuvi.get("timezone", "Asia/Ho_Chi_Minh")))


def today_label(now: datetime | None = None) -> str:
    now = now or _now()
    return f"{WEEKDAYS[now.weekday()]}, ngày {now.day:02d}/{now.month:02d}/{now.year}"


def _voice_rate() -> float:
    return float(config.ui.get("voice_rate", 1.0) or 1.0)


def resolve_voice_name() -> str:
    return str(config.tuvi.get("voice_name") or config.ui.get("voice_name") or "vieneu:Giọng kể chuyện")


def check_prerequisites(voice_name: str) -> None:
    """Fail fast with readable errors before spending LLM credits."""
    provider = str(config.app.get("llm_provider", "")).lower()
    spec = get_llm_provider(provider)
    if spec is None:
        raise PipelineError(f"LLM provider không hợp lệ: '{provider}'")
    if spec.requires_api_key and not config.app.get(spec.config_key("api_key")):
        raise PipelineError(
            f"Thiếu API key cho LLM '{provider}'. Điền {provider}_api_key trong config.toml."
        )
    if voice_name.startswith("vieneu:"):
        available = voice.get_vieneu_voices(fallback=False)
        if not available:
            raise PipelineError(
                "Không kết nối được VieNeu TTS server "
                f"({config.vieneu.get('base_url', 'http://127.0.0.1:8000/v1')}). Hãy bật server trước."
            )
        if voice_name not in available:
            logger.warning(f"voice {voice_name} is not listed by VieNeu server: {available}")
    if not images.is_enabled():
        raise PipelineError(images.missing_provider_hint())


def split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?…])\s+", text or "")]
    return [sentence for sentence in sentences if any(ch.isalnum() for ch in sentence)]


def synthesize_scenes(video_content: content.VideoContent, work_dir: str, voice_name: str) -> list[render.SceneAudio]:
    """TTS sentence by sentence, then join with explicit pauses so the narrator breathes between sentences."""
    audio_dir = os.path.join(work_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audios = []
    for index, scene in enumerate(video_content.scenes, start=1):
        parts = []
        for number, sentence in enumerate(split_sentences(scene.narration), start=1):
            path = os.path.join(audio_dir, f"scene-{index:02d}-{number:02d}.mp3")
            sub_maker = voice.tts(
                text=sentence,
                voice_name=voice.parse_voice_name(voice_name),
                voice_rate=_voice_rate(),
                voice_file=path,
            )
            if not sub_maker or not os.path.isfile(path):
                raise PipelineError(
                    f"Không tạo được giọng đọc cho cảnh {index}. Kiểm tra VieNeu server và tên giọng."
                )
            parts.append((sentence, path))
        if not parts:
            raise PipelineError(f"Cảnh {index} không có lời đọc.")
        audios.append(render.join_sentences(parts, os.path.join(audio_dir, f"scene-{index:02d}.wav")))
    return audios


def generate_illustrations(video_content: content.VideoContent, work_dir: str) -> dict[int, str]:
    """Draw every illustration scene, or raise so no TTS/render time is spent on a half-drawn video."""
    if not images.is_enabled():
        raise PipelineError(images.missing_provider_hint())
    jobs = {}
    for index, scene in enumerate(video_content.scenes):
        subject = scene.image_prompt or content.DEFAULT_IMAGE_PROMPT
        jobs[index] = (subject, os.path.join(work_dir, "art", f"art-{index + 1:02d}.png"))
    errors = images.generate_images(list(jobs.values()))
    for path, error in errors.items():
        logger.warning(f"illustration failed: {path}: {error}")
    if errors:
        reason = next(iter(errors.values()))
        if "credits are depleted" in reason or "402" in reason:
            reason = "hết credit trả trước của Gemini (nạp thêm tại ai.studio/projects)"
        elif "RESOURCE_EXHAUSTED" in reason or "429" in reason:
            reason = "hết hạn mức Gemini (cần bật billing cho API key)"
        raise PipelineError(f"Không vẽ được {len(errors)}/{len(jobs)} tranh: {reason[:160]}. Đã hủy video.")
    return {index: path for index, (_, path) in jobs.items()}


def build_captions(video_content: content.VideoContent, audios: list[render.SceneAudio], durations: list[float]) -> str:
    scene_words = []
    offset = 0.0
    for scene, audio, duration in zip(video_content.scenes, audios, durations):
        spans = audio.sentences or [render.SentenceSpan(scene.narration, audio.speech_start, audio.speech_end)]
        words = []
        for span in spans:
            words += captions.time_words(captions.tokenize(span.text), offset + span.start, offset + span.end)
        scene_words.append((words, captions.CAPTION_Y))
        offset += duration
    return captions.build_ass(scene_words)


def produce_video(video_content: content.VideoContent, work_dir: str, voice_name: str, progress=None) -> str:
    """Illustrations -> frames -> per-scene narration -> captions -> final MP4 in ``work_dir``."""
    progress = progress or (lambda message: None)
    os.makedirs(work_dir, exist_ok=True)

    progress(f"Đang vẽ tranh minh họa cho {len(video_content.scenes)} cảnh...")
    illustrations = generate_illustrations(video_content, work_dir)
    card_paths = cards.render_cards(video_content, os.path.join(work_dir, "cards"), illustrations)
    brand_path = cards.render_brand_overlay(
        os.path.join(work_dir, "cards", "brand.png"), config.tuvi.get("app_name", "Tử Vi AI")
    )

    progress("Đang đọc lời bằng VieNeu...")
    audios = synthesize_scenes(video_content, work_dir, voice_name)
    durations = render.scene_durations(audios)
    narration_path = render.build_narration(audios, durations, os.path.join(work_dir, "narration.wav"))
    ass_text = build_captions(video_content, audios, durations)

    output_path = os.path.join(work_dir, "final.mp4")
    return render.render_video(
        work_dir, card_paths, durations, narration_path, ass_text, output_path,
        progress=progress, brand_path=brand_path,
    )


def generate_video(item: sources.SourceItem, *, history: sources.History | None = None, progress=None) -> DailyVideoResult:
    """Turn one source item into a finished vertical video."""
    progress = progress or (lambda message: None)
    history = history or sources.History()
    voice_name = resolve_voice_name()

    if not _generation_lock.acquire(blocking=False):
        raise PipelineError("Đang có một video khác được tạo, vui lòng thử lại sau.")
    try:
        check_prerequisites(voice_name)

        progress(f"Đang viết kịch bản cho: {item.title}")
        video_content = content.generate_content(item, today_label())

        task_id = f"tuvi-{_now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        work_dir = utils.task_dir(task_id)
        with open(os.path.join(work_dir, "tuvi_content.json"), "w", encoding="utf-8") as fp:
            json.dump(
                {"source": sources.item_to_dict(item), "content": video_content.to_dict()},
                fp,
                ensure_ascii=False,
                indent=2,
            )
        try:
            video_path = produce_video(video_content, work_dir, voice_name, progress)
        except (render.RenderError, cards.MissingIllustrationError, OSError) as exc:
            raise PipelineError(f"Dựng video thất bại: {exc}") from exc

        history.add(item, video_path)
        return DailyVideoResult(
            task_id=task_id,
            video_path=video_path,
            title=video_content.title,
            post_text=video_content.post_text,
            source_title=item.title,
            source_url=item.url,
            source_item=item,
        )
    finally:
        _generation_lock.release()


def generate_daily_video(topic: str | None = None, progress=None) -> DailyVideoResult:
    """Daily run: pick a fresh item from the configured sources, or use ``topic``."""
    progress = progress or (lambda message: None)
    if topic:
        item = sources.item_from_user_input(topic)
    else:
        progress("Đang tìm chủ đề mới từ các nguồn...")
        item = sources.pick_daily_item()
    return generate_video(item, progress=progress)


if __name__ == "__main__":
    import sys

    outcome = generate_daily_video(" ".join(sys.argv[1:]) or None, progress=logger.info)
    print(outcome.video_path)
    print(outcome.post_text)
