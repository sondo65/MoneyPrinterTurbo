"""AI illustrations (Google Gemini image models) for Tử Vi scenes."""

import io
import os
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from PIL import Image

from app.config import config

DEFAULT_MODEL = "gemini-2.5-flash-image"
DEFAULT_STYLE = (
    "Traditional Vietnamese folk illustration in the style of old Vietnamese genre paintings: "
    "delicate hand-drawn ink outlines, soft muted watercolor washes in indigo, rust brown, "
    "faded plum and sage green, Vietnamese people from the early 20th century wearing ao dai, "
    "ao the and khan xep. Minimalist composition, one main subject centered in the middle of "
    "the frame with generous empty space around it. Plain warm beige rice paper (giay do) "
    "background in a warm sand-beige tone, with subtle paper fibre texture, filling the whole canvas. "
    "Calm, poetic, spiritual mood. Absolutely no text, no letters, no calligraphy, no "
    "signature, no seal, no watermark, no border, no frame."
)
MAX_WORKERS = 3
MAX_ATTEMPTS = 2


class ImageGenerationError(RuntimeError):
    pass


def _cfg() -> dict:
    return config.tuvi


def get_api_key() -> str:
    return str(
        os.getenv("TUVI_GEMINI_API_KEY")
        or _cfg().get("gemini_api_key")
        or config.app.get("gemini_api_key")
        or ""
    )


def is_enabled() -> bool:
    return str(_cfg().get("image_provider", "gemini")).lower() == "gemini" and bool(get_api_key())


def build_prompt(subject: str) -> str:
    style = str(_cfg().get("image_style") or DEFAULT_STYLE)
    return f"{style}\n\nScene: {subject.strip()}\n\nVertical 9:16 portrait illustration."


def _extract_image(response) -> Image.Image:
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return Image.open(io.BytesIO(inline.data)).convert("RGB")
    raise ImageGenerationError("Gemini response contains no image")


def generate_image(subject: str, output_path: str, client=None) -> str:
    """Generate one 9:16 illustration for ``subject`` and save it as PNG."""
    from google import genai
    from google.genai import types

    model = str(_cfg().get("image_model") or DEFAULT_MODEL)
    generation_config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="9:16"),
    )
    own_client = client is None
    client = client or genai.Client(api_key=get_api_key())
    last_error: Exception | None = None
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=model, contents=build_prompt(subject), config=generation_config
                )
                image = _extract_image(response)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path, "PNG")
                logger.info(f"gemini image saved: {output_path} ({image.width}x{image.height})")
                return output_path
            except Exception as exc:
                last_error = exc
                logger.warning(f"gemini image attempt {attempt} failed: {exc}")
    finally:
        if own_client:
            getattr(client, "close", lambda: None)()
    raise ImageGenerationError(str(last_error))


def generate_images(jobs: list[tuple[str, str]]) -> dict[str, str]:
    """Generate ``(subject, output_path)`` jobs in parallel; returns path -> error for failures."""
    if not jobs:
        return {}
    errors: dict[str, str] = {}

    def run(job):
        subject, output_path = job
        try:
            generate_image(subject, output_path)
        except Exception as exc:
            errors[output_path] = str(exc)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(run, jobs))
    return errors
