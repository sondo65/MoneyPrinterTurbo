"""AI illustrations for Tử Vi scenes: local FLUX (mflux) and/or Google Gemini, with an on-disk cache.

``[tuvi] image_provider`` is a provider chain tried left to right, e.g. ``"local,gemini"``:
the free local model first, paid Gemini only when it fails. ``"none"`` disables illustrations.
"""

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from PIL import Image

from app.config import config
from app.utils import utils

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
# FLUX has no negative prompt and tends to draw whatever it is told not to, so the local
# style only says what the picture is. Avoid "folk painting"/"rice paper": FLUX then adds fake
# Chinese calligraphy and red seals. Kept short: schnell's T5 encoder reads 256 tokens.
DEFAULT_LOCAL_STYLE = (
    "Soft watercolor illustration with delicate thin ink outlines, muted palette of indigo, "
    "rust brown, faded plum and sage green, early 20th century Vietnamese village people in "
    "ao dai and khan xep, minimalist composition with one centered subject and generous empty "
    "space, plain warm beige paper background with subtle texture, calm poetic mood, vertical "
    "portrait, clean image containing only the illustration."
)
DEFAULT_LOCAL_MODEL = "dhairyashil/FLUX.1-schnell-mflux-4bit"
DEFAULT_LOCAL_BASE_MODEL = "schnell"
DEFAULT_LOCAL_COMMAND = "mflux-generate"
LOCAL_SIZE = (576, 1024)  # multiples of 16, 9:16; the card renderer upscales to 1080x1920
LOCAL_STEPS = 4
LOCAL_TIMEOUT = 900
GEMINI_WORKERS = 3
MAX_ATTEMPTS = 2
PROVIDERS = ("local", "gemini")


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


def provider_chain() -> list[str]:
    """Providers to try in order; unknown names are ignored, ``none`` yields an empty chain."""
    raw = _cfg().get("image_provider", "gemini")
    names = raw if isinstance(raw, (list, tuple)) else str(raw).replace("+", ",").split(",")
    chain: list[str] = []
    for name in (str(item).strip().lower() for item in names):
        if name in PROVIDERS and name not in chain:
            chain.append(name)
    return chain


def local_command() -> str | None:
    """Resolve the mflux CLI: explicit config, PATH, then next to the running interpreter."""
    command = str(_cfg().get("local_command") or DEFAULT_LOCAL_COMMAND)
    if os.path.isabs(command):
        return command if os.path.isfile(command) else None
    return shutil.which(command) or shutil.which(command, path=os.path.dirname(sys.executable))


def _provider_ready(name: str) -> bool:
    if name == "gemini":
        return bool(get_api_key())
    return name == "local" and local_command() is not None


def available_providers() -> list[str]:
    return [name for name in provider_chain() if _provider_ready(name)]


def is_enabled() -> bool:
    return bool(available_providers())


def missing_provider_hint() -> str:
    chain = provider_chain()
    hints = []
    if "local" in chain:
        hints.append("cài mflux (uv tool install mflux) cho image_provider = \"local\"")
    if "gemini" in chain or not chain:
        hints.append("điền [tuvi].gemini_api_key cho Gemini")
    return "Chưa có nguồn vẽ tranh khả dụng: " + " hoặc ".join(hints) + "."


def build_prompt(subject: str) -> str:
    style = str(_cfg().get("image_style") or DEFAULT_STYLE)
    return f"{style}\n\nScene: {subject.strip()}\n\nVertical 9:16 portrait illustration."


def build_local_prompt(subject: str) -> str:
    style = str(_cfg().get("image_style") or DEFAULT_LOCAL_STYLE)
    return f"{subject.strip().rstrip('.')}. {style}"


# ---------------------------------------------------------------- cache

def _cache_dir() -> str:
    configured = str(_cfg().get("image_cache_dir") or "")
    return configured or utils.storage_dir("cache/tuvi_images")


def _cache_enabled() -> bool:
    return bool(_cfg().get("image_cache", True))


def _cache_path(provider: str, subject: str) -> str:
    prompt = build_local_prompt(subject) if provider == "local" else build_prompt(subject)
    model = _local_settings()["model"] if provider == "local" else str(_cfg().get("image_model") or DEFAULT_MODEL)
    digest = hashlib.sha256(f"{provider}\n{model}\n{prompt}".encode()).hexdigest()[:32]
    return os.path.join(_cache_dir(), f"{provider}-{digest}.png")


def _cache_lookup(subject: str, output_path: str) -> str | None:
    if not _cache_enabled():
        return None
    for provider in available_providers():
        cached = _cache_path(provider, subject)
        if os.path.isfile(cached):
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            shutil.copyfile(cached, output_path)
            logger.info(f"image cache hit ({provider}): {output_path}")
            return output_path
    return None


def _cache_store(provider: str, subject: str, image_path: str) -> None:
    if not _cache_enabled():
        return
    try:
        cached = _cache_path(provider, subject)
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        shutil.copyfile(image_path, cached)
    except OSError as exc:
        logger.warning(f"image cache write failed: {exc}")


# ---------------------------------------------------------------- gemini

def _extract_image(response) -> Image.Image:
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return Image.open(io.BytesIO(inline.data)).convert("RGB")
    raise ImageGenerationError("Gemini response contains no image")


def _generate_gemini(subject: str, output_path: str, client=None) -> str:
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


# ---------------------------------------------------------------- local (mflux)

def _int_setting(name: str, default: int) -> int:
    try:
        value = int(_cfg().get(name) or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _local_settings() -> dict:
    cfg = _cfg()
    return {
        "model": str(cfg.get("local_model") or DEFAULT_LOCAL_MODEL),
        "base_model": str(cfg.get("local_base_model") or DEFAULT_LOCAL_BASE_MODEL),
        "quantize": cfg.get("local_quantize") or "",
        "steps": _int_setting("local_steps", LOCAL_STEPS),
        "width": _int_setting("local_width", LOCAL_SIZE[0]),
        "height": _int_setting("local_height", LOCAL_SIZE[1]),
        "timeout": _int_setting("local_timeout", LOCAL_TIMEOUT),
        "lora": str(cfg.get("local_lora") or ""),
        "lora_scale": str(cfg.get("local_lora_scale") or "1.0"),
    }


def _seed_for(subject: str) -> int:
    """Stable per prompt so a retry of the same scene reproduces the same picture."""
    return int(hashlib.sha256(subject.strip().encode()).hexdigest()[:8], 16) % 1_000_000_000


def build_local_command(command: str, prompt: str, output_path: str, seed: int) -> list[str]:
    settings = _local_settings()
    args = [
        command,
        "--model", settings["model"],
        "--base-model", settings["base_model"],
        "--steps", str(settings["steps"]),
        "--width", str(settings["width"]),
        "--height", str(settings["height"]),
        "--seed", str(seed),
        "--prompt", prompt,
        "--output", output_path,
    ]
    if settings["quantize"]:
        args += ["--quantize", str(settings["quantize"])]
    if settings["lora"]:
        args += ["--lora-paths", settings["lora"], "--lora-scales", settings["lora_scale"]]
    return args


def _generate_local(subject: str, output_path: str) -> str:
    """Run mflux in a child process so the model's ~8GB is returned to the OS before TTS/render."""
    command = local_command()
    if not command:
        raise ImageGenerationError("mflux-generate not found (uv tool install mflux)")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tuvi-flux-") as scratch:
        raw_path = os.path.join(scratch, "raw.png")
        args = build_local_command(command, build_local_prompt(subject), raw_path, _seed_for(subject))
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=_local_settings()["timeout"]
            )
        except subprocess.TimeoutExpired as exc:
            raise ImageGenerationError(f"mflux timed out after {exc.timeout}s") from exc
        except OSError as exc:
            raise ImageGenerationError(f"cannot run mflux: {exc}") from exc
        if result.returncode != 0 or not os.path.isfile(raw_path):
            tail = (result.stderr or result.stdout or "").strip()[-400:]
            raise ImageGenerationError(f"mflux exited {result.returncode}: {tail}")
        with Image.open(raw_path) as raw:
            image = raw.convert("RGB")
        image.save(output_path, "PNG")
    logger.info(f"local image saved: {output_path} ({image.width}x{image.height})")
    return output_path


# ---------------------------------------------------------------- public API

def generate_image(subject: str, output_path: str, client=None) -> str:
    """Generate one 9:16 illustration for ``subject`` and save it as PNG.

    Serves the cache first, then walks the provider chain until one succeeds.
    """
    cached = _cache_lookup(subject, output_path)
    if cached:
        return cached
    providers = available_providers()
    if client is not None and "gemini" in provider_chain() and "gemini" not in providers:
        providers.append("gemini")  # an injected client needs no API key
    if not providers:
        raise ImageGenerationError(missing_provider_hint())
    errors: list[str] = []
    for provider in providers:
        try:
            if provider == "local":
                _generate_local(subject, output_path)
            else:
                _generate_gemini(subject, output_path, client=client)
        except Exception as exc:
            logger.warning(f"{provider} image failed, trying next provider: {exc}")
            errors.append(f"{provider}: {exc}")
            continue
        _cache_store(provider, subject, output_path)
        return output_path
    raise ImageGenerationError("; ".join(errors))


def generate_images(jobs: list[tuple[str, str]]) -> dict[str, str]:
    """Generate ``(subject, output_path)`` jobs; returns path -> error for failures.

    Local FLUX runs one image at a time (parallel runs would not fit next to VieNeu in RAM);
    a Gemini-only chain keeps the parallel fan-out.
    """
    if not jobs:
        return {}
    errors: dict[str, str] = {}

    def run(job):
        subject, output_path = job
        try:
            generate_image(subject, output_path)
        except Exception as exc:
            errors[output_path] = str(exc)

    workers = 1 if "local" in available_providers() else GEMINI_WORKERS
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, jobs))
    return errors
