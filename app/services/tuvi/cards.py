"""Scene frames in the "giấy dó" style: AI illustration on warm rice paper, small brand mark."""

import math
import os
import random

from loguru import logger
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.config import config
from app.services.tuvi.content import VideoContent
from app.utils import utils

WIDTH, HEIGHT = 1080, 1920

PAPER = (221, 195, 163)
INK_BROWN = (92, 52, 44)
BRAND_GOLD = (160, 118, 52)
LINE_GOLD = (170, 135, 80)

# The CTA brand block sits in the bottom band, below the centred captions.
CTA_FADE_TOP = 1260
CTA_ICON_TOP = 1430


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = os.path.join(utils.font_dir(), name)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        logger.warning(f"font not found: {path}, using default font")
        return ImageFont.load_default(size)


def serif(size: int) -> ImageFont.FreeTypeFont:
    return _font("NotoSerif-Bold.ttf", size)


def sans_bold(size: int) -> ImageFont.FreeTypeFont:
    return _font("BeVietnamPro-Bold.ttf", size)


def _load_asset(relative_path: str) -> Image.Image | None:
    base = str(config.tuvi.get("brand_assets_dir", "") or "")
    if not base:
        return None
    try:
        return Image.open(os.path.join(base, relative_path)).convert("RGBA")
    except OSError:
        return None


def paper_background(seed: int = 0) -> Image.Image:
    """Warm rice-paper texture: grain, faint fibres and a soft vignette."""
    rng = random.Random(seed)
    small = Image.effect_noise((WIDTH // 4, HEIGHT // 4), 18).resize((WIDTH, HEIGHT), Image.BICUBIC)
    base = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    grain = Image.merge("RGB", (small, small, small))
    paper = Image.blend(base, Image.composite(grain, base, Image.new("L", (WIDTH, HEIGHT), 40)), 0.35)

    fibres = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(fibres)
    for _ in range(260):
        x, y = rng.randrange(WIDTH), rng.randrange(HEIGHT)
        length = rng.randrange(8, 40)
        angle = rng.uniform(0, math.pi)
        tone = rng.choice(((255, 245, 230, 60), (150, 120, 90, 35)))
        draw.line(
            [(x, y), (x + length * math.cos(angle), y + length * math.sin(angle))], fill=tone, width=1
        )
    paper = Image.alpha_composite(paper.convert("RGBA"), fibres.filter(ImageFilter.GaussianBlur(0.6)))

    vignette = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(vignette).ellipse([-300, -200, WIDTH + 300, HEIGHT + 200], fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(220))
    shade = Image.new("RGBA", (WIDTH, HEIGHT), (150, 115, 80, 70))
    shade.putalpha(Image.eval(vignette, lambda v: int((255 - v) * 0.35)))
    return Image.alpha_composite(paper, shade)


def _cover(image: Image.Image) -> Image.Image:
    scale = max(WIDTH / image.width, HEIGHT / image.height)
    resized = image.resize((math.ceil(image.width * scale), math.ceil(image.height * scale)), Image.LANCZOS)
    left = (resized.width - WIDTH) // 2
    top = (resized.height - HEIGHT) // 2
    return resized.crop((left, top, left + WIDTH, top + HEIGHT))


def render_brand_overlay(output_path: str, app_name: str) -> str:
    """Transparent full-frame layer with the small centred mark (round app icon + spaced name).

    It is laid over the finished video, not baked into the scene cards, so the camera moves
    never carry the app name with them and it stays put in the viewer's eye line.
    """
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    font = serif(30)
    label = " ".join(app_name.upper())
    icon = _load_asset("app_icon.png")
    icon_size = 46 if icon else 0
    gap = 14 if icon else 0
    total = icon_size + gap + font.getlength(label)
    x = (WIDTH - total) / 2
    y = 120

    # Soft paper-coloured glow so the mark stays readable over any part of the illustration.
    glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    ImageDraw.Draw(glow).rounded_rectangle(
        [x - 28, y - 14, x + total + 28, y + 60], radius=40, fill=PAPER + (150,)
    )
    layer.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18)))

    draw = ImageDraw.Draw(layer)
    if icon:
        icon = icon.resize((icon_size, icon_size), Image.LANCZOS)
        mask = Image.new("L", icon.size, 0)
        ImageDraw.Draw(mask).ellipse([0, 0, icon_size - 1, icon_size - 1], fill=255)
        layer.paste(icon, (int(x), y), mask)
    draw.text((x + icon_size + gap, y + icon_size / 2), label, font=font, fill=BRAND_GOLD, anchor="lm")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    layer.save(output_path, "PNG")
    return output_path


def _illustration_canvas(illustration_path: str, seed: int) -> Image.Image:
    art = _cover(Image.open(illustration_path).convert("RGB")).convert("RGBA")
    # Pull the AI image towards the shared paper tone so every scene feels like one scroll.
    return Image.blend(art, paper_background(seed), 0.12)


def render_illustration_card(illustration_path: str, output_path: str, *, app_name: str, seed: int = 0) -> str:
    return _save(_illustration_canvas(illustration_path, seed), output_path)


def _paper_fade(canvas: Image.Image, top: int, seed: int) -> None:
    """Dissolve the illustration into plain paper from ``top`` down to the bottom edge."""
    mask = Image.new("L", (1, HEIGHT), 0)
    span = HEIGHT - top
    for y in range(top, HEIGHT):
        mask.putpixel((0, y), min(255, int(255 * ((y - top) / (span * 0.45)) ** 1.2)))
    paper = paper_background(seed + 100)
    paper.putalpha(mask.resize((WIDTH, HEIGHT)))
    canvas.alpha_composite(paper)


def render_cta_card(illustration_path: str, output_path: str, *, app_name: str, seed: int = 0) -> str:
    """Last scene: the illustration fades into paper at the bottom, where the app badge sits."""
    canvas = _illustration_canvas(illustration_path, seed)
    _paper_fade(canvas, CTA_FADE_TOP, seed)
    draw = ImageDraw.Draw(canvas)

    y = CTA_ICON_TOP
    icon = _load_asset("app_icon.png")
    if icon:
        size = 170
        icon = icon.resize((size, size), Image.LANCZOS)
        mask = Image.new("L", icon.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=38, fill=255)
        left = (WIDTH - size) // 2
        shadow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle(
            [left, y + 10, left + size, y + size + 10], radius=38, fill=(90, 60, 30, 90)
        )
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(12)))
        canvas.paste(icon, (left, y), mask)
        y += size + 20

    draw.text((WIDTH / 2, y + 40), app_name.upper(), font=serif(72), fill=INK_BROWN, anchor="mm")
    y += 110

    button_font = sans_bold(40)
    label = "Tải app miễn phí"
    width = button_font.getlength(label) + 110
    left = (WIDTH - width) / 2
    draw.rounded_rectangle([left, y, left + width, y + 88], radius=44, fill=INK_BROWN)
    draw.text((WIDTH / 2, y + 44), label, font=button_font, fill=(245, 225, 180), anchor="mm")
    return _save(canvas, output_path)


def _save(canvas: Image.Image, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.convert("RGB").save(output_path, "PNG")
    return output_path


class MissingIllustrationError(RuntimeError):
    pass


def render_cards(
    content: VideoContent,
    output_dir: str,
    illustrations: dict[int, str],
    app_name: str | None = None,
) -> list[str]:
    """One 1080x1920 frame per scene, every one an illustration; the last also carries the app badge.

    The small top-of-frame brand mark is not part of these frames: see ``render_brand_overlay``.
    """
    app_name = app_name or config.tuvi.get("app_name", "Tử Vi AI")
    paths = []
    last_index = len(content.scenes) - 1
    for index in range(len(content.scenes)):
        illustration = illustrations.get(index)
        if not illustration or not os.path.isfile(illustration):
            raise MissingIllustrationError(f"scene {index + 1} has no illustration")
        output_path = os.path.join(output_dir, f"scene-{index + 1:02d}.png")
        if index == last_index:
            render_cta_card(illustration, output_path, app_name=app_name, seed=index)
        else:
            render_illustration_card(illustration, output_path, app_name=app_name, seed=index)
        paths.append(output_path)
    return paths
