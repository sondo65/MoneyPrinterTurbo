"""Looping layer of drifting golden dust that gives the still illustrations a little life.

The layer is a grayscale mask video (white = dust). It is rendered once, cached, looped over
the whole video and tinted gold with ``alphamerge`` at composition time. Every particle moves
on a sine path with a whole number of cycles per loop, so the loop has no visible seam.
"""

import math
import os
import random
import subprocess

import numpy as np
from loguru import logger

from app.utils import utils

WIDTH, HEIGHT = 1080, 1920
FPS = 30
LOOP_SECONDS = 10
PARTICLES = 70
DUST_COLOR_HEX = "0xFFE2A0"
CACHE_VERSION = 1
# Sprite radii (px) the particles are drawn with; a few sizes read as depth.
SPRITE_RADII = (4, 7, 11)
MAX_ALPHA = 0.85


def _sprite(radius: int) -> np.ndarray:
    size = radius * 4
    axis = np.arange(size) - (size - 1) / 2
    dist = np.hypot(*np.meshgrid(axis, axis))
    return np.exp(-2.0 * (dist / radius) ** 2).astype(np.float32)


def _particles(seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [
        {
            "x": rng.uniform(0, WIDTH),
            "y": rng.uniform(0, HEIGHT),
            "ax": rng.uniform(20, 70),
            "ay": rng.uniform(50, 160),
            "nx": rng.choice((1, 2)),
            "ny": rng.choice((1, 2)),
            "nb": rng.choice((1, 2, 3)),
            "px": rng.uniform(0, math.tau),
            "py": rng.uniform(0, math.tau),
            "pb": rng.uniform(0, math.tau),
            "sprite": rng.randrange(len(SPRITE_RADII)),
        }
        for _ in range(PARTICLES)
    ]


def _frame(particles: list[dict], sprites: list[np.ndarray], phase: float) -> np.ndarray:
    canvas = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for p in particles:
        brightness = 0.15 + 0.85 * (0.5 + 0.5 * math.sin(math.tau * p["nb"] * phase + p["pb"])) ** 2
        cx = p["x"] + p["ax"] * math.sin(math.tau * p["nx"] * phase + p["px"])
        cy = p["y"] + p["ay"] * math.sin(math.tau * p["ny"] * phase + p["py"])
        sprite = sprites[p["sprite"]]
        half = sprite.shape[0] // 2
        left, top = int(cx) - half, int(cy) - half
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(WIDTH, left + sprite.shape[1]), min(HEIGHT, top + sprite.shape[0])
        if x0 >= x1 or y0 >= y1:
            continue
        canvas[y0:y1, x0:x1] += sprite[y0 - top:y1 - top, x0 - left:x1 - left] * brightness
    return np.clip(canvas * MAX_ALPHA, 0.0, 1.0)


def particle_loop_path() -> str:
    return os.path.join(utils.storage_dir("cache/tuvi_ambient"), f"dust-v{CACHE_VERSION}.mp4")


def ensure_particle_loop(ffmpeg: str, seconds: int = LOOP_SECONDS) -> str:
    """Return the cached dust mask video, rendering it on first use."""
    path = particle_loop_path() if seconds == LOOP_SECONDS else particle_loop_path().replace(".mp4", f"-{seconds}s.mp4")
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info("rendering tuvi dust loop (one-off)")

    particles = _particles(seed=7)
    sprites = [_sprite(r) for r in SPRITE_RADII]
    frames = seconds * FPS
    partial = path + ".part.mp4"
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", partial,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for index in range(frames):
            frame = _frame(particles, sprites, index / frames)
            process.stdin.write((frame * 255).astype(np.uint8).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed rendering dust loop: {error.strip()[-400:]}")
    except Exception:
        process.kill()
        if os.path.isfile(partial):
            os.remove(partial)
        raise
    os.replace(partial, path)
    return path
