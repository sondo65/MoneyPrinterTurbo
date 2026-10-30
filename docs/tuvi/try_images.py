"""Render a few sample Tử Vi illustrations to compare image providers before switching.

    uv run python docs/tuvi/try_images.py --provider local --out /tmp/tuvi-samples
    uv run python docs/tuvi/try_images.py --provider gemini --out /tmp/tuvi-samples-gemini

Writes sample-N.png (+ a contact sheet) and prints seconds per image. Ignores the cache.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image  # noqa: E402

from app.config import config  # noqa: E402
from app.services.tuvi import images  # noqa: E402

SAMPLES = [
    "A young scholar in a long indigo ao dai holding a paper lantern at dawn, looking up at the last stars.",
    "An old woman in ao the and khan van carefully pouring tea beside a small altar with incense smoke.",
    "Two hands cupping a glowing lotus bud floating above still water, a single koi below.",
    "A man in khan xep walking a narrow bridge toward a rising sun, a closed umbrella in his hand.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="local", help="image_provider chain, e.g. local, gemini, local,gemini")
    parser.add_argument("--out", required=True)
    parser.add_argument("--style", default="", help="override [tuvi].image_style to test a prompt")
    parser.add_argument("--count", type=int, default=len(SAMPLES))
    args = parser.parse_args()

    config.tuvi["image_provider"] = args.provider
    config.tuvi["image_cache"] = False
    if args.style:
        config.tuvi["image_style"] = args.style
    if not images.is_enabled():
        print(images.missing_provider_hint())
        return 1
    os.makedirs(args.out, exist_ok=True)

    paths = []
    for number, subject in enumerate(SAMPLES[: args.count], start=1):
        path = os.path.join(args.out, f"sample-{number}.png")
        started = time.time()
        images.generate_image(subject, path)
        print(f"sample-{number}: {time.time() - started:.1f}s -> {path}")
        paths.append(path)

    thumbs = [Image.open(p).convert("RGB").resize((360, 640)) for p in paths]
    sheet = Image.new("RGB", (360 * len(thumbs), 640))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, (360 * index, 0))
    sheet_path = os.path.join(args.out, "contact-sheet.png")
    sheet.save(sheet_path)
    print(f"contact sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
