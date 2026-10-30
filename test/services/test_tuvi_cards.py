import pytest
from PIL import Image

from app.config import config
from app.services.tuvi import cards
from app.services.tuvi.content import Scene, VideoContent


def _content():
    return VideoContent(
        title="Tứ Tượng là gì?",
        hook="",
        scenes=[
            Scene("Bốn thần thú", "", "a", image_prompt="a scholar"),
            Scene("Rồng", "", "b", zodiac="thin"),
            Scene("", "", "c"),
        ],
        cta="Tải app Tử Vi AI để xem lá số của riêng bạn",
        caption="",
    )


def _art(tmp_path, color=(200, 30, 30)):
    path = tmp_path / f"art-{color[0]}.png"
    Image.new("RGB", (768, 1344), color).save(path)
    return str(path)


def test_every_scene_is_an_illustration_and_last_is_cta(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "tuvi", {"app_name": "Tử Vi AI", "brand_assets_dir": ""})
    art = _art(tmp_path)
    calls = []
    for name in ("render_illustration_card", "render_cta_card"):
        original = getattr(cards, name)

        def spy(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(cards, name, spy)

    paths = cards.render_cards(_content(), str(tmp_path / "out"), {0: art, 1: art, 2: art})

    assert calls == ["render_illustration_card", "render_illustration_card", "render_cta_card"]
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (cards.WIDTH, cards.HEIGHT)
            red, green, _ = image.getpixel((cards.WIDTH // 2, cards.HEIGHT // 2))
            assert red > 150 and green < 90


def test_cta_card_fades_bottom_into_paper(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "tuvi", {"brand_assets_dir": ""})
    out = cards.render_cta_card(_art(tmp_path, (20, 20, 160)), str(tmp_path / "cta.png"), app_name="Tử Vi AI")

    with Image.open(out) as image:
        top_r, _, top_b = image.getpixel((60, 400))
        bottom_r, _, bottom_b = image.getpixel((60, cards.HEIGHT - 60))
    assert top_b > top_r
    assert abs(bottom_r - cards.PAPER[0]) < 40 and abs(bottom_b - cards.PAPER[2]) < 40


def test_missing_illustration_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "tuvi", {"brand_assets_dir": ""})
    with pytest.raises(cards.MissingIllustrationError):
        cards.render_cards(_content(), str(tmp_path), {0: _art(tmp_path)})


def test_brand_is_a_transparent_overlay_not_part_of_the_scene_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "tuvi", {"app_name": "Tử Vi AI", "brand_assets_dir": ""})
    art = _art(tmp_path, (200, 30, 30))
    card = cards.render_illustration_card(art, str(tmp_path / "card.png"), app_name="Tử Vi AI")
    brand = cards.render_brand_overlay(str(tmp_path / "brand.png"), "Tử Vi AI")

    with Image.open(card) as image:
        # The flat red artwork is only blended towards paper by the same amount everywhere,
        # so a baked-in mark would show up as extra colours in the top band.
        top_band = image.crop((0, 100, cards.WIDTH, 200)).convert("RGB").getcolors(maxcolors=1_000_000)
    assert len(top_band) < 50
    with Image.open(brand) as layer:
        assert layer.mode == "RGBA" and layer.size == (cards.WIDTH, cards.HEIGHT)
        assert layer.getpixel((5, cards.HEIGHT // 2))[3] == 0
        assert layer.crop((0, 100, cards.WIDTH, 200)).getchannel("A").getextrema()[1] > 0
