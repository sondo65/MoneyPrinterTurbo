import numpy as np

from app.services.tuvi import ambient, render


def test_dust_frames_loop_without_a_seam():
    particles = ambient._particles(seed=7)
    sprites = [ambient._sprite(r) for r in ambient.SPRITE_RADII]
    first = ambient._frame(particles, sprites, 0.0)
    wrapped = ambient._frame(particles, sprites, 1.0)
    assert np.allclose(first, wrapped, atol=1e-3)
    assert 0.0 < first.max() <= 1.0


def test_particle_loop_is_rendered_once_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(ambient, "particle_loop_path", lambda: str(tmp_path / "dust.mp4"))
    first = ambient.ensure_particle_loop(render.ffmpeg_exe(), seconds=1)
    assert (tmp_path / "dust-1s.mp4").stat().st_size > 0
    stamp = (tmp_path / "dust-1s.mp4").stat().st_mtime_ns
    assert ambient.ensure_particle_loop(render.ffmpeg_exe(), seconds=1) == first
    assert (tmp_path / "dust-1s.mp4").stat().st_mtime_ns == stamp
