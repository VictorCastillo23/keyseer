"""
tests.conftest
===============

Fixtures compartidas para tests de blobtrack.py y keyframes.py: videos
sinteticos chicos generados en disco (tmp_path), sin depender de ningun
video real commiteado al repo.
"""

import numpy as np
import pytest

FRAME_W, FRAME_H = 64, 48


def _write_noisy_video(path, n_frames, event=(15, 30), fps=10, seed=0):
    """Fondo con ruido leve (como evaluacion/harness.py) + un rectangulo
    moviendose durante `event` -- da frames unicos entre si (para chequeos
    de correspondencia pixel-exacta) y una señal de movimiento real para
    que blobtrack produzca tracks."""
    import cv2
    rng = np.random.default_rng(seed)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (FRAME_W, FRAME_H))
    es, ee = event
    for i in range(n_frames):
        f = np.full((FRAME_H, FRAME_W, 3), 40, np.uint8)
        f = cv2.add(f, rng.integers(0, 3, f.shape, dtype=np.uint8))
        if es <= i < ee:
            cx = 10 + (i - es) * 2
            cv2.rectangle(f, (cx, 15), (cx + 10, 30), (0, 180, 0), -1)
        out.write(f)
    out.release()


def _write_static_then_moving_video(path, quiet_before=20, moving=15,
                                    quiet_after=20, fps=10):
    """Frames BIT-IDENTICOS durante los tramos quietos (sin ruido) y un
    rectangulo moviendose durante `moving` frames en el medio. Pensado
    para tests de MotionGatedStride: el gate debe quedar quieto de forma
    determinista durante los tramos estaticos."""
    import cv2
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (FRAME_W, FRAME_H))
    bg = np.full((FRAME_H, FRAME_W, 3), 40, np.uint8)
    for _ in range(quiet_before):
        out.write(bg)
    for i in range(moving):
        f = bg.copy()
        cx = 10 + i * 2
        cv2.rectangle(f, (cx, 15), (cx + 10, 30), (0, 180, 0), -1)
        out.write(f)
    for _ in range(quiet_after):
        out.write(bg)
    out.release()


@pytest.fixture
def tiny_video(tmp_path):
    """Factory: tiny_video(n_frames=120, event=(20,100), fps=10, seed=0) ->
    path str. Event span (80 frames) is deliberately long enough that
    tracks clear BlobPersistenceTracker's default min_age_to_count=8 even
    under stride=3 sampling (~27 processed samples during the event)."""
    def _make(n_frames=120, event=(20, 100), fps=10, seed=0):
        path = str(tmp_path / "tiny.mp4")
        _write_noisy_video(path, n_frames, event=event, fps=fps, seed=seed)
        return path
    return _make


@pytest.fixture
def static_then_moving_video(tmp_path):
    """Factory: static_then_moving_video(quiet_before=20, moving=15,
    quiet_after=20, fps=10) -> path str."""
    def _make(quiet_before=20, moving=15, quiet_after=20, fps=10):
        path = str(tmp_path / "static_then_moving.mp4")
        _write_static_then_moving_video(path, quiet_before, moving,
                                        quiet_after, fps)
        return path
    return _make
