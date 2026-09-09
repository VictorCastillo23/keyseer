"""
keyseer.pipeline
=================

API de alto nivel: video -> keyframes, en una sola pasada.
"""

import numpy as np

from .core import KeySeer, KeySeerConfig
from .metrics import SignalAccumulator, combine
from .selection import select_submodular, select_peaks

__all__ = ["analyze_frames", "analyze_video", "extract_keyframes", "KeySeerResult"]


class KeySeerResult:
    """Resultado del analisis: series, score y keyframes."""

    def __init__(self, arrays, score, descriptors, keyframes=None):
        self.arrays = arrays
        self.score = score
        self.descriptors = descriptors
        self.keyframes = keyframes or []

    def __repr__(self):
        return (f"KeySeerResult(n_frames={len(self.score)}, "
                f"n_keyframes={len(self.keyframes)})")


def analyze_frames(frames, config=None, resize_to=None, pool_block=8,
                   descriptor_grid=8, progress=None):
    """
    Procesa un iterable de frames (H,W,C) y devuelve las series KeySeer.

    `resize_to` = (h, w) reduce la resolucion del modelo. Recomendado:
    el coste es O(H*W*M) por frame y la metrica es robusta a la escala.
    """
    model, acc = None, None
    for t, frame in enumerate(frames):
        f = np.asarray(frame)
        if f.ndim == 2:
            f = f[:, :, None]
        if resize_to is not None and f.shape[:2] != tuple(resize_to):
            f = _resize(f, resize_to)
        if model is None:
            model = KeySeer(f.shape[0], f.shape[1], f.shape[2], config)
            acc = SignalAccumulator(pool_block=pool_block,
                                    descriptor_grid=descriptor_grid)
        acc.push(model.update(f))
        if progress is not None and t % progress == 0:
            print(f"  frame {t}", flush=True)

    if model is None:
        raise ValueError("no se recibio ningun frame")

    # drenar la latencia pendiente: repetir el ultimo estado no altera el
    # modelo de forma significativa pero libera las consolidaciones en cola
    return acc, model


def _resize(f, hw):
    """Resize por vecino mas cercano, sin dependencias externas."""
    h, w = hw
    H, W = f.shape[:2]
    yi = (np.arange(h) * H // h).clip(0, H - 1)
    xi = (np.arange(w) * W // w).clip(0, W - 1)
    return f[yi][:, xi]


def analyze_video(video_path, config=None, resize_to=(180, 320), stride=1,
                  progress=None):
    """Analiza un video en disco. Requiere opencv-python para la lectura."""
    try:
        import cv2
    except ImportError:
        raise ImportError("analyze_video requiere opencv-python para leer video")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"no se pudo abrir: {video_path}")

    def gen():
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield fr.astype(np.float32)
            i += 1

    try:
        acc, model = analyze_frames(gen(), config=config, resize_to=resize_to,
                                    progress=progress)
    finally:
        cap.release()
    return acc, model


def extract_keyframes(video_path, budget=8, config=None, resize_to=(180, 320),
                      stride=1, method="submodular", min_distance=10,
                      weights=None, progress=None):
    """
    Extrae keyframes de un video.

    `method`: "submodular" (recomendado, con garantia 1-1/e) o "peaks".
    Devuelve un KeySeerResult. Los indices son relativos al submuestreo por
    `stride`; multiplica por `stride` para indices del video original.
    """
    acc, model = analyze_video(video_path, config=config, resize_to=resize_to,
                               stride=stride, progress=progress)
    arrays = acc.sig.as_arrays()
    w = weights or {}
    score = combine(arrays, **w)

    warm = int(1.0 / max((config or KeySeerConfig()).alpha, 1e-9))
    usable = np.zeros_like(score, dtype=bool)
    usable[min(warm, len(score)):] = True
    score_masked = np.where(usable, score, 0.0)

    if method == "submodular":
        kf = select_submodular(score_masked, acc.sig.descriptors,
                               budget=budget, min_distance=min_distance)
    elif method == "peaks":
        kf = select_peaks(score_masked, min_distance=min_distance,
                          prominence=0.05, top_n=budget)
    else:
        raise ValueError(f"metodo desconocido: {method}")

    return KeySeerResult(arrays, score, acc.sig.descriptors, [int(k) for k in kf])
