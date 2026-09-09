"""
keyseer.keyframes
==================

Modulo B: el ALGORITMO DE KEYFRAMES, separado del modelo de persistencia
(Modulo A). Este modulo no sabe nada de GMMs, MOG2, ni tracking -- solo
consume una señal por frame + descriptores espaciales y decide cuales
frames guardar. Cualquier backend de persistencia que produzca esa forma
de salida es compatible (blobtrack.py y pipeline.py/core.py ambos lo hacen).

Uso tipico (una sola llamada, para uso personal):

    from keyseer.keyframes import extract_keyframes

    result = extract_keyframes("mi_video.mp4", out_dir="keyframes/")
    print(result.keyframe_indices)

Por defecto usa el backend blobtrack (cv2 MOG2 + edad de blob): mas simple,
mas rapido, y en la comparacion de ESTADO.md/RESULTADOS.md igualo o supero
al modelo NumPy en todas las metricas medidas. El backend NumPy (core.py)
sigue disponible via backend="gmm" para quien necesite las señales finas
(Ψ, KL, sorpresa) que el blob tracker no expone.
"""

import os
import time
import numpy as np

from .selection import select_peaks, select_submodular

__all__ = ["KeyframeResult", "extract_keyframes"]


class KeyframeResult:
    def __init__(self, keyframe_indices, scores, elapsed, n_frames,
                fps_source, backend, saved_paths=None):
        self.keyframe_indices = keyframe_indices
        self.scores = scores
        self.elapsed = elapsed
        self.n_frames = n_frames
        self.fps_source = fps_source
        self.backend = backend
        self.saved_paths = saved_paths or []

    def timestamps(self):
        """Segundos de video para cada keyframe (requiere fps de la fuente)."""
        if not self.fps_source:
            return None
        return [k / self.fps_source for k in self.keyframe_indices]

    def __repr__(self):
        return (f"KeyframeResult(n_keyframes={len(self.keyframe_indices)}, "
                f"backend={self.backend!r}, n_frames={self.n_frames})")


def _run_backend(video_path, backend, resize_to, stride, backend_kwargs):
    if backend == "blobtrack":
        from .blobtrack import analyze_video_blobtrack
        r = analyze_video_blobtrack(video_path, resize_to=resize_to,
                                    stride=stride,
                                    tracker_kwargs=backend_kwargs)
        return r["scores"], r["descriptors"], r["n_frames"]

    elif backend == "gmm":
        from .pipeline import analyze_video
        from .metrics import combine
        from .core import KeySeerConfig
        cfg = KeySeerConfig(**(backend_kwargs or {}))
        acc, model = analyze_video(video_path, config=cfg,
                                   resize_to=resize_to, stride=stride)
        arrays = acc.sig.as_arrays()
        score = combine(arrays, gate_power=2.0)
        return score, acc.sig.descriptors, len(score)

    else:
        raise ValueError(f"backend desconocido: {backend!r} "
                         f"(usar 'blobtrack' o 'gmm')")


def _video_fps(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else None


def _save_frames(video_path, indices, stride, out_dir):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    paths = []
    for k in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k) * stride)
        ok, f = cap.read()
        if not ok:
            continue
        p = os.path.join(out_dir, f"keyframe_{int(k) * stride:06d}.jpg")
        cv2.imwrite(p, f)
        paths.append(p)
    cap.release()
    return paths


def extract_keyframes(video_path, budget=8, out_dir=None, backend="blobtrack",
                      method="submodular", min_distance=15, resize_to=(240, 320),
                      stride=1, min_gain_ratio=0.0, backend_kwargs=None):
    """
    Punto de entrada unico para extraer keyframes de un video.

    Parametros principales
    -----------------------
    video_path : ruta al video
    budget     : NUMERO MAXIMO de keyframes (puede devolver menos si el
                 video no tiene suficiente contenido genuinamente distinto)
    out_dir    : si se da, guarda los frames elegidos como .jpg ahi
    backend    : "blobtrack" (por defecto, recomendado) o "gmm" (NumPy,
                 mas lento, mas señales pero sin ventaja medida)
    method     : "submodular" (por defecto, con garantia de aproximacion)
                 o "peaks" (mas simple, sin diversidad explicita)

    Devuelve un KeyframeResult.
    """
    t0 = time.time()
    score, descriptors, n_frames = _run_backend(
        video_path, backend, resize_to, stride, backend_kwargs)

    if method == "submodular":
        kf = select_submodular(score, descriptors, budget=budget,
                               min_distance=min_distance,
                               min_gain_ratio=min_gain_ratio)
    elif method == "peaks":
        kf = select_peaks(score, min_distance=min_distance,
                          prominence=0.0, top_n=budget)
    else:
        raise ValueError(f"metodo desconocido: {method!r}")

    elapsed = time.time() - t0
    fps_source = _video_fps(video_path)

    saved = []
    if out_dir is not None:
        saved = _save_frames(video_path, kf, stride, out_dir)

    return KeyframeResult(
        keyframe_indices=[int(k) * stride for k in kf],
        scores=score, elapsed=elapsed, n_frames=n_frames,
        fps_source=fps_source, backend=backend, saved_paths=saved,
    )
