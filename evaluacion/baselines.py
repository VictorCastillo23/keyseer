"""
evaluacion.baselines
=====================

Metodos de comparacion para el estudio empirico (Opcion C). Cada funcion
devuelve un dict uniforme:

    {"keyframes": [...], "elapsed": float, "state_bytes": int, "n_frames": int}

`state_bytes` es el tamano del ESTADO que el metodo necesita mantener para
seguir procesando el siguiente frame (no el costo de generar el resultado
final). Para metodos online es O(HW); para VSUMM es O(T) porque acumula un
descriptor por frame ya visto.
"""

import time
import numpy as np

from keyseer.core import KeySeer, KeySeerConfig
from keyseer.selection import select_peaks, enforce_min_distance

__all__ = [
    "uniform_sampling", "frame_difference", "mog2_ratio_peaks",
    "vsumm_kmeans", "dual_background", "jacobs_pless_approx",
]


def _iter_video(video_path, resize_to=None, stride=1):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"no se pudo abrir: {video_path}")
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            f = f.astype(np.float32)
            if resize_to is not None:
                H, W = resize_to
                h0, w0 = f.shape[:2]
                yi = (np.arange(H) * h0 // H).clip(0, h0 - 1)
                xi = (np.arange(W) * w0 // W).clip(0, w0 - 1)
                f = f[yi][:, xi]
            yield f
        i += 1
    cap.release()


# --------------------------------------------------------------------- #
def uniform_sampling(video_path, budget, resize_to=None, stride=1):
    t0 = time.time()
    n = sum(1 for _ in _iter_video(video_path, resize_to, stride))
    idx = list(np.linspace(0, max(n - 1, 0), budget, dtype=int))
    return {"keyframes": sorted(set(idx)), "elapsed": time.time() - t0,
            "state_bytes": 8, "n_frames": n}


# --------------------------------------------------------------------- #
def frame_difference(video_path, budget, min_distance=10, resize_to=None,
                     stride=1):
    """Picos en |I_t - I_{t-1}|. Estado: un solo frame anterior -> O(HW)."""
    t0 = time.time()
    prev = None
    diffs = []
    n = 0
    for f in _iter_video(video_path, resize_to, stride):
        if prev is not None:
            diffs.append(float(np.mean(np.abs(f - prev))))
        else:
            diffs.append(0.0)
        prev = f
        n += 1
    diffs = np.asarray(diffs)
    peaks = select_peaks(diffs, min_distance=min_distance, prominence=0.0,
                         top_n=budget)
    state = prev.nbytes if prev is not None else 0
    return {"keyframes": peaks, "elapsed": time.time() - t0,
            "state_bytes": state, "n_frames": n}


# --------------------------------------------------------------------- #
def mog2_ratio_peaks(video_path, budget, min_distance=10, resize_to=None,
                     stride=1, history=500, var_threshold=16):
    """
    Metodo v1 (turno 1 de esta conversacion): cv2.createBackgroundSubtractorMOG2
    + ratio de foreground por frame, picos con find_peaks. Representa "usar
    MOG2 tal cual, sin leer su dinamica interna".
    """
    import cv2
    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=history,
                                                  varThreshold=var_threshold,
                                                  detectShadows=True)
    ratios = []
    n = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            if resize_to is not None:
                frame = cv2.resize(frame, (resize_to[1], resize_to[0]))
            mask = back_sub.apply(frame)
            mask[mask == 127] = 0
            ratios.append(cv2.countNonZero(mask) / mask.size)
            n += 1
        frame_idx += 1
    cap.release()
    ratios = np.asarray(ratios)
    peaks = select_peaks(ratios, min_distance=min_distance, prominence=0.0,
                         top_n=budget)
    # estado de MOG2: cv2 no expone el tamano del modelo interno via Python,
    # asi que se reporta como no verificable en vez de inventar un numero.
    return {"keyframes": peaks, "elapsed": time.time() - t0,
            "state_bytes": None,
            "n_frames": n}


# --------------------------------------------------------------------- #
def vsumm_kmeans(video_path, budget, resize_to=(60, 80), bins=(8, 8, 8),
                 stride=1, seed=0):
    """
    Estilo VSUMM (Avila et al. 2011), simplificado: histograma de color por
    frame + k-means, keyframe = frame mas cercano a cada centroide.

    IMPORTANTE (el punto del baseline): para poder hacer k-means al final
    hay que HABER GUARDADO el descriptor de cada frame visto. El estado
    crece O(T), a diferencia de los metodos online. Este baseline existe en
    el estudio precisamente para hacer esa comparacion de memoria explicita.
    """
    import cv2
    t0 = time.time()
    feats = []
    n = 0
    for f in _iter_video(video_path, resize_to, stride):
        hsv = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins),
                            [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        feats.append(hist.ravel())
        n += 1
    X = np.stack(feats).astype(np.float64)
    state_bytes = X.nbytes  # esto es lo que se acumula: O(T)

    k = min(budget, len(X))
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(15):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)
        for c in range(k):
            pts = X[assign == c]
            if len(pts):
                centers[c] = pts.mean(axis=0)
    keyframes = []
    for c in range(k):
        idxs = np.where(assign == c)[0]
        if len(idxs) == 0:
            continue
        d = np.linalg.norm(X[idxs] - centers[c], axis=1)
        keyframes.append(int(idxs[np.argmin(d)]))

    return {"keyframes": sorted(set(keyframes)), "elapsed": time.time() - t0,
            "state_bytes": state_bytes, "n_frames": n}


# --------------------------------------------------------------------- #
def dual_background(video_path, budget, alpha_fast=0.05, alpha_slow=0.005,
                    min_distance=10, resize_to=None, stride=1,
                    min_area_frac=0.001):
    """
    Modelo de fondo dual (fast/slow). Competidor directo identificado en la
    revision de literatura (Park et al. 2019 y familia): un modelo rapido
    absorbe objetos detenidos pronto (su foreground cae a 0), un modelo
    lento tarda mas (su foreground sigue en 1). La diferencia
    slow_fg & ~fast_fg aisla objetos estacionarios.

    Estado: DOS modelos KeySeer completos -> ~2x la memoria de un solo modelo.
    """
    t0 = time.time()
    fast = slow = None
    signal = []
    n = 0
    for f in _iter_video(video_path, resize_to, stride):
        if fast is None:
            H, W, C = f.shape
            fast = KeySeer(H, W, C, KeySeerConfig(alpha=alpha_fast))
            slow = KeySeer(H, W, C, KeySeerConfig(alpha=alpha_slow))
        rf = fast.update(f)
        rs = slow.update(f)
        stationary = rs.fg_mask & (~rf.fg_mask)
        signal.append(float(stationary.mean()))
        n += 1
    signal = np.asarray(signal)
    peaks = select_peaks(signal, min_distance=min_distance, prominence=0.0,
                         top_n=budget)
    state_bytes = 0
    if fast is not None:
        per_model = fast.w.nbytes + fast.mu.nbytes + fast.var.nbytes
        state_bytes = 2 * per_model
    return {"keyframes": peaks, "elapsed": time.time() - t0,
            "state_bytes": state_bytes, "n_frames": n, "signal": signal}


# --------------------------------------------------------------------- #
def jacobs_pless_approx(video_path, budget, n_scales=4, tau_min=4,
                        tau_ratio=4.0, min_distance=10, resize_to=None,
                        stride=1):
    """
    APROXIMACION del enfoque de Jacobs & Pless (2006/2008): banco de medias
    moviles exponenciales del frame a escalas temporales geometricamente
    espaciadas tau_k = tau_min * tau_ratio^k. La "novedad a escala k" es
    |I_t - EMA_k|; se agrega energia entre escalas adyacentes como
    representacion de scale-space y se usa su suma como senal escalar.

    ADVERTENCIA DE HONESTIDAD: no tengo el texto completo del paper, solo
    resumenes/abstracts. Esto NO reproduce sus filtros causales exactos;
    es una aproximacion razonable del principio (multiescala, causal,
    memoria constante) construida para que el baseline sea comparable, no
    una replica validada. Reportar resultados de este baseline con esa
    salvedad explicita.

    Estado: K medias moviles (una imagen por escala) -> O(HWK).
    """
    t0 = time.time()
    taus = [tau_min * (tau_ratio ** k) for k in range(n_scales)]
    alphas = [1.0 / t for t in taus]
    emas = [None] * n_scales
    signal = []
    n = 0
    for f in _iter_video(video_path, resize_to, stride):
        novelties = []
        for k, a in enumerate(alphas):
            if emas[k] is None:
                emas[k] = f.copy()
            novelties.append(np.mean(np.abs(f - emas[k])))
            emas[k] = emas[k] + a * (f - emas[k])
        # energia de scale-space: diferencias entre escalas adyacentes
        adj = [abs(novelties[i] - novelties[i + 1]) for i in range(n_scales - 1)]
        signal.append(float(sum(novelties) + sum(adj)))
        n += 1
    signal = np.asarray(signal)
    peaks = select_peaks(signal, min_distance=min_distance, prominence=0.0,
                         top_n=budget)
    state_bytes = sum(e.nbytes for e in emas if e is not None)
    return {"keyframes": peaks, "elapsed": time.time() - t0,
            "state_bytes": state_bytes, "n_frames": n, "signal": signal}
