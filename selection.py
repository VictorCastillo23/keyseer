"""
mog3.selection
==============

Seleccion del conjunto de keyframes a partir del score por frame.

Se ofrecen dos criterios:

1. `select_peaks` -- deteccion de maximos locales. Baseline simple.

2. `select_submodular` -- maximizacion voraz de una funcion de
   facility-location ponderada por informatividad. Preferido: tiene
   garantia de aproximacion.

   Objetivo:
       F(K) = sum_t  u(t) * max_{k in K} sim(t, k)

   donde u(t) es la informatividad del frame t (el score MOG3) y
   sim(t,k) la similitud entre descriptores espaciales.

   F es monotona y submodular (facility location ponderada con pesos no
   negativos lo es), por lo que el algoritmo voraz garantiza

       F(K_greedy) >= (1 - 1/e) F(K_optimo)

   Esta garantia es la razon de preferirlo sobre picos: da una
   afirmacion demostrable sobre la calidad del resumen, no solo una
   heuristica.
"""

import numpy as np

__all__ = ["select_peaks", "select_submodular", "enforce_min_distance"]


def enforce_min_distance(indices, scores, min_distance):
    """Filtra indices para que disten al menos `min_distance`, priorizando score."""
    if min_distance <= 1 or len(indices) == 0:
        return sorted(indices)
    order = sorted(indices, key=lambda i: -scores[i])
    kept = []
    for i in order:
        if all(abs(i - j) >= min_distance for j in kept):
            kept.append(i)
    return sorted(kept)


def select_peaks(score, min_distance=15, prominence=0.05, top_n=None,
                 raw_floor=None, raw_score=None, raw_window=3):
    """Baseline por maximos locales, con validacion opcional contra la senal cruda."""
    try:
        from scipy.signal import find_peaks
    except ImportError:
        raise ImportError("select_peaks requiere scipy")

    peaks, _ = find_peaks(score, distance=min_distance, prominence=prominence)

    if raw_score is not None and raw_floor is not None:
        valid = []
        for p in peaks:
            lo, hi = max(0, p - raw_window), min(len(raw_score), p + raw_window + 1)
            if raw_score[lo:hi].max() >= raw_floor:
                valid.append(p)
        peaks = np.asarray(valid, dtype=int)

    if top_n is not None and len(peaks) > top_n:
        peaks = np.sort(peaks[np.argsort(score[peaks])[::-1][:top_n]])
    return list(peaks)


def select_submodular(score, descriptors, budget, min_distance=1,
                      candidate_floor=0.0, return_gains=False):
    """
    Seleccion voraz submodular con garantia (1 - 1/e).

    Parametros
    ----------
    score : (T,) informatividad por frame u(t)
    descriptors : dict {frame_index -> vector normalizado}
    budget : numero de keyframes a seleccionar
    min_distance : separacion temporal minima entre keyframes
    candidate_floor : descarta candidatos con score por debajo de esto
    """
    score = np.asarray(score, dtype=np.float64)
    T = len(score)
    idxs = sorted(i for i in descriptors if 0 <= i < T)
    if not idxs or budget <= 0:
        return ([], []) if return_gains else []

    D = np.stack([descriptors[i] for i in idxs])          # (N, d)
    S = D @ D.T                                            # similitud coseno
    np.clip(S, 0.0, None, out=S)
    u = score[idxs]                                        # (N,)

    candidates = [j for j in range(len(idxs)) if score[idxs[j]] >= candidate_floor]
    if not candidates:
        candidates = list(range(len(idxs)))

    selected, gains = [], []
    best_cov = np.zeros(len(idxs))                         # max sim actual

    while len(selected) < budget and candidates:
        best_j, best_gain, best_new = None, -np.inf, None
        for j in candidates:
            if min_distance > 1 and any(
                abs(idxs[j] - idxs[s]) < min_distance for s in selected
            ):
                continue
            new_cov = np.maximum(best_cov, S[:, j])
            gain = float(np.sum(u * (new_cov - best_cov)))
            if gain > best_gain:
                best_j, best_gain, best_new = j, gain, new_cov
        if best_j is None or best_gain <= 1e-12:
            break
        selected.append(best_j)
        gains.append(best_gain)
        best_cov = best_new
        candidates.remove(best_j)

    frames = sorted(idxs[j] for j in selected)
    if return_gains:
        order = {idxs[j]: g for j, g in zip(selected, gains)}
        return frames, [order[f] for f in frames]
    return frames
