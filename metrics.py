"""
mog3.metrics
============

Agregacion de los mapas por pixel de MOG3 a escalares por frame.

El pooling espacial no es trivial: la media simple es sensible a ruido
disperso, y el maximo es sensible a outliers de un solo pixel. Se usa
pooling por bloques + media recortada, que suprime activaciones aisladas
y preserva activaciones espacialmente coherentes.
"""

import numpy as np

__all__ = ["spatial_pool", "spatial_coherence", "FrameSignals",
           "SignalAccumulator", "combine"]


def spatial_pool(m, block=8, trim=0.9):
    """
    Pooling robusto de un mapa (H,W) a un escalar.

    1. Media por bloques de `block`x`block`  -> exige coherencia espacial:
       un pixel aislado se diluye, una region compacta sobrevive.
    2. Media de los bloques por debajo del cuantil `trim` mas la masa de
       los que lo superan -> robusta a outliers sin descartar la senal.
    """
    H, W = m.shape
    bh, bw = max(1, H // block), max(1, W // block)
    Hc, Wc = bh * block, bw * block
    blocks = m[:Hc, :Wc].reshape(bh, block, bw, block).mean(axis=(1, 3))
    v = blocks.ravel()
    if v.size == 0:
        return 0.0
    thr = np.quantile(v, trim)
    body = v[v <= thr]
    tail = v[v > thr]
    base = body.mean() if body.size else 0.0
    extra = tail.sum() / v.size if tail.size else 0.0
    return float(base + extra)


def _box_smooth(m, r=2):
    """Media movil 2D separable de radio r, via imagen integral."""
    p = np.pad(m.astype(np.float64), r + 1, mode="edge")
    ii = p.cumsum(0).cumsum(1)
    H, W = m.shape
    k = 2 * r + 1
    y0, x0 = np.arange(H), np.arange(W)
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    A = ii[Y0, X0]
    B = ii[Y0, X0 + k]
    C = ii[Y0 + k, X0]
    D = ii[Y0 + k, X0 + k]
    return (D - B - C + A) / (k * k)


def spatial_coherence(m, r=2):
    """
    Coherencia espacial de un mapa no negativo, en [0,1].

        coh = E[ smooth(m)^2 ] / E[ m^2 ]

    Una estructura suave (objeto, region de un corte de escena) sobrevive
    al suavizado -> coh cercano a 1. Ruido sal-y-pimienta se cancela al
    promediar localmente -> coh bajo.

    Esta cantidad es ORTOGONAL a la extension: un corte de escena cambia
    el 100% de los pixeles pero es espacialmente coherente, asi que no se
    penaliza; una rafaga de ruido tambien cambia el 100% pero es
    incoherente y si se penaliza.
    """
    m = np.asarray(m, dtype=np.float64)
    denom = float(np.mean(m * m))
    if denom < 1e-15:
        return 0.0
    s = _box_smooth(m, r)
    return float(np.clip(np.mean(s * s) / denom, 0.0, 1.0))


class FrameSignals:
    """Series temporales acumuladas de todas las senales."""

    __slots__ = ("surprisal", "kl", "births", "consolidation", "coherence",
                 "fg_ratio", "n_active", "descriptors", "n_frames")

    def __init__(self):
        self.surprisal, self.kl, self.births = [], [], []
        self.consolidation, self.fg_ratio, self.n_active = {}, [], []
        self.coherence = {}
        self.descriptors = {}
        self.n_frames = 0

    def as_arrays(self):
        n = self.n_frames
        cons = np.zeros(n, dtype=np.float64)
        for i, v in self.consolidation.items():
            if 0 <= i < n:
                cons[i] = v
        coh = np.zeros(n, dtype=np.float64)
        for i, v in self.coherence.items():
            if 0 <= i < n:
                coh[i] = v

        births = np.asarray(self.births, dtype=np.float64)

        # IMPORTANTE: `consolidation` es masa total (peso sobreviviente por
        # pixel promediado sobre TODO el frame), asi que confunde area con
        # persistencia: un destello global consolida poco por pixel pero
        # cubre todo el frame, dando masa comparable a un objeto pequeno y
        # muy persistente. `consolidation_ratio` divide entre el numero de
        # nacimientos -> persistencia media POR NACIMIENTO, independiente
        # del area. Es la cantidad que discrimina transitorio/persistente.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(births > 1e-9, cons / np.maximum(births, 1e-9), 0.0)
        ratio = np.clip(ratio, 0.0, 1.0)

        return {
            "surprisal": np.asarray(self.surprisal, dtype=np.float64),
            "kl": np.asarray(self.kl, dtype=np.float64),
            "births": births,
            "consolidation": cons,
            "consolidation_ratio": ratio,
            "coherence": coh,
            "fg_ratio": np.asarray(self.fg_ratio, dtype=np.float64),
            "n_active": np.asarray(self.n_active, dtype=np.float64),
        }


class SignalAccumulator:
    """Consume FrameReadout y acumula las series por frame."""

    def __init__(self, pool_block=8, descriptor_grid=8, trim=0.9):
        self.pool_block = pool_block
        self.descriptor_grid = descriptor_grid
        self.trim = trim
        self.sig = FrameSignals()

    def _descriptor(self, m):
        """Firma espacial gruesa del mapa, para la seleccion submodular."""
        g = self.descriptor_grid
        H, W = m.shape
        bh, bw = max(1, H // g), max(1, W // g)
        Hc, Wc = bh * g, bw * g
        d = m[:Hc, :Wc].reshape(g, bh, g, bw).mean(axis=(1, 3)).ravel()
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else d

    def push(self, r):
        s = self.sig
        s.surprisal.append(spatial_pool(r.surprisal, self.pool_block, self.trim))
        s.kl.append(spatial_pool(r.kl, self.pool_block, self.trim))
        s.births.append(float(r.births.mean()))
        s.fg_ratio.append(float(r.fg_mask.mean()))
        s.n_active.append(float(r.n_active.mean()))
        s.n_frames = max(s.n_frames, r.index + 1)

        if r.consolidation is not None and r.consolidation_index >= 0:
            i = r.consolidation_index
            s.consolidation[i] = spatial_pool(r.consolidation,
                                              self.pool_block, self.trim)
            s.coherence[i] = spatial_coherence(r.consolidation)
            s.descriptors[i] = self._descriptor(r.consolidation)
        return self


def _robust_norm(x):
    """Normaliza a [0,1] con cuantiles (resistente a outliers)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo, hi = np.quantile(x, 0.01), np.quantile(x, 0.99)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def combine(arrays, w_consolidation=0.6, w_surprisal=0.25, w_kl=0.15,
            gate_power=1.0):
    """
    Score de keyframe por frame.

    La masa de consolidacion se *modula* por el ratio de persistencia:

        score_consolidacion = norm(masa) * norm(ratio)^gate_power

    La masa dice CUANTO cambio; el ratio dice si el cambio era REAL.
    Multiplicarlos suprime eventos de area grande y baja persistencia
    (destellos globales, cambios de iluminacion, artefactos de
    compresion) sin penalizar objetos pequenos y persistentes.

    Ademas se modula por la coherencia espacial, que suprime rafagas de
    ruido incoherente sin penalizar cortes de escena (que son globales
    pero espacialmente estructurados).

    Subir `gate_power` endurece ambos filtros.
    """
    mass = _robust_norm(arrays["consolidation"])
    ratio = _robust_norm(arrays.get("consolidation_ratio",
                                    arrays["consolidation"]))
    coh = np.clip(arrays.get("coherence", np.ones_like(mass)), 0.0, 1.0)
    c = (mass
         * np.power(np.clip(ratio, 0.0, 1.0), gate_power)
         * np.power(coh, gate_power))
    c = _robust_norm(c)

    s = _robust_norm(arrays["surprisal"])
    k = _robust_norm(arrays["kl"])
    total = w_consolidation + w_surprisal + w_kl
    return (w_consolidation * c + w_surprisal * s + w_kl * k) / total
