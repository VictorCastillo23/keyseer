"""
evaluacion.harness
===================

Suite de senuelos con ground truth y funciones de puntuacion para el
estudio comparativo (Opcion C). Formaliza el vídeo ad hoc usado durante el
desarrollo de KeySeer en un generador parametrizado y reutilizable.
"""

import numpy as np

__all__ = ["generate_trap_video", "score_events", "pairwise_redundancy",
           "TrapEvent"]


class TrapEvent:
    __slots__ = ("start", "end", "label", "is_real")

    def __init__(self, start, end, label, is_real):
        self.start, self.end, self.label, self.is_real = start, end, label, is_real

    def __repr__(self):
        tag = "REAL" if self.is_real else "SEÑUELO"
        return f"<{tag} {self.label} [{self.start},{self.end})>"


def generate_trap_video(path, W=320, H=240, fps=25, seed=0,
                        extra_transient_lookalike=True):
    """
    Genera un video con eventos reales y señuelos deliberados, y devuelve
    la lista de eventos con ground truth. Extiende el vídeo de evaluación
    usado durante el desarrollo (turno 3 de esta conversación) con un
    señuelo adicional: un transitorio breve con la MISMA apariencia que un
    evento real (para separar "detecta algo ahí" de "sabe que es efímero").
    """
    import cv2
    rng = np.random.default_rng(seed)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    events = []
    t = 0

    def bg(n):
        for _ in range(n):
            f = np.full((H, W, 3), 45, np.uint8)
            f = cv2.add(f, rng.integers(0, 4, (H, W, 3), dtype=np.uint8))
            out.write(f)

    def advance(n, label, is_real, draw):
        nonlocal t
        s = t
        for i in range(n):
            f = np.full((H, W, 3), 45, np.uint8)
            draw(f, i)
            f = cv2.add(f, rng.integers(0, 4, (H, W, 3), dtype=np.uint8))
            out.write(f)
        t += n
        events.append(TrapEvent(s, s + n, label, is_real))

    bg(60); t += 60
    advance(50, "moviendose_verde", True,
           lambda f, i: cv2.circle(f, (40 + i * 4, 120), 22, (0, 190, 0), -1))
    bg(60); t += 60
    advance(2, "destello_global", False, lambda f, i: f.__setitem__(slice(None), 225))
    bg(30); t += 30
    advance(15, "ruido_denso", False,
           lambda f, i: f.__setitem__(slice(None), cv2.add(
               f, rng.integers(0, 110, f.shape, dtype=np.uint8))))
    bg(60); t += 60
    advance(80, "rectangulo_persistente", True,
           lambda f, i: cv2.rectangle(f, (200, 60), (280, 140), (0, 0, 200), -1))
    bg(40); t += 40
    if extra_transient_lookalike:
        advance(4, "rectangulo_transitorio_gemelo", False,
               lambda f, i: cv2.rectangle(f, (200, 60), (280, 140), (0, 0, 200), -1))
        bg(40); t += 40
    advance(70, "doble_evento", True,
           lambda f, i: (cv2.circle(f, (60 + i * 3, 80), 18, (200, 0, 0), -1),
                        cv2.rectangle(f, (240 - i * 2, 170), (280 - i * 2, 205),
                                     (0, 200, 200), -1)))
    bg(30); t += 30

    out.release()
    return events


def score_events(keyframes, events, stride=1):
    """
    keyframes: indices (en unidades de frame MUESTREADO, multiplicar por
    stride para comparar contra `events`, que estan en frames originales).

    Devuelve:
      recall_real: fraccion de eventos reales con >=1 keyframe dentro
      decoy_hits: numero de señuelos con >=1 keyframe dentro (cuanto mas
                  bajo, mejor)
      unassigned: keyframes que no caen en ningun evento (ni real ni señuelo)
    """
    kf = [k * stride for k in keyframes]
    real = [e for e in events if e.is_real]
    decoys = [e for e in events if not e.is_real]

    def hit(e):
        return any(e.start <= k < e.end for k in kf)

    recall = sum(hit(e) for e in real) / max(len(real), 1)
    decoy_hits = sum(hit(e) for e in decoys)
    assigned = sum(1 for k in kf if any(e.start <= k < e.end for e in events))
    return {
        "recall_real": recall,
        "decoy_hits": decoy_hits,
        "n_decoys": len(decoys),
        "unassigned": len(kf) - assigned,
        "n_keyframes": len(kf),
    }


def pairwise_redundancy(video_path, keyframes, resize_to=(60, 80), stride=1):
    """Correlacion de histograma HSV media entre todos los pares de
    keyframes seleccionados. Mas alto = mas redundante (peor)."""
    import cv2
    if len(keyframes) < 2:
        return 0.0
    cap = cv2.VideoCapture(video_path)
    hists = []
    for k in keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k) * stride)
        ok, f = cap.read()
        if not ok:
            continue
        if resize_to is not None:
            f = cv2.resize(f, (resize_to[1], resize_to[0]))
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        hists.append(h)
    cap.release()
    if len(hists) < 2:
        return 0.0
    corrs = []
    for i in range(len(hists)):
        for j in range(i + 1, len(hists)):
            corrs.append(cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL))
    return float(np.mean(corrs))
