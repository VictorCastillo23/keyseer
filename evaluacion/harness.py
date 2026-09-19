"""
evaluacion.harness
===================

Suite de senuelos con ground truth y funciones de puntuacion para el
estudio comparativo (Opcion C). Formaliza el vídeo ad hoc usado durante el
desarrollo de KeySeer en un generador parametrizado y reutilizable.
"""

import numpy as np

__all__ = ["generate_trap_video", "generate_scale_contrast_video",
           "score_events", "pairwise_redundancy", "TrapEvent"]


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


# Con 12 frames ninguno pasa el config "aggressive": el gate de movimiento solo
# re-procesa cada ~16 frames (heartbeat) y luego cada 2 (tracks activos), asi
# que un evento estatico corto no llega a min_age_to_count=8 pasadas.
# Medido: score>0 con aggressive exige >=28 frames (mota) y >=20 (parche).
MOTA_FRAMES = 32
PARCHE_FRAMES = 24


def generate_scale_contrast_video(path, W=320, H=240, fps=25, seed=0,
                                  gate_passing_decoys=False):
    """
    Un evento real PEQUEÑO y corto (cuadrado estatico) frente a un evento real
    GRANDE y largo (circulo en movimiento continuo), mas un destello global
    como señuelo. Existe porque generate_trap_video solo mezcla eventos
    reales de tamaño parecido: con eso, el sesgo de escala del score de
    blobtrack (area * edad) y de select_submodular (pondera cobertura por
    score) es invisible. Aqui el score pico del evento grande es >=8x el del
    pequeño (ver tests/test_benchmark.py), asi que un presupuesto chico de
    keyframes tiende a gastarse entero en posiciones del evento grande.

    Toda la geometria esta escrita en coordenadas de referencia 320x240 y se
    escala a (W, H), asi que el escenario se puede generar a cualquier
    resolucion (a 320x240 los pixeles coinciden con la version original).

    `gate_passing_decoys=True` agrega DESPUES de los eventos anteriores (sin
    moverlos) dos señuelos que SI pasan los gates de blobtrack (score > 0,
    con descriptor), a diferencia de los de generate_trap_video, cuyo score
    es exactamente 0:
      - mota_pequena_transitoria: blob de ~1.5e-3 del frame, breve pero mas
        largo que min_age_to_count. Con pesos binarios pesa igual que un
        evento real (riesgo: magnitud ignorada).
      - parche_de_luz_grande: parche localizado de ~37% del frame (bajo
        max_area_frac=0.5). Con pesos identidad su area*edad domina
        (riesgo opuesto: magnitud dominante).
    """
    import cv2
    rng = np.random.default_rng(seed)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    events = []
    t = 0
    sx, sy = W / 320.0, H / 240.0

    def X(v):
        return int(round(v * sx))

    def Y(v):
        return int(round(v * sy))

    radius = max(1, int(round(30 * min(sx, sy))))

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
    advance(40, "objeto_pequeno_persistente", True,
           lambda f, i: cv2.rectangle(f, (X(200), Y(150)), (X(218), Y(168)),
                                     (0, 190, 0), -1))
    bg(40); t += 40
    advance(160, "movimiento_grande_continuo", True,
           lambda f, i: cv2.circle(f, (X(40 + int(i * 1.5)), Y(120)), radius,
                                  (0, 190, 0), -1))
    bg(40); t += 40
    advance(2, "destello_global", False, lambda f, i: f.__setitem__(slice(None), 225))
    bg(30); t += 30
    if gate_passing_decoys:
        advance(MOTA_FRAMES, "mota_pequena_transitoria", False,
               lambda f, i: cv2.rectangle(f, (X(272), Y(30)), (X(283), Y(41)),
                                         (0, 190, 0), -1))
        bg(30); t += 30
        advance(PARCHE_FRAMES, "parche_de_luz_grande", False,
               lambda f, i: cv2.rectangle(f, (0, 0), (X(200), Y(144)),
                                         (200, 200, 200), -1))
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
