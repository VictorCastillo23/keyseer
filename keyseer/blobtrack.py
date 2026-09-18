"""
keyseer.blobtrack
==================

Backend de persistencia alternativo, ligero: cv2.createBackgroundSubtractorMOG2
(C++, rapido, probado) + rastreo de blobs por centroide, usando la EDAD del
track como señal de persistencia directamente.

Diferencia de diseño con core.py (el GMM NumPy):
- core.py mide Ψ = peso de una componente L frames DESPUES de nacer
  (requiere recordar birth_time por pixel y esperar la latencia).
- Aqui la edad del track YA ES la señal de persistencia en el instante t;
  no hay que esperar ni recordar nada aparte del track mismo. Es mas simple
  porque el rastreo por centroide ya impone continuidad espacial, que es lo
  que en core.py se obtiene indirectamente vigilando el peso por pixel.

Se exige una edad minima (`min_age_to_count`) para que un track cuente en
el score: es un GATE duro, no una ponderacion suave. Sin el, un destello de
2 frames que cubre casi toda el area del frame pesaria casi tanto como un
objeto real por pura area, igual que el confound de area/persistencia que
se encontro en metrics.py (turno 4) para el modelo KeySeer original.

Estado por frame: O(numero de tracks activos), NO O(H*W). Para escenas
personales (pocos objetos) esto es mucho mas chico que el GMM por pixel.
"""

import time
import numpy as np

__all__ = ["BlobPersistenceTracker", "MotionGatedStride",
           "analyze_video_blobtrack", "extract_keyframes_blobtrack"]


class BlobPersistenceTracker:
    def __init__(self, max_match_dist=40, min_area=60, max_missed=2,
                min_age_to_count=8, age_cap=60, history=500,
                var_threshold=16, morph_kernel=5, max_area_frac=0.5):
        import cv2
        self.cv2 = cv2
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        self.max_match_dist = max_match_dist
        self.min_area = min_area
        self.max_missed = max_missed
        self.min_age_to_count = min_age_to_count
        self.age_cap = age_cap
        self.max_area_frac = max_area_frac
        self._learning_rate = 1.0 / max(history, 1)
        self.tracks = {}       # id -> dict(centroid, area, age, missed)
        self.next_id = 0
        self.frame_shape = None

    def _detect(self, frame):
        cv2 = self.cv2
        # IMPORTANTE: se fuerza una tasa de aprendizaje EXPLICITA en vez de
        # la automatica de OpenCV (learningRate=-1 por defecto). La
        # automatica usa aproximadamente 1/n_frames_vistos al principio del
        # video, mucho mas rapida que 1/history -- esto causaba que un
        # objeto estatico que aparece TEMPRANO en el video se absorbiera al
        # fondo en ~9 frames, mientras el mismo objeto apareciendo tarde
        # sobrevivia >50 frames. Verificado empiricamente (ver ESTADO.md).
        # Con tasa fija, el resultado no depende de cuando aparece el
        # objeto en el video -- critico para clips cortos de uso personal
        # donde el contenido interesante suele empezar casi de inmediato.
        mask = self.back_sub.apply(frame, learningRate=self._learning_rate)
        mask[mask == 127] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        H, W = mask.shape[:2]
        max_area = self.max_area_frac * H * W
        dets = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            # rechaza blobs que cubren una fraccion implausible del frame
            # para UN objeto (ruido global, destellos, cortes de escena):
            # esto no es un objeto, es un evento global de otra naturaleza.
            if self.min_area <= area <= max_area:
                dets.append({"centroid": tuple(centroids[i]), "area": float(area)})
        return dets, mask

    def update(self, frame):
        """Procesa un frame. Devuelve (score, n_active_tracks)."""
        dets, mask = self._detect(frame)
        H, W = mask.shape[:2]
        self.frame_shape = (H, W)

        matched_tracks, matched_dets = set(), set()
        for tid, tr in self.tracks.items():
            best_j, best_d = None, self.max_match_dist
            for j, d in enumerate(dets):
                if j in matched_dets:
                    continue
                dist = float(np.hypot(tr["centroid"][0] - d["centroid"][0],
                                      tr["centroid"][1] - d["centroid"][1]))
                if dist < best_d:
                    best_d, best_j = dist, j
            if best_j is not None:
                tr["centroid"] = dets[best_j]["centroid"]
                tr["area"] = dets[best_j]["area"]
                tr["age"] += 1
                tr["missed"] = 0
                matched_tracks.add(tid)
                matched_dets.add(best_j)

        for tid in list(self.tracks):
            if tid not in matched_tracks:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]

        for j, d in enumerate(dets):
            if j not in matched_dets:
                self.tracks[self.next_id] = {"centroid": d["centroid"],
                                             "area": d["area"], "age": 1,
                                             "missed": 0}
                self.next_id += 1

        frame_area = float(H * W)
        score = sum(
            (tr["area"] / frame_area) * min(tr["age"], self.age_cap)
            for tr in self.tracks.values()
            if tr["age"] >= self.min_age_to_count
        )
        return score, len(self.tracks)

    def state_bytes(self):
        """~200 bytes por track activo (estimado, dict de Python)."""
        return 200 * len(self.tracks)

    def descriptor(self, grid=8):
        """Firma espacial gruesa para deduplicación/diversidad, basada en
        los centroides de tracks que cuentan (misma idea que en metrics.py,
        pero mucho mas barata: no hay mapa por pixel que promediar)."""
        if self.frame_shape is None:
            return np.zeros(grid * grid)
        H, W = self.frame_shape
        d = np.zeros((grid, grid))
        for tr in self.tracks.values():
            if tr["age"] < self.min_age_to_count:
                continue
            cx, cy = tr["centroid"]
            gx = min(grid - 1, int(cx / W * grid))
            gy = min(grid - 1, int(cy / H * grid))
            d[gy, gx] += tr["area"]
        n = np.linalg.norm(d)
        return (d / n).ravel() if n > 1e-9 else d.ravel()


class MotionGatedStride:
    """Gate barato para saltar procesamiento en tramos estaticos: un
    cv2.absdiff(...).mean() sobre un frame gris ya reducido es mucho mas
    barato que un tracker.update() completo (MOG2 + 2x morfologia +
    componentes conectadas) -- esa diferencia es el compute que se ahorra.

    HALLAZGO DE CALIBRACION (ver docs/ESTADO.md seccion 11): un heartbeat
    UNICO no alcanza. Un objeto persistente-pero-ESTATICO (la señal central
    que este proyecto existe para detectar) solo dispara `motion` en su
    frame de aparicion; despues de eso, la escena esta quieta y solo el
    heartbeat vuelve a alimentar tracker.update(). Como `age` solo avanza
    en esas pasadas (no por tiempo real transcurrido), un heartbeat
    calibrado para "escena vacia, es seguro saltar mucho" (p.ej. 8-16
    frames) deja que el objeto tarde `min_age_to_count * heartbeat_interval`
    frames reales en contar -- mas que la ventana tipica de un evento corto,
    causando recall_real=0.5 en la suite sintetica (verificado
    empiricamente). La correccion: el llamador informa via `tracks_active`
    si el tracker tiene tracks vivos ahora mismo, y en ese caso se usa
    `active_heartbeat_interval` (mucho mas chico) en vez de
    `heartbeat_interval` -- manteniendo el heartbeat largo (y el ahorro de
    compute grande) solo para escenas realmente vacias. Calibrado y
    verificado en 24 semillas de evaluacion/harness.py: recall_real=1.0,
    decoy_hits=0.0 con estos defaults, procesando ~25% de los frames.
    """

    def __init__(self, base_stride=1, max_quiet_stride=16, growth_factor=2,
                motion_threshold=2.0, heartbeat_interval=None,
                active_heartbeat_interval=2):
        import cv2
        self.cv2 = cv2
        if max_quiet_stride < base_stride:
            raise ValueError("max_quiet_stride must be >= base_stride")
        self.base_stride = base_stride
        self.max_quiet_stride = max_quiet_stride
        self.growth_factor = growth_factor
        self.motion_threshold = motion_threshold
        self.heartbeat_interval = heartbeat_interval or max_quiet_stride
        self.active_heartbeat_interval = active_heartbeat_interval
        self._prev = None
        self._quiet_streak = 0
        self._gap = base_stride
        self._last_processed_index = 0

    def step(self, gray_probe, real_frame_index, tracks_active=False):
        """Alimenta un probe gris ya decodificado+reducido, su indice real
        de frame, y si el tracker tiene tracks activos ahora mismo. Devuelve
        True sii el llamador debe procesar el frame completo (movimiento,
        heartbeat, o primer frame visto)."""
        interval = self.active_heartbeat_interval if tracks_active else self.heartbeat_interval
        heartbeat_due = (real_frame_index - self._last_processed_index) >= interval
        motion = (self._prev is None or
                 float(self.cv2.absdiff(gray_probe, self._prev).mean())
                 > self.motion_threshold)
        self._prev = gray_probe
        process = motion or heartbeat_due
        if process:
            self._quiet_streak = 0
            self._gap = self.base_stride
            self._last_processed_index = real_frame_index
        else:
            self._quiet_streak += 1
            # el gap de SCHEDULING debe quedar acotado por el heartbeat
            # (el que aplique) restante: sin este limite, el crecimiento
            # geometrico puede programar el proximo probe mas alla de
            # last_processed_index + interval, violando la garantia de "a
            # lo sumo `interval` frames reales entre pasadas completas"
            # (el chequeo de heartbeat_due de arriba solo se evalua AL
            # llegar al probe, no antes de programarlo).
            remaining_to_heartbeat = interval - (real_frame_index - self._last_processed_index)
            self._gap = min(self.max_quiet_stride,
                            self.base_stride * (self.growth_factor ** self._quiet_streak),
                            remaining_to_heartbeat)
        return process

    def gap(self):
        return self._gap


def _iter_video(video_path, resize_to=None, stride=1, grayscale=True,
                motion_gate=None, tracks_active_fn=None):
    """Yields (frame, real_frame_index) pairs. real_frame_index es siempre
    la posicion 0-based en el video FUENTE, sin importar stride/gating.
    Los frames no seleccionados para procesar se cap.grab()-ean (sin
    decodificar) en vez de cap.read() (grab+decode+return).

    `tracks_active_fn`: callable sin argumentos que informa al motion_gate
    si el tracker tiene tracks vivos AHORA MISMO (ver MotionGatedStride);
    None equivale a "nunca hay tracks activos" (comportamiento legacy)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"no se pudo abrir: {video_path}")
    i, next_check = 0, 0
    while True:
        if not cap.grab():
            break
        if i == next_check:
            ok, f = cap.retrieve()
            if not ok:
                break
            if resize_to is not None:
                f = cv2.resize(f, (resize_to[1], resize_to[0]))
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if (grayscale or motion_gate is not None) else None
            out = gray if grayscale else f
            if motion_gate is None:
                yield out, i
                next_check = i + stride
            else:
                tracks_active = tracks_active_fn() if tracks_active_fn else False
                if motion_gate.step(gray, i, tracks_active=tracks_active):
                    yield out, i
                next_check = i + motion_gate.gap()
        i += 1
    cap.release()


def analyze_video_blobtrack(video_path, resize_to=(180, 320), stride=1,
                            tracker_kwargs=None, grayscale=True,
                            motion_gate=True, motion_gate_kwargs=None):
    t0 = time.time()
    tracker = BlobPersistenceTracker(**(tracker_kwargs or {}))
    if motion_gate is True:
        gate = MotionGatedStride(base_stride=stride, **(motion_gate_kwargs or {}))
    elif motion_gate in (False, None):
        gate = None
    else:
        gate = motion_gate
    tracks_active_fn = (lambda: len(tracker.tracks) > 0) if gate is not None else None
    scores, n_active, descriptors, frame_indices = [], [], {}, []
    n = 0
    for f, real_i in _iter_video(video_path, resize_to, stride,
                                 grayscale=grayscale, motion_gate=gate,
                                 tracks_active_fn=tracks_active_fn):
        s, na = tracker.update(f)
        scores.append(s)
        n_active.append(na)
        frame_indices.append(real_i)
        if s > 0:
            descriptors[n] = tracker.descriptor()
        n += 1
    return {
        "scores": np.asarray(scores, dtype=np.float64),
        "n_active": np.asarray(n_active),
        "descriptors": descriptors,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "elapsed": time.time() - t0,
        "n_frames": n,
        "state_bytes": tracker.state_bytes(),
        "peak_state_bytes": 200 * (max(n_active) if n_active else 0),
    }


def extract_keyframes_blobtrack(video_path, budget=6, method="submodular",
                                min_distance=10, resize_to=(180, 320),
                                stride=1, tracker_kwargs=None, grayscale=True,
                                motion_gate=True, motion_gate_kwargs=None):
    from .selection import select_peaks, select_submodular
    r = analyze_video_blobtrack(video_path, resize_to, stride, tracker_kwargs,
                                grayscale=grayscale, motion_gate=motion_gate,
                                motion_gate_kwargs=motion_gate_kwargs)
    if method == "submodular":
        kf = select_submodular(r["scores"], r["descriptors"], budget=budget,
                               min_distance=min_distance)
    elif method == "peaks":
        kf = select_peaks(r["scores"], min_distance=min_distance,
                          prominence=0.0, top_n=budget)
    else:
        raise ValueError(f"metodo desconocido: {method}")
    real_kf = [int(r["frame_indices"][k]) for k in kf]
    return {"keyframes": real_kf, "elapsed": r["elapsed"], "n_frames": r["n_frames"],
            "state_bytes": r["peak_state_bytes"], "scores": r["scores"],
            "frame_indices": r["frame_indices"]}
