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

__all__ = ["BlobPersistenceTracker", "analyze_video_blobtrack",
           "extract_keyframes_blobtrack"]


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
            if resize_to is not None:
                f = cv2.resize(f, (resize_to[1], resize_to[0]))
            yield f
        i += 1
    cap.release()


def analyze_video_blobtrack(video_path, resize_to=(180, 320), stride=1,
                            tracker_kwargs=None):
    t0 = time.time()
    tracker = BlobPersistenceTracker(**(tracker_kwargs or {}))
    scores, n_active, descriptors = [], [], {}
    n = 0
    for f in _iter_video(video_path, resize_to, stride):
        s, na = tracker.update(f)
        scores.append(s)
        n_active.append(na)
        if s > 0:
            descriptors[n] = tracker.descriptor()
        n += 1
    return {
        "scores": np.asarray(scores, dtype=np.float64),
        "n_active": np.asarray(n_active),
        "descriptors": descriptors,
        "elapsed": time.time() - t0,
        "n_frames": n,
        "state_bytes": tracker.state_bytes(),
        "peak_state_bytes": 200 * (max(n_active) if n_active else 0),
    }


def extract_keyframes_blobtrack(video_path, budget=6, method="submodular",
                                min_distance=10, resize_to=(180, 320),
                                stride=1, tracker_kwargs=None):
    from .selection import select_peaks, select_submodular
    r = analyze_video_blobtrack(video_path, resize_to, stride, tracker_kwargs)
    if method == "submodular":
        kf = select_submodular(r["scores"], r["descriptors"], budget=budget,
                               min_distance=min_distance)
    elif method == "peaks":
        kf = select_peaks(r["scores"], min_distance=min_distance,
                          prominence=0.0, top_n=budget)
    else:
        raise ValueError(f"metodo desconocido: {method}")
    return {"keyframes": kf, "elapsed": r["elapsed"], "n_frames": r["n_frames"],
            "state_bytes": r["peak_state_bytes"], "scores": r["scores"]}
