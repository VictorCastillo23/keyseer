import numpy as np
import pytest

from keyseer.blobtrack import _iter_video, analyze_video_blobtrack, MotionGatedStride


# ---------------------------------------------------------------------------
# Phase 1: decode-skip + real-index yielding (grayscale=False, motion_gate=None)
# ---------------------------------------------------------------------------

def _naive_read_every_frame(video_path, resize_to, stride):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            if resize_to is not None:
                f = cv2.resize(f, (resize_to[1], resize_to[0]))
            frames.append(f)
        i += 1
    cap.release()
    return frames


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_iter_video_yields_same_frames_as_before_for_constant_stride(tiny_video, stride):
    video = tiny_video()
    resize_to = (90, 160)
    expected = _naive_read_every_frame(video, resize_to, stride)
    got = [f for f, _ in _iter_video(video, resize_to=resize_to, stride=stride,
                                     grayscale=False)]
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert np.array_equal(a, b)


def test_iter_video_yields_real_frame_index(tiny_video):
    video = tiny_video()
    stride = 4
    indices = [i for _, i in _iter_video(video, resize_to=(90, 160),
                                         stride=stride, grayscale=False)]
    assert indices == list(range(0, indices[-1] + 1, stride))
    assert indices[0] == 0


def test_iter_video_decode_skip_frame_count_matches_grab_calls(monkeypatch, tiny_video):
    """Frames NOT selected for processing must be cap.grab()-ed (no decode)
    instead of cap.read() -- proves decode is actually skipped."""
    import cv2
    video = tiny_video()
    stride = 5

    real_cap_cls = cv2.VideoCapture
    calls = {"grab": 0, "retrieve": 0}

    class CountingCap:
        def __init__(self, path):
            self._cap = real_cap_cls(path)

        def isOpened(self):
            return self._cap.isOpened()

        def grab(self):
            calls["grab"] += 1
            return self._cap.grab()

        def retrieve(self):
            calls["retrieve"] += 1
            return self._cap.retrieve()

        def release(self):
            return self._cap.release()

    monkeypatch.setattr(cv2, "VideoCapture", CountingCap)
    out = list(_iter_video(video, resize_to=(90, 160), stride=stride, grayscale=False))
    monkeypatch.undo()

    # el ultimo grab() de la iteracion falla (EOF) pero igual se contabiliza
    # en el wrapper de conteo -- los frames reales son uno menos.
    n_real_frames = calls["grab"] - 1
    assert calls["retrieve"] == len(out)
    assert calls["retrieve"] < calls["grab"]
    expected_retrieves = len(range(0, n_real_frames, stride))
    assert calls["retrieve"] == expected_retrieves


def test_analyze_video_blobtrack_frame_indices_constant_stride_regression(tiny_video):
    """frame_indices == arange(n)*stride exactly under constant stride, and
    scores/n_active must match the PRE-CHANGE golden baseline exactly
    (tests/fixtures/golden_blobtrack_stride3.npz, captured before blobtrack.py
    was modified for this feature)."""
    import pathlib
    golden = np.load(pathlib.Path(__file__).parent / "fixtures" / "golden_blobtrack_stride3.npz")

    video = tiny_video()
    r = analyze_video_blobtrack(video, resize_to=(180, 320), stride=3,
                                grayscale=False, motion_gate=None)

    assert r["n_frames"] == int(golden["n_frames"])
    np.testing.assert_array_equal(r["frame_indices"], np.arange(r["n_frames"]) * 3)
    np.testing.assert_allclose(r["scores"], golden["scores"])
    np.testing.assert_array_equal(r["n_active"], golden["n_active"])


# ---------------------------------------------------------------------------
# Phase 2: grayscale
# ---------------------------------------------------------------------------

def test_iter_video_grayscale_output_shape_and_dtype(tiny_video):
    video = tiny_video()
    resize_to = (90, 160)
    frame, _ = next(iter(_iter_video(video, resize_to=resize_to, stride=1,
                                     grayscale=True)))
    assert frame.ndim == 2
    assert frame.shape == resize_to
    assert frame.dtype == np.uint8


def test_tracker_update_accepts_grayscale_frame(tiny_video):
    """Verifica que MOG2 acepta entrada 2-D en tiempo de ejecucion (no solo
    que compile) -- BlobPersistenceTracker.update no asume 3 canales."""
    from keyseer.blobtrack import BlobPersistenceTracker
    video = tiny_video()
    tracker = BlobPersistenceTracker()
    for frame, _ in _iter_video(video, resize_to=(90, 160), stride=1, grayscale=True):
        score, n_active = tracker.update(frame)
        assert np.isfinite(score)
        assert n_active >= 0


def test_grayscale_does_not_change_recall_or_decoy_capture(tmp_path):
    from evaluacion.harness import generate_trap_video, score_events
    from keyseer.selection import select_submodular

    video_path = str(tmp_path / "trap.mp4")
    events = generate_trap_video(video_path, seed=0)

    for grayscale in (False, True):
        r = analyze_video_blobtrack(video_path, resize_to=(180, 320), stride=1,
                                    grayscale=grayscale, motion_gate=False)
        kf = select_submodular(r["scores"], r["descriptors"], budget=8,
                               min_distance=10)
        real_kf = [int(r["frame_indices"][k]) for k in kf]
        scored = score_events(real_kf, events, stride=1)
        assert scored["recall_real"] == 1.0, (grayscale, scored)
        assert scored["decoy_hits"] == 0, (grayscale, scored)


# ---------------------------------------------------------------------------
# Phase 4: MotionGatedStride (pure unit tests, no video I/O)
# ---------------------------------------------------------------------------

def _gray(value, shape=(4, 4)):
    return np.full(shape, value, dtype=np.uint8)


def test_motion_gated_stride_first_frame_always_processes():
    gate = MotionGatedStride()
    assert gate.step(_gray(10), 0) is True


def test_motion_gated_stride_gap_sequence_under_sustained_quiet():
    gate = MotionGatedStride(base_stride=1, max_quiet_stride=8,
                             growth_factor=2, motion_threshold=2.0,
                             heartbeat_interval=1000)
    frame = _gray(10)
    processed = gate.step(frame, 0)
    assert processed is True
    assert gate.gap() == 1

    expected_gaps = [2, 4, 8, 8, 8]
    for idx, expected in enumerate(expected_gaps, start=1):
        processed = gate.step(frame, idx)
        assert processed is False
        assert gate.gap() == expected


def test_motion_gated_stride_snap_back_to_base_on_motion():
    gate = MotionGatedStride(base_stride=1, max_quiet_stride=8,
                             growth_factor=2, motion_threshold=2.0,
                             heartbeat_interval=1000)
    frame = _gray(10)
    gate.step(frame, 0)
    for idx in range(1, 4):
        gate.step(frame, idx)
    assert gate.gap() > 1

    moved = gate.step(_gray(80), 4)
    assert moved is True
    assert gate.gap() == 1


def test_motion_gated_stride_motion_threshold_boundary():
    gate = MotionGatedStride(base_stride=1, motion_threshold=2.0,
                             heartbeat_interval=1000)
    gate.step(_gray(10), 0)
    # diff mean == threshold exactamente -> NO cuenta como movimiento (>)
    assert gate.step(_gray(12), 1) is False
    gate2 = MotionGatedStride(base_stride=1, motion_threshold=2.0,
                              heartbeat_interval=1000)
    gate2.step(_gray(10), 0)
    # diff mean > threshold -> SI cuenta como movimiento
    assert gate2.step(_gray(13), 1) is True


def test_motion_gated_stride_heartbeat_forces_processing_after_real_frames():
    gate = MotionGatedStride(base_stride=1, max_quiet_stride=8,
                             motion_threshold=2.0, heartbeat_interval=5)
    frame = _gray(10)
    assert gate.step(frame, 0) is True
    assert gate.step(frame, 4) is False   # 4 - 0 < 5, sigue quieto
    assert gate.step(frame, 5) is True    # 5 - 0 >= 5, heartbeat fuerza


def test_motion_gated_stride_gap_never_exceeds_max_quiet_stride():
    gate = MotionGatedStride(base_stride=1, max_quiet_stride=8,
                             growth_factor=2, motion_threshold=2.0,
                             heartbeat_interval=10_000)
    frame = _gray(10)
    gate.step(frame, 0)
    for idx in range(1, 30):
        gate.step(frame, idx)
        assert gate.gap() <= 8


def test_motion_gated_stride_heartbeat_defaults_to_max_quiet_stride():
    gate = MotionGatedStride(max_quiet_stride=5)
    assert gate.heartbeat_interval == 5


def test_motion_gated_stride_constructor_raises_when_max_quiet_stride_below_base():
    with pytest.raises(ValueError):
        MotionGatedStride(base_stride=4, max_quiet_stride=2)


def test_motion_gated_stride_tracks_active_uses_tighter_heartbeat():
    """Calibracion (docs/ESTADO.md seccion 11): un track PRESENTE PERO
    ESTATICO deja de disparar `motion` tras su frame de aparicion, asi que
    solo el heartbeat sigue alimentando tracker.update() -- y age solo
    avanza en esas pasadas. Con heartbeat_interval=8 fijo, un objeto
    persistente-pero-quieto no alcanza min_age_to_count=8 dentro de la
    ventana tipica de un evento corto (se verifico empiricamente: recall
    cae a 0.5 en la suite sintetica). La correccion es que el gate reciba
    si el tracker tiene tracks activos y use un heartbeat MUCHO mas
    frecuente en ese caso -- sin este mecanismo, motion-gating es
    incompatible con la deteccion de objetos persistentes-pero-estaticos,
    que es la razon de ser del proyecto."""
    gate = MotionGatedStride(base_stride=1, max_quiet_stride=100,
                             motion_threshold=2.0, heartbeat_interval=100,
                             active_heartbeat_interval=3)
    frame = _gray(10)
    gate.step(frame, 0)  # primer frame, siempre procesa

    # tracks_active=False: heartbeat largo (100) sigue rigiendo, quieto en idx=3
    assert gate.step(frame, 3, tracks_active=False) is False

    gate2 = MotionGatedStride(base_stride=1, max_quiet_stride=100,
                              motion_threshold=2.0, heartbeat_interval=100,
                              active_heartbeat_interval=3)
    gate2.step(frame, 0)
    # tracks_active=True: heartbeat corto (3) fuerza procesar en idx=3
    assert gate2.step(frame, 3, tracks_active=True) is True


def test_motion_gated_stride_active_heartbeat_interval_default_is_small():
    gate = MotionGatedStride(max_quiet_stride=32)
    assert gate.active_heartbeat_interval < gate.max_quiet_stride


def test_motion_gated_stride_probe_scheduling_respects_heartbeat_bound():
    """El GAP DE SCHEDULING debe quedar acotado por el heartbeat restante:
    si no, el proximo probe puede programarse mas alla de
    `heartbeat_interval` frames reales desde el ultimo frame procesado,
    violando la garantia documentada en step() ('a lo sumo
    heartbeat_interval frames reales entre pasadas completas')."""
    gate = MotionGatedStride(base_stride=1, max_quiet_stride=8, growth_factor=2,
                             motion_threshold=1000.0, heartbeat_interval=5)
    frame = _gray(10)
    idx = 0
    last_processed = idx
    gate.step(frame, idx)
    for _ in range(20):
        idx += gate.gap()
        assert idx - last_processed <= gate.heartbeat_interval
        if gate.step(frame, idx):
            last_processed = idx


# ---------------------------------------------------------------------------
# Phase 4: MotionGatedStride wired into _iter_video / analyze_video_blobtrack
# ---------------------------------------------------------------------------

def test_iter_video_with_motion_gate_skips_probes_during_quiet(static_then_moving_video):
    video = static_then_moving_video(quiet_before=20, moving=15, quiet_after=20)
    gate = MotionGatedStride(base_stride=1, max_quiet_stride=8,
                             growth_factor=2, motion_threshold=2.0,
                             heartbeat_interval=8)
    yielded = list(_iter_video(video, resize_to=(48, 64), stride=1,
                               grayscale=True, motion_gate=gate))
    yielded_indices = [i for _, i in yielded]

    # muchisimos menos frames procesados que el total del video (20+15+20=55)
    assert len(yielded_indices) < 55
    # el inicio del movimiento (frame 20) no debe perderse: algun frame
    # procesado debe caer dentro de la ventana del evento [20, 35)
    assert any(20 <= i < 35 for i in yielded_indices)
    # los indices reales deben ser estrictamente crecientes
    assert yielded_indices == sorted(set(yielded_indices))


def test_analyze_video_blobtrack_frame_indices_adaptive_stride_are_monotonic_and_bounded(
        static_then_moving_video):
    video = static_then_moving_video(quiet_before=30, moving=15, quiet_after=30)
    gate = MotionGatedStride(base_stride=1, max_quiet_stride=8)
    r = analyze_video_blobtrack(video, resize_to=(48, 64), stride=1,
                               motion_gate=gate)
    fi = r["frame_indices"]
    assert len(fi) == r["n_frames"]
    assert np.all(np.diff(fi) > 0)
    assert fi[0] == 0
    assert fi[-1] < 75


def test_motion_gate_defaults_keep_recall_on_trap_suite(tmp_path):
    """Regresion directa del bug de calibracion: con motion_gate=True (los
    defaults de produccion), la suite sintetica de senuelos -- que incluye
    un rectangulo persistente pero ESTATICO tras aparecer -- debe seguir
    dando recall_real==1.0 y decoy_hits==0, igual que con motion_gate=False.
    Sin el heartbeat active-aware (tracks_active), este test falla con
    recall_real==0.5 (ver docs/ESTADO.md seccion 11)."""
    from evaluacion.harness import generate_trap_video, score_events
    from keyseer.selection import select_submodular

    for seed in range(4):
        video_path = str(tmp_path / f"trap_{seed}.mp4")
        events = generate_trap_video(video_path, seed=seed)
        r = analyze_video_blobtrack(video_path, resize_to=(180, 320), stride=1,
                                    motion_gate=True)
        kf = select_submodular(r["scores"], r["descriptors"], budget=8,
                               min_distance=10)
        real_kf = [int(r["frame_indices"][k]) for k in kf]
        scored = score_events(real_kf, events, stride=1)
        assert scored["recall_real"] == 1.0, (seed, scored)
        assert scored["decoy_hits"] == 0, (seed, scored)


def test_blobtrack_entry_points_default_grayscale_and_motion_gate_on():
    """Behavior changes 2/3 del plan: grayscale=True y motion_gate=True son
    el default en todos los puntos de entrada respaldados por blobtrack,
    tras la calibracion de MotionGatedStride (docs/ESTADO.md seccion 11)."""
    import inspect
    from keyseer.blobtrack import extract_keyframes_blobtrack

    for fn in (analyze_video_blobtrack, extract_keyframes_blobtrack):
        params = inspect.signature(fn).parameters
        assert params["grayscale"].default is True, fn
        assert params["motion_gate"].default is True, fn
