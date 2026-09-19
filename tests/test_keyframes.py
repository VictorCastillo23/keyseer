import inspect

import numpy as np


# ---------------------------------------------------------------------------
# Phase 3: frame-index plumbing through keyframes.py (motion_gate still off)
# ---------------------------------------------------------------------------

def test_extract_keyframes_blobtrack_keyframes_are_real_frame_numbers(tiny_video):
    from keyseer.blobtrack import analyze_video_blobtrack, extract_keyframes_blobtrack

    video = tiny_video()
    stride = 3
    r = analyze_video_blobtrack(video, resize_to=(180, 320), stride=stride,
                                motion_gate=None)
    result = extract_keyframes_blobtrack(video, budget=4, resize_to=(180, 320),
                                        stride=stride, motion_gate=None)

    assert len(result["keyframes"]) > 0
    real_positions = set(int(x) for x in r["frame_indices"])
    for k in result["keyframes"]:
        assert k in real_positions
        assert k % stride == 0


def test_keyframes_extract_keyframes_save_frames_uses_real_indices(tiny_video, tmp_path):
    """Regresion directa contra la clase de bug de doble multiplicacion:
    _save_frames ya NO debe recibir `stride` -- `indices` son numeros de
    frame reales, no posiciones densas."""
    import cv2
    from keyseer.keyframes import _save_frames

    video = tiny_video()
    real_indices = [5, 47, 90]
    out_dir = str(tmp_path / "out")

    assert "stride" not in inspect.signature(_save_frames).parameters

    paths = _save_frames(video, real_indices, out_dir)

    assert len(paths) == len(real_indices)
    cap = cv2.VideoCapture(video)
    for real_idx, path in zip(real_indices, paths):
        assert f"{real_idx:06d}" in path
        cap.set(cv2.CAP_PROP_POS_FRAMES, real_idx)
        ok, expected = cap.read()
        assert ok
        got = cv2.imread(path)
        # tolerancia por compresion JPEG (con perdida) al guardar -- una
        # diferencia grande indicaria el frame EQUIVOCADO (bug de indice),
        # no ruido de compresion.
        assert got.shape == expected.shape
        mean_abs_diff = np.abs(got.astype(np.int16) - expected.astype(np.int16)).mean()
        assert mean_abs_diff < 3.0, mean_abs_diff
    cap.release()


def test_keyframes_extract_keyframes_timestamps_match_real_frame_numbers(tiny_video):
    from keyseer.keyframes import extract_keyframes

    for stride in (1, 3):
        video = tiny_video()
        result = extract_keyframes(video, budget=4, backend="blobtrack",
                                   stride=stride, motion_gate=False)
        assert result.fps_source is not None
        timestamps = result.timestamps()
        for idx, ts in zip(result.keyframe_indices, timestamps):
            assert ts == idx / result.fps_source


def test_extract_keyframes_gmm_backend_unaffected(tiny_video):
    """Smoke test tras el cambio de firma de _run_backend (ahora devuelve
    un 4-tuple con frame_indices=None para el backend gmm)."""
    from keyseer.keyframes import extract_keyframes

    video = tiny_video(n_frames=30, event=(10, 20))
    stride = 2
    result = extract_keyframes(video, budget=2, backend="gmm",
                               resize_to=(60, 80), stride=stride)
    assert isinstance(result.keyframe_indices, list)
    for k in result.keyframe_indices:
        assert isinstance(k, int)
        assert k % stride == 0


def test_blobtrack_default_resize_constant_value():
    from keyseer.blobtrack import DEFAULT_RESIZE_TO
    assert DEFAULT_RESIZE_TO == (180, 320)


def test_blobtrack_entry_points_default_to_the_shared_constant():
    from keyseer.blobtrack import (DEFAULT_RESIZE_TO, analyze_video_blobtrack,
                                   extract_keyframes_blobtrack)
    for fn in (analyze_video_blobtrack, extract_keyframes_blobtrack):
        assert inspect.signature(fn).parameters["resize_to"].default is DEFAULT_RESIZE_TO


def test_benchmark_defaults_to_the_shared_constant():
    from evaluacion import benchmark as bm
    from keyseer.blobtrack import DEFAULT_RESIZE_TO
    for fn in (bm.analyze, bm.run_one, bm.run_suite, bm.run_weighting_experiment):
        assert inspect.signature(fn).parameters["resize_to"].default is DEFAULT_RESIZE_TO


def test_extract_keyframes_resize_to_defaults_to_none_meaning_per_backend():
    from keyseer.keyframes import extract_keyframes
    assert inspect.signature(extract_keyframes).parameters["resize_to"].default is None


def _spy_resize_to(monkeypatch, backend):
    """Envuelve (call-through) el analizador real del backend y registra el
    `resize_to` con que lo invoca extract_keyframes."""
    seen = []
    if backend == "blobtrack":
        import keyseer.blobtrack as mod
        name = "analyze_video_blobtrack"
    else:
        import keyseer.pipeline as mod
        name = "analyze_video"
    real = getattr(mod, name)

    def spy(*a, **kw):
        seen.append(kw.get("resize_to"))
        return real(*a, **kw)

    monkeypatch.setattr(mod, name, spy)
    return seen


def test_extract_keyframes_none_resize_resolves_per_backend(tiny_video, monkeypatch):
    from keyseer.blobtrack import DEFAULT_RESIZE_TO
    from keyseer.keyframes import extract_keyframes

    video = tiny_video(n_frames=12, event=(4, 10))
    seen_bt = _spy_resize_to(monkeypatch, "blobtrack")
    extract_keyframes(video, budget=2, backend="blobtrack")
    seen_gmm = _spy_resize_to(monkeypatch, "gmm")
    extract_keyframes(video, budget=2, backend="gmm")

    assert seen_bt == [DEFAULT_RESIZE_TO]
    assert seen_gmm == [(180, 320)]


def test_extract_keyframes_explicit_resize_to_is_honoured_by_both_backends(
        tiny_video, monkeypatch):
    from keyseer.keyframes import extract_keyframes

    video = tiny_video(n_frames=12, event=(4, 10))
    seen_bt = _spy_resize_to(monkeypatch, "blobtrack")
    extract_keyframes(video, budget=2, backend="blobtrack", resize_to=(90, 160))
    seen_gmm = _spy_resize_to(monkeypatch, "gmm")
    extract_keyframes(video, budget=2, backend="gmm", resize_to=(60, 80))

    assert seen_bt == [(90, 160)]
    assert seen_gmm == [(60, 80)]


def test_extract_keyframes_default_grayscale_and_motion_gate_on():
    """Behavior changes 2/3 del plan, a nivel de la API publica unica."""
    from keyseer.keyframes import extract_keyframes
    params = inspect.signature(extract_keyframes).parameters
    assert params["grayscale"].default is True
    assert params["motion_gate"].default is True


# ---------------------------------------------------------------------------
# Phase 4: dense_positions field on KeyframeResult
# ---------------------------------------------------------------------------

def test_extract_keyframes_dense_positions_pair_with_blobtrack_frame_indices(tiny_video):
    """dense_positions[i] must be the dense index whose frame_indices value
    equals keyframe_indices[i], for the blobtrack backend under a uniform
    stride (motion_gate off)."""
    from keyseer.blobtrack import analyze_video_blobtrack
    from keyseer.keyframes import extract_keyframes

    video = tiny_video()
    stride = 3
    r = analyze_video_blobtrack(video, resize_to=(180, 320), stride=stride,
                                motion_gate=None)
    result = extract_keyframes(video, budget=4, backend="blobtrack",
                               stride=stride, resize_to=(180, 320),
                               motion_gate=False)

    dp = result.dense_positions
    ki = result.keyframe_indices
    frame_indices = r["frame_indices"]

    assert len(dp) == len(ki)
    assert len(dp) > 0
    for i, p in enumerate(dp):
        assert type(p) is int
        assert 0 <= p < len(result.scores)
        assert int(frame_indices[p]) == ki[i]


def test_extract_keyframes_dense_positions_gmm_match_stride(tiny_video):
    """For the gmm backend (no frame_indices mapping), dense_positions[i]
    must satisfy dense_positions[i] * stride == keyframe_indices[i]."""
    from keyseer.keyframes import extract_keyframes

    video = tiny_video(n_frames=30, event=(10, 20))
    stride = 2
    result = extract_keyframes(video, budget=2, backend="gmm",
                               resize_to=(60, 80), stride=stride)

    dp = result.dense_positions
    ki = result.keyframe_indices

    assert len(dp) == len(ki)
    assert len(dp) > 0
    for i, p in enumerate(dp):
        assert type(p) is int
        assert 0 <= p < len(result.scores)
        assert p * stride == ki[i]


def test_extract_keyframes_dense_positions_hold_under_motion_gate(tiny_video):
    """Under motion-gated adaptive stride (non-uniform frame_indices), the
    pairing invariant and ordering must still hold. Uses the public
    `analyze_video_blobtrack` entry point (not `MotionGatedStride`
    internals) with matching params to independently obtain the real
    `frame_indices` mapping and check the exact pairing formula, the same
    way the uniform-stride test above does."""
    from keyseer.blobtrack import analyze_video_blobtrack
    from keyseer.keyframes import extract_keyframes

    video = tiny_video()
    stride = 3
    r = analyze_video_blobtrack(video, resize_to=(180, 320), stride=stride,
                                motion_gate=True)
    result = extract_keyframes(video, budget=4, backend="blobtrack",
                               stride=stride, resize_to=(180, 320),
                               motion_gate=True)

    dp = result.dense_positions
    ki = result.keyframe_indices
    frame_indices = r["frame_indices"]

    assert len(dp) == len(ki)
    assert len(dp) > 0
    for i, p in enumerate(dp):
        assert type(p) is int
        assert 0 <= p < len(result.scores)
        assert int(frame_indices[p]) == ki[i]
    assert all(dp[i] < dp[i + 1] for i in range(len(dp) - 1))
    assert all(ki[i] < ki[i + 1] for i in range(len(ki) - 1))


def test_extract_keyframes_dense_positions_empty_when_zero_selected(tiny_video):
    """Scenario: Empty selection yields empty dense positions. A budget of
    zero forces select_submodular to return zero keyframes (see
    selection.py: `if not idxs or budget <= 0: return []`), which must
    propagate to both keyframe_indices and dense_positions as empty
    lists."""
    from keyseer.keyframes import extract_keyframes

    video = tiny_video()
    result = extract_keyframes(video, budget=0, backend="blobtrack",
                               stride=3, resize_to=(180, 320),
                               motion_gate=False)

    assert result.keyframe_indices == []
    assert result.dense_positions == []


def test_gmm_resize_constant_matches_pipeline_default():
    from keyseer.keyframes import GMM_RESIZE_TO
    from keyseer.pipeline import analyze_video
    assert GMM_RESIZE_TO == inspect.signature(analyze_video).parameters["resize_to"].default
