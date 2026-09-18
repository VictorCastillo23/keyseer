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


def test_extract_keyframes_default_resize_to_is_180x320():
    from keyseer.keyframes import extract_keyframes
    default = inspect.signature(extract_keyframes).parameters["resize_to"].default
    assert default == (180, 320)


def test_extract_keyframes_default_grayscale_and_motion_gate_on():
    """Behavior changes 2/3 del plan, a nivel de la API publica unica."""
    from keyseer.keyframes import extract_keyframes
    params = inspect.signature(extract_keyframes).parameters
    assert params["grayscale"].default is True
    assert params["motion_gate"].default is True
