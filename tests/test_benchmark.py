def test_run_one_smoke(tmp_path):
    from evaluacion.harness import generate_trap_video
    from evaluacion.benchmark import run_one, CONFIGS

    video_path = str(tmp_path / "trap.mp4")
    events = generate_trap_video(video_path, seed=0)

    result = run_one(video_path, events, CONFIGS["baseline"], budget=8)

    assert result["recall_real"] == 1.0
    assert "fps" in result
    assert result["n_frames_processed"] > 0


def test_run_suite_smoke():
    from evaluacion.benchmark import run_suite, CONFIGS

    results = run_suite(seeds=range(1), configs={"baseline": CONFIGS["baseline"]})

    assert set(results.keys()) == {"baseline"}
    assert len(results["baseline"]) == 1
    assert results["baseline"][0]["recall_real"] == 1.0


def _read_frame_count(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def _event(events, label):
    return next(e for e in events if e.label == label)


def _peak_in(r, ev):
    # motion_gate=False -> frame_indices == 0..n-1, indexable por frame real
    return float(r["scores"][ev.start:ev.end].max())


def test_scale_contrast_generator_returns_expected_events(tmp_path):
    from evaluacion.harness import generate_scale_contrast_video

    video_path = str(tmp_path / "scale.mp4")
    events = generate_scale_contrast_video(video_path, seed=0)

    assert [e.label for e in events if e.is_real] == [
        "objeto_pequeno_persistente", "movimiento_grande_continuo"]
    assert [e.label for e in events if not e.is_real] == ["destello_global"]
    ordered = sorted(events, key=lambda e: e.start)
    for prev, nxt in zip(ordered, ordered[1:]):
        assert prev.end <= nxt.start
    assert _read_frame_count(video_path) >= max(e.end for e in events)


def test_scale_contrast_default_timeline_is_unchanged(tmp_path):
    from evaluacion.harness import generate_scale_contrast_video

    events = generate_scale_contrast_video(str(tmp_path / "s.mp4"), seed=0)

    assert [(e.label, e.start, e.end) for e in events] == [
        ("objeto_pequeno_persistente", 60, 100),
        ("movimiento_grande_continuo", 140, 300),
        ("destello_global", 340, 342)]


def test_scale_contrast_gate_passing_decoys_are_extra_labelled_decoys(tmp_path):
    from evaluacion.harness import generate_scale_contrast_video

    base = generate_scale_contrast_video(str(tmp_path / "a.mp4"), seed=0)
    video_path = str(tmp_path / "b.mp4")
    events = generate_scale_contrast_video(video_path, seed=0,
                                           gate_passing_decoys=True)

    assert [(e.label, e.start, e.end, e.is_real) for e in events[:len(base)]] == \
        [(e.label, e.start, e.end, e.is_real) for e in base]
    extra = events[len(base):]
    assert [e.label for e in extra] == ["mota_pequena_transitoria",
                                        "parche_de_luz_grande"]
    assert not any(e.is_real for e in extra)
    ordered = sorted(events, key=lambda e: e.start)
    for prev, nxt in zip(ordered, ordered[1:]):
        assert prev.end <= nxt.start
    assert _read_frame_count(video_path) >= max(e.end for e in events)


def test_scale_contrast_geometry_scales_with_resolution(tmp_path):
    import cv2
    from evaluacion.harness import generate_scale_contrast_video

    video_path = str(tmp_path / "hi.mp4")
    W, H = 640, 480
    events = generate_scale_contrast_video(video_path, W=W, H=H, seed=0)
    small = _event(events, "objeto_pequeno_persistente")

    cap = cv2.VideoCapture(video_path)
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == W
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == H
    cap.set(cv2.CAP_PROP_POS_FRAMES, small.start + 20)
    ok, inside = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ok2, empty = cap.read()
    cap.release()

    assert ok and ok2
    # centro del cuadrado pequeno: (209, 159) en coordenadas de referencia 320x240
    cx, cy = 209 * W // 320, 159 * H // 240
    assert inside[cy, cx, 1] > 150 and inside[cy, cx, 0] < 60
    assert empty[cy, cx].max() < 80


def _gate_passing_analysis(tmp_path, config_name, **video_kw):
    from evaluacion.harness import generate_scale_contrast_video
    from evaluacion.benchmark import analyze, CONFIGS

    video_path = str(tmp_path / f"decoys_{config_name}.mp4")
    events = generate_scale_contrast_video(video_path, seed=0,
                                           gate_passing_decoys=True, **video_kw)
    return events, analyze(video_path, CONFIGS[config_name])


def test_gate_passing_decoys_have_score_and_descriptors_baseline(tmp_path):
    events, r = _gate_passing_analysis(tmp_path, "baseline")

    for label in ("mota_pequena_transitoria", "parche_de_luz_grande"):
        ev = _event(events, label)
        window = range(ev.start, ev.end)
        assert any(r["scores"][i] > 0 for i in window), label
        assert any(i in r["descriptors"] for i in window), label


def test_gate_passing_decoys_have_score_and_descriptors_aggressive(tmp_path):
    events, r = _gate_passing_analysis(tmp_path, "aggressive")

    for label in ("mota_pequena_transitoria", "parche_de_luz_grande"):
        ev = _event(events, label)
        dense = [n for n, real in enumerate(r["frame_indices"])
                 if ev.start <= real < ev.end]
        assert any(r["scores"][n] > 0 for n in dense), label
        assert any(n in r["descriptors"] for n in dense), label


def test_scale_contrast_score_contrast_large_over_small(tmp_path):
    from evaluacion.harness import generate_scale_contrast_video
    from keyseer.blobtrack import analyze_video_blobtrack

    video_path = str(tmp_path / "scale.mp4")
    events = generate_scale_contrast_video(video_path, seed=0)
    r = analyze_video_blobtrack(video_path, resize_to=(180, 320),
                                motion_gate=False)

    small = _peak_in(r, _event(events, "objeto_pequeno_persistente"))
    large = _peak_in(r, _event(events, "movimiento_grande_continuo"))

    assert small > 0.0
    assert large >= 8.0 * small


def test_apply_weighting_identity_and_binary():
    import numpy as np
    from evaluacion.benchmark import apply_weighting

    s = np.array([0.0, 0.3, 4.1, 0.0])

    assert apply_weighting(s, None) is s
    assert apply_weighting(s, "identity") is s
    assert apply_weighting(s, "binary").tolist() == [0.0, 1.0, 1.0, 0.0]


def test_apply_weighting_rejects_unknown_mode():
    import numpy as np
    import pytest
    from evaluacion.benchmark import apply_weighting

    with pytest.raises(ValueError):
        apply_weighting(np.zeros(3), "sqrt")


def test_run_one_binary_weighting_same_keys_and_default_unchanged(tmp_path):
    from evaluacion.harness import generate_trap_video
    from evaluacion.benchmark import run_one, CONFIGS

    video_path = str(tmp_path / "trap.mp4")
    events = generate_trap_video(video_path, seed=0)

    default = run_one(video_path, events, CONFIGS["baseline"], budget=8)
    identity = run_one(video_path, events, CONFIGS["baseline"], budget=8,
                       weighting="identity")
    binary = run_one(video_path, events, CONFIGS["baseline"], budget=8,
                     weighting="binary")

    assert set(binary.keys()) == set(default.keys())
    for k in ("recall_real", "decoy_hits", "n_keyframes", "unassigned"):
        assert identity[k] == default[k]


def test_binary_weighting_selects_small_event_at_small_budget(tmp_path):
    from evaluacion.harness import generate_scale_contrast_video
    from evaluacion.benchmark import analyze, select_keyframes, CONFIGS

    video_path = str(tmp_path / "scale.mp4")
    events = generate_scale_contrast_video(video_path, seed=0)
    small = _event(events, "objeto_pequeno_persistente")
    r = analyze(video_path, CONFIGS["baseline"])

    def hits_small(weighting):
        kf = select_keyframes(r, budget=1, weighting=weighting)
        return any(small.start <= k < small.end for k in kf)

    assert hits_small("binary")
    assert not hits_small("identity")


def test_run_suite_accepts_scale_contrast_scenario():
    from evaluacion.benchmark import run_suite, CONFIGS

    results = run_suite(seeds=range(1), configs={"baseline": CONFIGS["baseline"]},
                        budget=8, scenario="scale_contrast")

    assert set(results.keys()) == {"baseline"}
    assert results["baseline"][0]["recall_real"] == 1.0


def test_run_suite_rejects_unknown_scenario():
    import pytest
    from evaluacion.benchmark import run_suite

    with pytest.raises(ValueError):
        run_suite(seeds=range(1), scenario="inexistente")


def test_weighting_experiment_analyzes_once_per_video_and_config(monkeypatch):
    import evaluacion.benchmark as bm

    calls = []
    real_analyze = bm.analyze_video_blobtrack

    def counting(*a, **kw):
        calls.append(1)
        return real_analyze(*a, **kw)

    monkeypatch.setattr(bm, "analyze_video_blobtrack", counting)

    rows = bm.run_weighting_experiment(
        seeds=range(1),
        scenarios=("trap", "scale_contrast"),
        weightings=("identity", "binary"),
        budgets=(3, 8),
        configs={"baseline": bm.CONFIGS["baseline"]})

    assert len(calls) == 2                      # 2 escenarios x 1 seed x 1 config
    assert len(rows) == 2 * 2 * 2 * 1           # escenario x weighting x budget x config
    for row in rows:
        assert {"scenario", "weighting", "budget", "config", "recall_real",
                "decoy_hits", "fps", "n_seeds"} <= set(row)
        assert row["n_seeds"] == 1


def test_scale_contrast_decoys_scenario_registered_and_has_decoys(tmp_path):
    from evaluacion.benchmark import SCENARIOS

    events = SCENARIOS["scale_contrast_decoys"](str(tmp_path / "d.mp4"), seed=0)

    assert {"mota_pequena_transitoria", "parche_de_luz_grande"} <= {
        e.label for e in events if not e.is_real}


def test_resolution_experiment_analyzes_once_per_video_config_and_size(monkeypatch):
    import evaluacion.benchmark as bm

    calls = []
    real_analyze = bm.analyze_video_blobtrack

    def counting(path, resize_to=None, **kw):
        calls.append(resize_to)
        return real_analyze(path, resize_to=resize_to, **kw)

    monkeypatch.setattr(bm, "analyze_video_blobtrack", counting)

    rows = bm.run_resolution_experiment(
        seeds=range(1),
        sizes=((90, 160), (180, 320)),
        scenarios={"trap": bm.generate_trap_video},
        weightings=("identity", "binary"),
        budgets=(3, 8),
        configs={"baseline": bm.CONFIGS["baseline"]})

    assert sorted(calls) == [(90, 160), (180, 320)]     # 1 video x 1 config x 2 tamanos
    assert len(rows) == 1 * 2 * 1 * 2 * 2               # escenario x tamano x config x weighting x budget
    for row in rows:
        assert {"scenario", "size", "config", "weighting", "budget", "recall_real",
                "decoy_hits", "fps", "elapsed", "n_seeds"} <= set(row)
        assert row["size"] in ((90, 160), (180, 320))
        assert row["n_seeds"] == 1


def test_resolution_experiment_default_grid_matches_protocol():
    import evaluacion.benchmark as bm

    assert bm.RESOLUTION_SIZES == ((180, 320), (270, 480), (360, 640), (540, 960))
    assert set(bm.RESOLUTION_SCENARIOS) == {"trap", "scale_contrast_hd"}
    hd = bm.RESOLUTION_SCENARIOS["scale_contrast_hd"]
    assert (hd.keywords["W"], hd.keywords["H"]) == (1280, 720)
    assert hd.keywords["gate_passing_decoys"] is True


def test_format_experiment_shows_size_column_when_present():
    from evaluacion.benchmark import format_experiment

    row = {"scenario": "trap", "size": (270, 480), "weighting": "binary", "budget": 5,
           "config": "baseline", "recall_real": 1.0, "decoy_hits": 0.0, "fps": 123.0}

    assert "270x480" in format_experiment([row])
