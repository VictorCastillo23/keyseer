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
