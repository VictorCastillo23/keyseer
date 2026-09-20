# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

KeySeer: keyframe detection from video by reading internal dynamics of an online background model (component consolidation), rather than thresholding a single foreground mask. Python package `keyseer`, MIT, `requires-python >= 3.9`.

**Status**: README labels this a research prototype ("NO listo para uso") — the composite scoring metric of the GMM backend fails to separate real events from decoys (negative separation across all tested weight variants). Since then (see `docs/ESTADO.md` §10, §12, §13), the project pivoted: the practical/default path is now the `blobtrack` backend, which has zero decoy-capture in the synthetic eval suite and was run end-to-end on a local real-video dataset (`dataset/`, 14 folders: 23/31 expected event windows hit with 8 extra keyframes, roughly on par with uniform sampling — not conclusive, wide windows, small sources). Treat the GMM/`core.py` path as the research artifact and `blobtrack` as the maintained "personal tool" path.

## Commands

Install (editable, all optional deps):
```
pip install -e ".[all]"
```
Extras: `video` (opencv-python, needed for any real video I/O or the blobtrack backend), `peaks` (scipy, needed for `select_peaks`). No extras are required for the GMM core (`keyseer.core`) or metrics tests — only `numpy` is mandatory. `matplotlib` is NOT a declared extra: only the dataset scripts (`visualizar_dataset`, `graficar_dataset`) need it (`pip install matplotlib`); it is imported lazily, so the logic tests do not need it and the drawing tests are skipped (`pytest.importorskip`) when it is missing.

Run tests:
```
pytest tests/
```
Run a single test: `pytest tests/test_core.py::test_consolidation_separates_transient_from_persistent`

`pyproject.toml` sets `[tool.pytest.ini_options] pythonpath = ["."]` so `evaluacion/` (not an installed package) is importable from tests without needing `PYTHONPATH=.` manually.

Run the preprocessing benchmark (compares baseline/grayscale/motion-gated blobtrack configs on the synthetic decoy suite):
```
python -m evaluacion.benchmark --seeds 8
```
Score-weighting and resize experiments (see `docs/ESTADO.md` §12); other flag: `--out results.json`:
```
python -m evaluacion.benchmark --experiment weighting --seeds 8
python -m evaluacion.benchmark --experiment resolution --seeds 8
```
Real-dataset evaluation (needs the local `dataset/`, see below; one PNG per folder plus `resumen.png`; flags `--only NAME ...`, `--budget N`, `--out DIR`). Without `--out` they overwrite the versioned images, so use `--out` for scratch runs:
```
python -m evaluacion.visualizar_dataset   # tables  -> evaluacion/resultados_dataset/
python -m evaluacion.graficar_dataset     # x-y charts -> evaluacion/resultados_grafica/
```

Run `python -m evaluacion...` from the repo root. No linter or CI config exists in the repo. No console-script entry points — everything is used as a library, plus those `python -m evaluacion.*` scripts.

## Architecture

The codebase is split into two deliberately decoupled halves, referred to in code comments/docs as **Módulo A** and **Módulo B**:

- **Módulo A — persistence backend**: produces a per-frame novelty score + spatial descriptor. Two interchangeable implementations:
  - `keyseer/core.py` (`KeySeer`, `KeySeerConfig`, `FrameReadout`): the GMM/MOG2-style model with an explicit Dirichlet-prior component-consolidation mechanism. Emits multiple signals (`surprisal`, `kl`, `births`, `consolidation`/Ψ) at ~17-47 fps in pure NumPy. Deliberately deviates from OpenCV's MOG2 variance update (isotropic MLE `‖δ‖²/C`, normalized by channel count) — see README for why.
  - `keyseer/blobtrack.py` (`BlobPersistenceTracker`, `analyze_video_blobtrack`): thin wrapper around `cv2.createBackgroundSubtractorMOG2` + centroid-based blob age tracking. The tracker class is ~120 lines (the module is ~330 with `MotionGatedStride` and the video iterator); ~200-450 fps on the synthetic suite (noisy, see `docs/ESTADO.md` §10.1/§12); near-zero memory. Score = sum over tracks older than `min_age_to_count` of (relative area × age, capped by `age_cap`). **This is the default backend** (`keyframes.py`'s `extract_keyframes(backend="blobtrack")`). Uses a forced explicit `learningRate=1/history` in `back_sub.apply()` — the automatic OpenCV default (~1/frames_seen) made objects appearing early in a video get absorbed into the background far faster than identical objects appearing late; this was found and fixed empirically (see `docs/ESTADO.md` §10.2). For real fixed-camera deployments, `grayscale=True` and `motion_gate=True` (a `MotionGatedStride` adaptive-stride gate) are on by default, cutting decode+processing cost during static stretches — see `docs/ESTADO.md` §11 for the calibration story, including a real gotcha: a naive motion gate breaks detection of persistent-but-static objects (the project's core signal) unless it's fed the tracker's active-track state.
  - `keyseer/pipeline.py` (`analyze_video`, `analyze_frames`, `extract_keyframes`) is the GMM-backend driver: turns a frame stream into `(SignalAccumulator, KeySeer model)`, then combines signals into a score. Used by the `backend="gmm"` path in `keyseer/keyframes.py` and directly by `evaluacion/baselines.py`'s `dual_background` baseline (which instantiates `KeySeer` twice at different `alpha` to emulate a dual fast/slow background model).

- **Módulo B — keyframe selection**, independent of which backend produced the score:
  - `keyseer/metrics.py`: `combine(arrays, w_consolidation, w_surprisal, w_kl, gate_power)` is the composite score — this is the metric documented as not yet separating real events from decoys. `spatial_coherence` is a known-weak signal (scores a uniform/flash map ~1.0, same as real structured content) — don't rely on it as a discriminator. `spatial_pool` does trimmed block-averaging for a noise-resistant scalar.
  - `keyseer/selection.py`: `select_submodular` (greedy facility-location maximization over cosine-similarity of spatial descriptors — gives the `(1-1/e)` approximation guarantee mentioned in the README; this is the recommended selector) vs `select_peaks` (simpler `scipy.signal.find_peaks` wrapper, requires the `peaks` extra). `min_gain_ratio` on `select_submodular` is an adaptive early-stopping option that is explicitly **not recommended** — it can drop genuine lower-intensity real events.
  - `keyseer/keyframes.py`: the single public entry point `extract_keyframes(video_path, budget=8, backend="blobtrack", method="submodular", backend_kwargs=None, ...)`. Its internal `_run_backend` is the seam between Módulo A and B — it normalizes either backend's output into `(scores, descriptors, n_frames)` before handing off to selection. Note: `backend="gmm"` here calls `combine(..., gate_power=2.0)`, stricter than `metrics.py`'s own default of `1.0` — check which one applies when tuning. `KeyframeResult` also exposes `dense_positions` (dense/processed-frame indices, index-aligned with `keyframe_indices`) so a consumer can locate a selected keyframe inside `scores`. `resize_to=None` (default) resolves per backend (`keyseer.blobtrack.DEFAULT_RESIZE_TO`, `keyseer.keyframes.GMM_RESIZE_TO`, both `(180, 320)` today) and does NOT mean "no resize" — pass the video size for native resolution. `DEFAULT_RESIZE_TO` was briefly `(360, 640)` and was reverted after the real-dataset comparison (`docs/ESTADO.md` §12 "Reversion", §13.4). `budget` is a maximum but there is no content-based stopping criterion: on real clips the count is set by the budget.

- **Public API** (`keyseer/__init__.py`): only `extract_keyframes`, `KeyframeResult`, `KeySeer`, `KeySeerConfig`, `FrameReadout` are exported at package level. `metrics`, `selection`, `pipeline`, `blobtrack` must be imported from their submodules directly.

- **`evaluacion/`** — a separate evaluation suite (not shipped as part of the `keyseer` package; imports it as a dependency), used to justify the blobtrack-vs-GMM and consolidation-vs-baseline decisions with evidence rather than intuition:
  - `evaluacion/harness.py`: synthetic decoy-video generators — `generate_trap_video` (real events moving/persistent vs decoys: global flashes, dense noise, transient look-alikes) and `generate_scale_contrast_video` (small persistent event next to large continuous motion + global flash; `gate_passing_decoys=True` adds two decoys that do pass blobtrack's gates) — plus scoring (`score_events`, `pairwise_redundancy`).
  - `evaluacion/baselines.py`: baseline keyframe methods for comparison (`uniform_sampling`, `frame_difference`, `mog2_ratio_peaks`, `vsumm_kmeans`, `dual_background`, `jacobs_pless_approx`) — all reuse `keyseer.selection` for peak-picking, and `dual_background` reuses `keyseer.core.KeySeer` directly.
  - `evaluacion/benchmark.py`: blobtrack config benchmark (`CONFIGS`: baseline / grayscale_only / aggressive) and the `--experiment weighting|resolution` matrices (`SCENARIOS`, `apply_weighting` identity|binary, `run_weighting_experiment`, `run_resolution_experiment`, `RESOLUTION_SIZES`).
  - `evaluacion/visualizar_dataset.py` / `graficar_dataset.py` + `dataset_esperado.json`: run the default `extract_keyframes` path on each folder of `dataset/` (assembled into a temporary lossless video first) and save table / x-y chart PNGs; the JSON holds expected event windows and key-count ranges for the 14 folders (windows were fixed by visual inspection BEFORE running the algorithm). matplotlib imported lazily.
  - `evaluacion/PROTOCOLO.md` / `RESULTADOS.md`: the evaluation protocol and results. Key finding of the original GMM study: GMM+submodular gets recall 1.00 / decoy-capture 1.00 at half the dual-background model's memory; the dual model structurally fails on pure-motion events because it's designed for objects that stop moving. The dense-noise-rejection problem (noise decoys getting selected as keyframes) is explicitly **unsolved** for the GMM backend. §8-§9 of RESULTADOS cover blobtrack (synthetic) and the real dataset.
  - `benchmark.py`, `visualizar_dataset.py` and `graficar_dataset.py` are runnable with `python -m`; `harness.py` and `baselines.py` are plain importable modules with no `__main__` block.
  - **`dataset/`** (repo root) is LOCAL evaluation data: 14 folders of consecutive PNG frames (~622 MB), listed in `.gitignore`, never versioned or shipped. Expected layout: `dataset/<Folder>/<Folder>_NNNNNN.png` (files are ordered by their numeric suffix; `Candela_m1_10` files are named `Candela_m1.10_NNNNNN.png`). `dataset/analisis-frames.md` is the user's own description of events/expected key counts — do not edit it; record discrepancies with the frames in `docs/ESTADO.md` §13.2. The result images in `evaluacion/resultados_dataset/` and `evaluacion/resultados_grafica/` ARE versioned (regenerable).

- **`docs/ESTADO.md`** is the authoritative running log of what's been validated, refuted, and decided (including the blobtrack-as-default pivot, the real-dataset evaluation and discarded experiments in §13, and the literature-review finding that component-persistence itself is prior art, not a novel contribution — Stauffer-Grimson 1999, Canon patent US8305440). Read it before making claims about what's proven vs. still open. Never delete history there: mark superseded statements as superseded.
