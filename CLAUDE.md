# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

KeySeer: keyframe detection from video by reading internal dynamics of an online background model (component consolidation), rather than thresholding a single foreground mask. Python package `keyseer`, MIT, `requires-python >= 3.9`.

**Status**: README labels this a research prototype ("NO listo para uso") — the composite scoring metric fails to separate real events from decoys (negative separation across all tested weight variants), validated only on synthetic data. Since then (see `docs/ESTADO.md` §10), the project pivoted: the practical/default path is now the `blobtrack` backend, evaluated end-to-end against a real video and shown to have zero decoy-capture in the eval suite. Treat the GMM/`core.py` path as the research artifact and `blobtrack` as the maintained "personal tool" path.

## Commands

Install (editable, all optional deps):
```
pip install -e ".[all]"
```
Extras: `video` (opencv-python, needed for any real video I/O or the blobtrack backend), `peaks` (scipy, needed for `select_peaks`). No extras are required for the GMM core (`keyseer.core`) or metrics tests — only `numpy` is mandatory.

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

No linter or CI config exists in the repo. No console-script entry points — everything is used as a library.

## Architecture

The codebase is split into two deliberately decoupled halves, referred to in code comments/docs as **Módulo A** and **Módulo B**:

- **Módulo A — persistence backend**: produces a per-frame novelty score + spatial descriptor. Two interchangeable implementations:
  - `keyseer/core.py` (`KeySeer`, `KeySeerConfig`, `FrameReadout`): the GMM/MOG2-style model with an explicit Dirichlet-prior component-consolidation mechanism. Emits multiple signals (`surprisal`, `kl`, `births`, `consolidation`/Ψ) at ~17-47 fps in pure NumPy. Deliberately deviates from OpenCV's MOG2 variance update (isotropic MLE `‖δ‖²/C`, normalized by channel count) — see README for why.
  - `keyseer/blobtrack.py` (`BlobPersistenceTracker`, `analyze_video_blobtrack`): thin wrapper around `cv2.createBackgroundSubtractorMOG2` + centroid-based blob age tracking. ~150 LOC, 345-455 fps, near-zero memory. **This is the default backend** (`keyframes.py`'s `extract_keyframes(backend="blobtrack")`). Uses a forced explicit `learningRate=1/history` in `back_sub.apply()` — the automatic OpenCV default (~1/frames_seen) made objects appearing early in a video get absorbed into the background far faster than identical objects appearing late; this was found and fixed empirically (see `docs/ESTADO.md` §10.1). For real fixed-camera deployments, `grayscale=True` and `motion_gate=True` (a `MotionGatedStride` adaptive-stride gate) are on by default, cutting decode+processing cost during static stretches — see `docs/ESTADO.md` §11 for the calibration story, including a real gotcha: a naive motion gate breaks detection of persistent-but-static objects (the project's core signal) unless it's fed the tracker's active-track state.
  - `keyseer/pipeline.py` (`analyze_video`, `analyze_frames`, `extract_keyframes`) is the GMM-backend driver: turns a frame stream into `(SignalAccumulator, KeySeer model)`, then combines signals into a score. Used by the `backend="gmm"` path in `keyseer/keyframes.py` and directly by `evaluacion/baselines.py`'s `dual_background` baseline (which instantiates `KeySeer` twice at different `alpha` to emulate a dual fast/slow background model).

- **Módulo B — keyframe selection**, independent of which backend produced the score:
  - `keyseer/metrics.py`: `combine(arrays, w_consolidation, w_surprisal, w_kl, gate_power)` is the composite score — this is the metric documented as not yet separating real events from decoys. `spatial_coherence` is a known-weak signal (scores a uniform/flash map ~1.0, same as real structured content) — don't rely on it as a discriminator. `spatial_pool` does trimmed block-averaging for a noise-resistant scalar.
  - `keyseer/selection.py`: `select_submodular` (greedy facility-location maximization over cosine-similarity of spatial descriptors — gives the `(1-1/e)` approximation guarantee mentioned in the README; this is the recommended selector) vs `select_peaks` (simpler `scipy.signal.find_peaks` wrapper, requires the `peaks` extra). `min_gain_ratio` on `select_submodular` is an adaptive early-stopping option that is explicitly **not recommended** — it can drop genuine lower-intensity real events.
  - `keyseer/keyframes.py`: the single public entry point `extract_keyframes(video_path, budget=8, backend="blobtrack", method="submodular", backend_kwargs=None, ...)`. Its internal `_run_backend` is the seam between Módulo A and B — it normalizes either backend's output into `(scores, descriptors, n_frames)` before handing off to selection. Note: `backend="gmm"` here calls `combine(..., gate_power=2.0)`, stricter than `metrics.py`'s own default of `1.0` — check which one applies when tuning. `KeyframeResult` also exposes `dense_positions` (dense/processed-frame indices, index-aligned with `keyframe_indices`) so a consumer can locate a selected keyframe inside `scores`.

- **Public API** (`keyseer/__init__.py`): only `extract_keyframes`, `KeyframeResult`, `KeySeer`, `KeySeerConfig`, `FrameReadout` are exported at package level. `metrics`, `selection`, `pipeline`, `blobtrack` must be imported from their submodules directly.

- **`evaluacion/`** — a separate evaluation suite (not shipped as part of the `keyseer` package; imports it as a dependency), used to justify the blobtrack-vs-GMM and consolidation-vs-baseline decisions with evidence rather than intuition:
  - `evaluacion/harness.py`: synthetic decoy-video generator (`generate_trap_video`) mixing real events (moving/persistent objects) with decoys (global flashes, dense noise, transient look-alikes), plus scoring (`score_events`, `pairwise_redundancy`).
  - `evaluacion/baselines.py`: baseline keyframe methods for comparison (`uniform_sampling`, `frame_difference`, `mog2_ratio_peaks`, `vsumm_kmeans`, `dual_background`, `jacobs_pless_approx`) — all reuse `keyseer.selection` for peak-picking, and `dual_background` reuses `keyseer.core.KeySeer` directly.
  - `evaluacion/PROTOCOLO.md` / `RESULTADOS.md`: the evaluation protocol and results. Key finding: GMM+submodular gets recall 1.00 / decoy-capture 1.00 at half the dual-background model's memory; the dual model structurally fails on pure-motion events because it's designed for objects that stop moving. The dense-noise-rejection problem (noise decoys getting selected as keyframes) is explicitly **unsolved** as of the latest results.
  - Neither file is a CLI — both are plain importable modules with no `__main__` block.

- **`docs/ESTADO.md`** is the authoritative running log of what's been validated, refuted, and decided (including the blobtrack-as-default pivot and the literature-review finding that component-persistence itself is prior art, not a novel contribution — Stauffer-Grimson 1999, Canon patent US8305440). Read it before making claims about what's proven vs. still open.

## Known documentation drift

The README's low-level usage example passes `config=` to `extract_keyframes`; the actual signature (`keyseer/keyframes.py`) takes `backend_kwargs=` instead.
