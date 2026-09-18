"""
evaluacion.benchmark
======================

python -m evaluacion.benchmark [--seeds 8] [--out results.json]

Compara tres configuraciones de blobtrack sobre la suite sintetica de
senuelos (evaluacion/harness.py): recall de eventos reales, captura de
senuelos, fps y memoria de estado pico.

    baseline       : color, decode de todos los frames, stride constante
    grayscale_only : gris, decode de todos los frames, stride constante
    aggressive     : gris + MotionGatedStride (decode-skip + stride adaptativo)

"aggressive" debe igualar recall_real/decoy_hits de "baseline" mostrando
mejor fps y sin peor state_bytes -- si no, los defaults de MotionGatedStride
no estan calibrados (ver docs/ESTADO.md seccion 11).
"""

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from evaluacion.harness import generate_trap_video, score_events
from keyseer.blobtrack import analyze_video_blobtrack
from keyseer.selection import select_submodular

__all__ = ["CONFIGS", "run_one", "run_suite", "summarize", "main"]

CONFIGS = {
    "baseline":       dict(grayscale=False, motion_gate=False),
    "grayscale_only": dict(grayscale=True, motion_gate=False),
    "aggressive":     dict(grayscale=True, motion_gate=True),
}


def run_one(video_path, events, config_kwargs, budget=8, resize_to=(180, 320)):
    r = analyze_video_blobtrack(video_path, resize_to=resize_to, **config_kwargs)
    kf = select_submodular(r["scores"], r["descriptors"], budget=budget,
                           min_distance=10)
    real_kf = [int(r["frame_indices"][k]) for k in kf]
    scored = score_events(real_kf, events, stride=1)   # real_kf ya son indices reales
    fps = r["n_frames"] / r["elapsed"] if r["elapsed"] > 0 else float("inf")
    return {**scored, "fps": fps, "elapsed": r["elapsed"],
            "n_frames_processed": r["n_frames"], "state_bytes": r["peak_state_bytes"]}


def run_suite(seeds=range(8), configs=None, budget=8, resize_to=(180, 320)):
    configs = configs if configs is not None else CONFIGS
    results = {name: [] for name in configs}
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            video_path = str(Path(tmp) / f"trap_{seed}.mp4")
            events = generate_trap_video(video_path, seed=seed)
            for name, kwargs in configs.items():
                results[name].append(run_one(video_path, events, kwargs,
                                             budget=budget, resize_to=resize_to))
    return results


def summarize(results):
    summary = {}
    for name, runs in results.items():
        summary[name] = {
            "recall_real": statistics.fmean(r["recall_real"] for r in runs),
            "decoy_hits": statistics.fmean(r["decoy_hits"] for r in runs),
            "fps": statistics.fmean(r["fps"] for r in runs),
            "elapsed": statistics.fmean(r["elapsed"] for r in runs),
            "state_bytes": statistics.fmean(r["state_bytes"] for r in runs),
            "n_seeds": len(runs),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    results = run_suite(seeds=range(args.seeds))
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
