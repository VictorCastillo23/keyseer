"""
evaluacion.benchmark
======================

python -m evaluacion.benchmark [--seeds 8] [--out results.json]
python -m evaluacion.benchmark --experiment weighting [--seeds 8] [--out results.json]

Compara tres configuraciones de blobtrack sobre la suite sintetica de
senuelos (evaluacion/harness.py): recall de eventos reales, captura de
senuelos, fps y memoria de estado pico.

    baseline       : color, decode de todos los frames, stride constante
    grayscale_only : gris, decode de todos los frames, stride constante
    aggressive     : gris + MotionGatedStride (decode-skip + stride adaptativo)

"aggressive" debe igualar recall_real/decoy_hits de "baseline" mostrando
mejor fps y sin peor state_bytes -- si no, los defaults de MotionGatedStride
no estan calibrados (ver docs/ESTADO.md seccion 11).

--experiment weighting: matriz escenario {trap, scale_contrast,
scale_contrast_decoys} x ponderacion del score en select_submodular
{identity, binary} x presupuesto x config. Mide si descartar la magnitud del
score (binary) rescata eventos chicos frente a uno grande sin subir la
captura de señuelos. El tracker corre UNA vez por (video, config);
ponderacion y presupuesto solo repiten la seleccion.

--experiment resolution: mismo tipo de matriz pero variando el tamano de
resize (RESOLUTION_SIZES) sobre trap (320x240 nativo) y scale_contrast
generado en 1280x720 con los señuelos que pasan los gates. El tracker corre
UNA vez por (video, tamano, config). Ver docs/ESTADO.md seccion 12.
"""

import argparse
import json
import statistics
import tempfile
from functools import partial
from pathlib import Path

from evaluacion.harness import (generate_scale_contrast_video,
                                generate_trap_video, score_events)
from keyseer.blobtrack import DEFAULT_RESIZE_TO, analyze_video_blobtrack
from keyseer.selection import select_submodular

__all__ = ["CONFIGS", "SCENARIOS", "WEIGHTINGS", "RESOLUTION_SIZES",
           "RESOLUTION_SCENARIOS", "apply_weighting", "analyze",
           "select_keyframes", "run_one", "run_suite", "run_weighting_experiment",
           "run_resolution_experiment", "summarize", "format_experiment", "main"]

CONFIGS = {
    "baseline":       dict(grayscale=False, motion_gate=False),
    "grayscale_only": dict(grayscale=True, motion_gate=False),
    "aggressive":     dict(grayscale=True, motion_gate=True),
}


SCENARIOS = {
    "trap":                  generate_trap_video,
    "scale_contrast":        generate_scale_contrast_video,
    "scale_contrast_decoys": partial(generate_scale_contrast_video,
                                     gate_passing_decoys=True),
}

WEIGHTINGS = ("identity", "binary")

# (540, 960) es solo informativo: nunca elegible como default (ver ESTADO.md).
RESOLUTION_SIZES = ((180, 320), (270, 480), (360, 640), (540, 960))

# El trap suite queda en su 320x240 nativo (control de regresion: subir el
# tamano ahi solo interpola). scale_contrast se genera en 1280x720 para que
# reducir el frame pierda informacion de verdad.
RESOLUTION_SCENARIOS = {
    "trap":              generate_trap_video,
    "scale_contrast_hd": partial(generate_scale_contrast_video, W=1280, H=720,
                                 gate_passing_decoys=True),
}


def apply_weighting(scores, weighting=None):
    """Transformacion del score ANTES de select_submodular (que pondera la
    cobertura por u(t)=score). None/"identity": sin cambios. "binary": solo
    importa si el frame tiene evento (score > 0), no cuanto pesa."""
    if weighting in (None, "identity"):
        return scores
    if weighting == "binary":
        return (scores > 0).astype(float)
    raise ValueError(f"ponderacion desconocida: {weighting!r}")


def analyze(video_path, config_kwargs, resize_to=DEFAULT_RESIZE_TO):
    return analyze_video_blobtrack(video_path, resize_to=resize_to, **config_kwargs)


def select_keyframes(r, budget=8, weighting=None):
    u = apply_weighting(r["scores"], weighting)
    kf = select_submodular(u, r["descriptors"], budget=budget, min_distance=10)
    return [int(r["frame_indices"][k]) for k in kf]


def _score_analysis(r, events, budget, weighting):
    real_kf = select_keyframes(r, budget=budget, weighting=weighting)
    scored = score_events(real_kf, events, stride=1)   # real_kf ya son indices reales
    fps = r["n_frames"] / r["elapsed"] if r["elapsed"] > 0 else float("inf")
    return {**scored, "fps": fps, "elapsed": r["elapsed"],
            "n_frames_processed": r["n_frames"], "state_bytes": r["peak_state_bytes"]}


def run_one(video_path, events, config_kwargs, budget=8, resize_to=DEFAULT_RESIZE_TO,
            weighting=None):
    r = analyze(video_path, config_kwargs, resize_to)
    return _score_analysis(r, events, budget, weighting)


def _scenario_generator(scenario):
    try:
        return SCENARIOS[scenario]
    except KeyError:
        raise ValueError(f"escenario desconocido: {scenario!r}") from None


def run_suite(seeds=range(8), configs=None, budget=8, resize_to=DEFAULT_RESIZE_TO,
              scenario="trap", weighting=None):
    generate = _scenario_generator(scenario)
    configs = configs if configs is not None else CONFIGS
    results = {name: [] for name in configs}
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            video_path = str(Path(tmp) / f"{scenario}_{seed}.mp4")
            events = generate(video_path, seed=seed)
            for name, kwargs in configs.items():
                results[name].append(run_one(video_path, events, kwargs,
                                             budget=budget, resize_to=resize_to,
                                             weighting=weighting))
    return results


def run_weighting_experiment(seeds=range(8), scenarios=tuple(SCENARIOS),
                             weightings=WEIGHTINGS, budgets=(3, 5, 8),
                             configs=None, resize_to=DEFAULT_RESIZE_TO):
    configs = configs if configs is not None else {
        k: CONFIGS[k] for k in ("baseline", "aggressive")}
    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in scenarios:
            generate = _scenario_generator(scenario)
            for seed in seeds:
                video_path = str(Path(tmp) / f"{scenario}_{seed}.mp4")
                events = generate(video_path, seed=seed)
                for name, kwargs in configs.items():
                    r = analyze(video_path, kwargs, resize_to)
                    for weighting in weightings:
                        for budget in budgets:
                            runs.setdefault((scenario, weighting, budget, name), []
                                            ).append(_score_analysis(
                                                r, events, budget, weighting))
    return [{"scenario": sc, "weighting": w, "budget": b, "config": c,
             **_mean_row(xs)}
            for (sc, w, b, c), xs in runs.items()]


def _mean_row(xs):
    return {"recall_real": statistics.fmean(x["recall_real"] for x in xs),
            "decoy_hits": statistics.fmean(x["decoy_hits"] for x in xs),
            "fps": statistics.fmean(x["fps"] for x in xs),
            "elapsed": statistics.fmean(x["elapsed"] for x in xs),
            "n_seeds": len(xs)}


def run_resolution_experiment(seeds=range(8), sizes=RESOLUTION_SIZES,
                              scenarios=None, weightings=WEIGHTINGS,
                              budgets=(3, 5, 8), configs=None):
    """Matriz escenario x tamano de resize x config x ponderacion x presupuesto.
    El tracker corre UNA vez por (video, tamano, config); ponderacion y
    presupuesto solo repiten la seleccion. `fps` = frames procesados / tiempo
    de analisis (incluye decode+resize del video fuente, asi que es ruidoso y
    depende de la maquina; `elapsed` es el tiempo total en segundos)."""
    scenarios = scenarios if scenarios is not None else RESOLUTION_SCENARIOS
    configs = configs if configs is not None else {
        k: CONFIGS[k] for k in ("baseline", "aggressive")}
    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for scenario, generate in scenarios.items():
            for seed in seeds:
                video_path = str(Path(tmp) / f"{scenario}_{seed}.mp4")
                events = generate(video_path, seed=seed)
                for size in sizes:
                    for name, kwargs in configs.items():
                        r = analyze(video_path, kwargs, size)
                        for weighting in weightings:
                            for budget in budgets:
                                runs.setdefault(
                                    (scenario, tuple(size), name, weighting, budget), []
                                ).append(_score_analysis(r, events, budget, weighting))
    return [{"scenario": sc, "size": size, "config": c, "weighting": w,
             "budget": b, **_mean_row(xs)}
            for (sc, size, c, w, b), xs in runs.items()]


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


def format_experiment(rows):
    with_size = bool(rows) and "size" in rows[0]
    sw = max([len(r["scenario"]) for r in rows] + [8]) + 1
    lines = [f"{'scenario':<{sw}}" + (f"{'size':<9}" if with_size else "")
             + f"{'weighting':<10}{'budget':>6}  {'config':<11}"
             f"{'recall_real':>11}{'decoy_hits':>11}{'fps':>7}"]
    for r in rows:
        size = f"{r['size'][0]}x{r['size'][1]}" if with_size else ""
        lines.append(f"{r['scenario']:<{sw}}" + (f"{size:<9}" if with_size else "")
                     + f"{r['weighting']:<10}{r['budget']:>6}  "
                     f"{r['config']:<11}{r['recall_real']:>11.3f}"
                     f"{r['decoy_hits']:>11.3f}{r['fps']:>7.0f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--experiment", choices=["weighting", "resolution"],
                        default=None)
    args = parser.parse_args()

    if args.experiment in ("weighting", "resolution"):
        run = (run_weighting_experiment if args.experiment == "weighting"
               else run_resolution_experiment)
        rows = run(seeds=range(args.seeds))
        print(format_experiment(rows))
        if args.out:
            Path(args.out).write_text(json.dumps({"rows": rows}, indent=2))
        return

    results = run_suite(seeds=range(args.seeds))
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
