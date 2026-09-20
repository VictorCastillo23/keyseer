"""
evaluacion.graficar_dataset
=============================

python -m evaluacion.graficar_dataset [--dataset dataset]
                                      [--esperado evaluacion/dataset_esperado.json]
                                      [--out evaluacion/resultados_grafica]
                                      [--only NOMBRE ...] [--budget N]

Version "grafica x-y" de evaluacion/visualizar_dataset.py: en vez de una
tabla con una celda por frame, dibuja por carpeta el score de persistencia
(eje y) contra el numero de frame (eje x). Franja verde = rango donde se
esperaba un keyframe (numerada 1, 2, ...), linea roja = frame elegido por el
algoritmo, con un marcador sobre la curva. Al final guarda `resumen.png`:
tambien una grafica x-y (carpeta vs cantidad de keyframes: banda verde =
rango esperado, punto rojo = encontrados).

Los frames encontrados salen de extract_keyframes (camino por defecto de la
libreria) y la curva de analyze_video_blobtrack con sus parametros por
defecto: dos corridas del mismo tracker determinista, sin reimplementar la
seleccion. La curva tiene un punto por frame PROCESADO (el motion gate se
salta frames), por eso se dibuja contra `frame_indices` y no se inventan
scores para los frames saltados.

Cada carpeta se arma como video temporal sin perdida (ver `armar_video` en
visualizar_dataset). budget por defecto = extremo superior de
`keys_esperados` (minimo 1).

matplotlib no es dependencia declarada del paquete:  pip install matplotlib
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from evaluacion.visualizar_dataset import (
    DEFAULT_DATASET, DEFAULT_ESPERADO, COLOR_ENCONTRADO, COLOR_ESPERADO,
    COLOR_FALLA, COLOR_OK, DPI, _fmt_rango, _lineas_pie,
    _pyplot, armar_video, cargar_esperado, clasificar, fila_resumen)

__all__ = ["marcadores_encontrados", "preparar_datos", "asignar_niveles",
           "fila_resumen_grafica", "analizar_video", "dibujar_grafica",
           "dibujar_resumen_grafica", "procesar_carpeta", "main"]

DEFAULT_OUT = Path(__file__).resolve().parent / "resultados_grafica"

COLOR_CURVA = "#4a6fa5"
COLOR_ERROR = "#7a7a7a"
COLOR_TEXTO_ALGUNOS = "#8a5a00"
COLORES_ACIERTOS = {"todos": COLOR_OK, "algunos": COLOR_TEXTO_ALGUNOS,
                    "ninguno": COLOR_FALLA, "sin_ventanas": "#555555",
                    "error": COLOR_ERROR}
COLOR_BORDE_ESPERADO = "#5fae7b"

ANCHO = 14.0
ALTO_GRAFICA = 3.9
NIVELES_ETIQUETA = 3
SEPARACION_ETIQUETA = 0.045   # fraccion del eje x entre etiquetas del mismo nivel


# ------------------------------------------------------------ datos (puro)

def marcadores_encontrados(encontrados, scores, frame_indices):
    """[{"frame", "score"}] ordenado por frame. El score es el de la curva en
    ese numero de frame REAL (via `frame_indices`); un frame sin punto en la
    curva (o una curva vacia) queda en 0.0."""
    por_frame = {int(f): float(s) for f, s in zip(frame_indices, scores)}
    return [{"frame": int(k), "score": por_frame.get(int(k), 0.0)}
            for k in sorted(int(k) for k in encontrados)]


def preparar_datos(n_frames, encontrados, ventanas, scores, frame_indices):
    """Junta lo que necesita el dibujo: curva (x, y), ventanas numeradas desde
    1 con su acierto, marcadores de los frames encontrados y la clasificacion
    de `clasificar`. `n_frames` no se usa para filtrar: el dibujo recorta."""
    scores = np.asarray(scores, dtype=np.float64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if scores.shape != frame_indices.shape:
        raise ValueError("scores y frame_indices deben tener la misma longitud "
                         f"({scores.shape} vs {frame_indices.shape})")
    r = clasificar(encontrados, ventanas)
    numeradas = [{"numero": i, "nombre": w["nombre"], "inicio": w["inicio"],
                  "fin": w["fin"], "acierto": w["acierto"],
                  "encontrados": w["encontrados"]}
                 for i, w in enumerate(r["ventanas"], start=1)]
    return {"x": frame_indices, "y": scores, "ventanas": numeradas,
            "encontrados": marcadores_encontrados(encontrados, scores,
                                                  frame_indices),
            "clasificacion": r}


def asignar_niveles(frames, x_max, niveles=NIVELES_ETIQUETA,
                    separacion=SEPARACION_ETIQUETA):
    """Nivel (fila vertical) de la etiqueta de cada frame, en el orden dado.

    Toma el primer nivel cuyo ultimo rotulo quede a >= `separacion * x_max`;
    si todos estan ocupados usa el de rotulo mas lejano (cicla, y nunca repite
    el nivel del vecino inmediato)."""
    minimo = max(separacion * x_max, 1e-9)
    ultimo = [None] * niveles
    salida = []
    previo = None
    for f in frames:
        elegido = None
        for n in range(niveles):
            if ultimo[n] is None or f - ultimo[n] >= minimo:
                elegido = n
                break
        if elegido is None:
            candidatos = [n for n in range(niveles) if n != previo]
            elegido = max(candidatos, key=lambda n: f - ultimo[n])
        ultimo[elegido] = f
        salida.append(elegido)
        previo = elegido
    return salida


def fila_resumen_grafica(carpeta, n_frames, budget, keys_esperados, encontrados,
                         ventanas):
    """`fila_resumen` mas lo numerico que necesita la grafica resumen:
    keys_min, keys_max y n_encontrados (None si la carpeta fallo)."""
    fila = fila_resumen(carpeta, n_frames, budget, keys_esperados, encontrados,
                        ventanas)
    fila.update(keys_min=int(keys_esperados[0]), keys_max=int(keys_esperados[1]),
                n_encontrados=None if encontrados is None else len(encontrados))
    return fila


def analizar_video(video, budget):
    """(encontrados, scores, frame_indices) de un video: los frames salen de
    extract_keyframes (default de la libreria) y la curva de
    analyze_video_blobtrack con sus defaults, que es la misma configuracion.
    Falla si la curva no coincide con la que uso el selector."""
    from keyseer.blobtrack import analyze_video_blobtrack
    from keyseer.keyframes import extract_keyframes

    res = extract_keyframes(str(video), budget=budget)
    curva = analyze_video_blobtrack(str(video))
    scores = np.asarray(curva["scores"], dtype=np.float64)
    if scores.shape != np.asarray(res.scores).shape or \
            not np.allclose(scores, res.scores):
        raise RuntimeError("la curva de analyze_video_blobtrack no coincide "
                           "con la de extract_keyframes")
    encontrados = sorted(int(k) for k in res.keyframe_indices)
    return encontrados, scores, np.asarray(curva["frame_indices"],
                                           dtype=np.int64)


# ----------------------------------------------------------------- dibujo

def _fmt_frame(n):
    return f"{int(n)}"


def dibujar_grafica(carpeta, n_frames, encontrados, ventanas, keys_esperados,
                    budget, scores, frame_indices, destino):
    """Guarda en `destino` la grafica x-y de una carpeta y devuelve el
    resultado de `clasificar`."""
    plt = _pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    import matplotlib.transforms as mtransforms

    datos = preparar_datos(n_frames, encontrados, ventanas, scores,
                           frame_indices)
    r = datos["clasificacion"]
    x, y = datos["x"], datos["y"]
    marcas = [m for m in datos["encontrados"] if 0 <= m["frame"] < n_frames]

    entradas = [(f"{w['numero']}. {w['nombre']} [{w['inicio']}-{w['fin']}]: "
                 f"{'OK' if w['acierto'] else 'FALLA'}", w["acierto"])
                for w in datos["ventanas"]]
    _, lineas = _lineas_pie(encontrados, keys_esperados, ventanas, r)
    ncols = 3
    filas_ventanas = -(-len(entradas) // ncols)

    margen_x = 0.9
    ancho_util = ANCHO - margen_x - 0.35
    alto_cab = 1.15
    alto_ejes_inf = 0.62           # etiquetas del eje x
    alto_pie = 0.25 + 0.24 * filas_ventanas + 0.22 * len(lineas)
    alto = alto_cab + ALTO_GRAFICA + alto_ejes_inf + alto_pie

    def fy(pulgadas_desde_arriba):
        return 1 - pulgadas_desde_arriba / alto

    fig = plt.figure(figsize=(ANCHO, alto), dpi=DPI)
    fig.text(0.5, fy(0.12), f"{carpeta} - {n_frames} frames - budget {budget}",
             ha="center", va="top", fontsize=14, fontweight="bold")
    if ventanas:
        sub = (f"aciertos {r['aciertos']}/{r['total_ventanas']} ventanas - "
               f"extras {len(r['extras'])} - "
               f"esperado: {_fmt_rango(keys_esperados)} keyframes")
    else:
        n = len(encontrados)
        lo, hi = keys_esperados
        veredicto = "dentro" if lo <= n <= hi else "fuera"
        sub = (f"sin ventanas esperadas - esperado: "
               f"{_fmt_rango(keys_esperados)} keyframes | encontrados: {n} "
               f"-> {veredicto} del rango")
    if not len(encontrados):
        sub += " - 0 keyframes encontrados"
    fig.text(0.5, fy(0.46), sub, ha="center", va="top", fontsize=10.5)

    ax = fig.add_axes([margen_x / ANCHO, fy(alto_cab + ALTO_GRAFICA),
                       ancho_util / ANCHO, ALTO_GRAFICA / alto])
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    for w in datos["ventanas"]:
        ax.axvspan(w["inicio"] - 0.5, w["fin"] + 0.5, facecolor=COLOR_ESPERADO,
                   edgecolor=COLOR_BORDE_ESPERADO, linewidth=0.6, alpha=0.55,
                   zorder=1)
        ax.text((w["inicio"] + w["fin"]) / 2, 0.975, str(w["numero"]),
                transform=trans, ha="center", va="top", fontsize=9,
                fontweight="bold", color=COLOR_OK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec=COLOR_BORDE_ESPERADO, lw=0.7))

    hay_curva = len(y) > 0 and float(y.max()) > 0
    if len(x):
        ax.plot(x, y, color=COLOR_CURVA, linewidth=1.0, zorder=3)
    ymax = float(y.max()) if hay_curva else 1.0
    ax.set_ylim(0, ymax * 1.5)
    ax.set_xlim(-0.5, max(n_frames, 1) - 0.5)
    if not hay_curva:
        ax.text(0.5, 0.5, "sin scores > 0 (curva vacia)", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#888888")

    filas_y = [0.895, 0.815, 0.735][:NIVELES_ETIQUETA]
    niveles = asignar_niveles([m["frame"] for m in marcas], max(n_frames, 1))
    x_lo, x_hi = ax.get_xlim()
    pad = 0.02 * (x_hi - x_lo)
    for m, nivel in zip(marcas, niveles):
        k, s = m["frame"], m["score"]
        ax.axvline(k, color=COLOR_ENCONTRADO, linewidth=1.8, alpha=0.9,
                   zorder=4)
        ax.plot([k], [s], "o", color=COLOR_ENCONTRADO, markeredgecolor="white",
                markersize=7, zorder=5)
        ax.text(min(max(k, x_lo + pad), x_hi - pad), filas_y[nivel], _fmt_frame(k),
                transform=trans, ha="center", va="center", fontsize=9,
                fontweight="bold", color=COLOR_ENCONTRADO, zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=COLOR_ENCONTRADO, lw=0.7))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.set_xlabel("frame")
    ax.set_ylabel("score de persistencia (blobtrack)")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    handles = [Line2D([0], [0], color=COLOR_CURVA, linewidth=1.4,
                      label="score de persistencia (blobtrack)"),
               Line2D([0], [0], color=COLOR_ENCONTRADO, linewidth=2,
                      marker="o", markeredgecolor="white",
                      label="frame encontrado por el algoritmo")]
    if ventanas:
        handles.insert(0, Patch(facecolor=COLOR_ESPERADO,
                                edgecolor=COLOR_BORDE_ESPERADO, alpha=0.7,
                                label="rango esperado (ventana numerada)"))
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, fy(0.72)), ncol=len(handles),
               frameon=False, fontsize=9.5)

    y_pie = alto_cab + ALTO_GRAFICA + alto_ejes_inf + 0.08
    ancho_col = ancho_util / ncols
    for i, (linea, ok) in enumerate(entradas):
        fila_v, col_v = divmod(i, ncols)
        fig.text((margen_x + col_v * ancho_col) / ANCHO,
                 fy(y_pie + 0.24 * fila_v), linea, ha="left", va="top",
                 fontsize=9, family="monospace",
                 color=COLOR_OK if ok else COLOR_FALLA, fontweight="bold")
    y_pie += 0.24 * filas_ventanas + (0.04 if entradas else 0)
    for i, linea in enumerate(lineas):
        fig.text(margen_x / ANCHO, fy(y_pie + 0.22 * i), linea, ha="left",
                 va="top", fontsize=9, family="monospace", color="#222222")

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=DPI, facecolor="white")
    plt.close(fig)
    return r


def dibujar_resumen_grafica(filas, destino):
    """Grafica x-y resumen: eje x = carpeta, eje y = cantidad de keyframes.
    Banda verde = rango esperado [min, max]; punto rojo = encontrados, con
    `X/Y` ventanas acertadas encima ('-' sin ventanas); una carpeta que
    fallo lleva un marcador gris ERROR."""
    plt = _pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import MaxNLocator

    n = len(filas)
    ancho = max(8.0, 0.9 * n + 3.0)
    alto = 6.6
    fig = plt.figure(figsize=(ancho, alto), dpi=DPI)
    fig.text(0.5, 1 - 0.12 / alto,
             "KeySeer - keyframes encontrados vs rango esperado por carpeta",
             ha="center", va="top", fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.07, 0.25, 0.91, 0.59])

    tope = 1
    for i, f in enumerate(filas):
        lo, hi = f["keys_min"], f["keys_max"]
        ax.add_patch(Rectangle((i - 0.36, lo - 0.4), 0.72, hi - lo + 0.8,
                               facecolor=COLOR_ESPERADO, alpha=0.6,
                               edgecolor=COLOR_BORDE_ESPERADO, linewidth=0.8,
                               zorder=1))
        tope = max(tope, hi)
        if f["n_encontrados"] is None:
            ax.plot([i], [0], "X", color=COLOR_ERROR, markersize=10, zorder=4)
            ax.text(i, 0.55, "ERROR", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=COLOR_ERROR, zorder=5)
            continue
        ne = f["n_encontrados"]
        tope = max(tope, ne)
        ax.plot([i], [ne], "o", color=COLOR_ENCONTRADO, markeredgecolor="white",
                markersize=10, zorder=4)
        ax.text(i, ne + 0.5, f["aciertos"], ha="center", va="bottom",
                fontsize=13 if f["aciertos"] == "-" else 10, fontweight="bold",
                color=COLORES_ACIERTOS[f["estado"]], zorder=5)

    ax.set_xlim(-0.7, max(n, 1) - 0.3)
    ax.set_ylim(-1, tope + 2)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f["carpeta"] for f in filas], rotation=40, ha="right",
                       fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("cantidad de keyframes")
    ax.set_xlabel("carpeta del dataset")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=COLOR_ESPERADO, edgecolor=COLOR_BORDE_ESPERADO,
                     alpha=0.7, label="rango esperado de keyframes"),
               Line2D([0], [0], marker="o", linestyle="", color=COLOR_ENCONTRADO,
                      markeredgecolor="white", markersize=9,
                      label="keyframes encontrados")]
    if any(f["n_encontrados"] is None for f in filas):
        handles.append(Line2D([0], [0], marker="X", linestyle="",
                              color=COLOR_ERROR, markersize=9,
                              label="carpeta con ERROR"))
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.5 / alto), ncol=len(handles),
               frameon=False, fontsize=9.5)
    fig.text(0.07, 0.02, "X/Y sobre cada punto = ventanas esperadas acertadas "
             "('-' = carpeta sin ventanas)", ha="left", va="bottom",
             fontsize=9.5, color="#333333")

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=DPI, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------- driver

def procesar_carpeta(nombre, dataset, entrada, out, budget=None):
    """Arma el video temporal, corre el algoritmo, dibuja la grafica y
    devuelve (fila_resumen_grafica, encontrados, segundos)."""
    keys = entrada["keys_esperados"]
    ventanas = entrada["ventanas"]
    budget = max(1, keys[1]) if budget is None else budget
    tmp = Path(tempfile.mkdtemp(prefix="keyseer_graf_"))
    try:
        video = tmp / f"{nombre}.avi"
        n_frames = armar_video(Path(dataset) / nombre, video)
        t0 = time.time()
        encontrados, scores, frame_indices = analizar_video(video, budget)
        segundos = time.time() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    dibujar_grafica(nombre, n_frames, encontrados, ventanas, keys, budget,
                    scores, frame_indices, Path(out) / f"{nombre}.png")
    fila = fila_resumen_grafica(nombre, n_frames, budget, keys, encontrados,
                                ventanas)
    return fila, encontrados, segundos


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Graficas x-y por carpeta del dataset: score de "
                    "persistencia, frames encontrados vs rangos esperados.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--esperado", default=str(DEFAULT_ESPERADO))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", nargs="+", metavar="NOMBRE", default=None)
    ap.add_argument("--budget", type=int, default=None,
                    help="pisa el budget por defecto (extremo superior de "
                         "keys_esperados, minimo 1)")
    args = ap.parse_args(argv)

    esperado = cargar_esperado(args.esperado)
    nombres = args.only or list(esperado)
    desconocidas = [n for n in nombres if n not in esperado]
    if desconocidas:
        ap.error(f"carpetas sin entrada en {args.esperado}: {desconocidas}")

    filas, errores = [], 0
    for nombre in nombres:
        entrada = esperado[nombre]
        try:
            fila, encontrados, seg = procesar_carpeta(
                nombre, args.dataset, entrada, args.out, budget=args.budget)
            print(f"{nombre:<17} frames={fila['frames']:<4} "
                  f"budget={fila['budget']:<2} encontrados={encontrados} "
                  f"aciertos={fila['aciertos']} extras={fila['extras']} "
                  f"({seg:.1f}s)", flush=True)
        except Exception as e:  # una carpeta rota no aborta las demas
            errores += 1
            print(f"{nombre:<17} ERROR: {type(e).__name__}: {e}", flush=True)
            budget = (args.budget if args.budget is not None
                      else max(1, entrada["keys_esperados"][1]))
            fila = fila_resumen_grafica(nombre, "-", budget,
                                        entrada["keys_esperados"], None,
                                        entrada["ventanas"])
        filas.append(fila)

    dibujar_resumen_grafica(filas, Path(args.out) / "resumen.png")
    print(f"resumen -> {Path(args.out) / 'resumen.png'}")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
