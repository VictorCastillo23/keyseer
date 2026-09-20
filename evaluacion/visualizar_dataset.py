"""
evaluacion.visualizar_dataset
===============================

python -m evaluacion.visualizar_dataset [--dataset dataset]
                                        [--esperado evaluacion/dataset_esperado.json]
                                        [--out evaluacion/resultados_dataset]
                                        [--only NOMBRE ...] [--budget N]
                                        [--columnas 25]

Corre el camino por defecto de la libreria (extract_keyframes: blobtrack, gris
+ motion gate, resize 180x320, submodular) sobre cada carpeta de PNGs
consecutivos de `dataset/` y guarda una imagen por carpeta con una TABLA
matplotlib (una celda por frame): verde = rango donde se esperaba un keyframe,
rojo = frame elegido por el algoritmo (gana sobre el verde). Al final guarda
`resumen.png` con una fila por carpeta (con --only, solo las procesadas).

extract_keyframes necesita un archivo de video, asi que cada carpeta se arma
como un video temporal SIN perdida (FFV1 en .avi; MJPG si el writer no abre)
que se borra al terminar. Los numeros de frame de salida son los de la carpeta.

budget por defecto = extremo superior de `keys_esperados` (minimo 1).
Las ventanas de dataset_esperado.json se fijaron por inspeccion visual ANTES
de correr el algoritmo.

matplotlib no es dependencia declarada del paquete:  pip install matplotlib
"""

import argparse
import json
import math
import re
import shutil
import tempfile
import textwrap
import time
from pathlib import Path

__all__ = ["cargar_esperado", "clasificar", "categoria_por_frame",
           "armar_video", "fila_resumen", "dibujar_tabla", "dibujar_resumen",
           "procesar_carpeta", "main"]

DEFAULT_DATASET = "dataset"
DEFAULT_ESPERADO = Path(__file__).resolve().parent / "dataset_esperado.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "resultados_dataset"

COLOR_NORMAL = "#f0f0f0"
COLOR_ESPERADO = "#a8dfb8"
COLOR_ENCONTRADO = "#e8481c"
COLOR_BORDE = "#c8c8c8"
COLOR_OK = "#1b6b34"
COLOR_FALLA = "#b3261e"
COLORES_ESTADO = {
    "todos": "#a8dfb8",
    "algunos": "#ffe08a",
    "ninguno": "#f4a3a3",
    "sin_ventanas": "#dcdcdc",
    "error": "#f4a3a3",
}
DPI = 150
ANCHO_CELDA = 0.56   # pulgadas
ALTO_CELDA = 0.32


def cargar_esperado(ruta):
    """{carpeta: {"keys_esperados": [min, max], "ventanas": [...]}}. Ignora
    las claves de metadatos (`_nota`) del JSON."""
    data = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return data["carpetas"]


def _en_ventana(frame, ventana):
    return ventana["inicio"] <= frame <= ventana["fin"]


def clasificar(encontrados, ventanas):
    """Cruza los frames encontrados con las ventanas esperadas (inclusivas).

    Una ventana acierta si tiene >= 1 frame encontrado. `extras` = fuera de
    toda ventana; `redundantes` = dentro de alguna ventana pero sin ser el
    primero encontrado de ninguna (2do keyframe en una ventana ya cubierta).
    Un frame en el borde compartido de dos ventanas cuenta para ambas.
    """
    encontrados = sorted(int(k) for k in encontrados)
    por_ventana = []
    primeros = set()
    for v in ventanas:
        dentro = [k for k in encontrados if _en_ventana(k, v)]
        if dentro:
            primeros.add(dentro[0])
        por_ventana.append({**v, "encontrados": dentro,
                            "acierto": bool(dentro)})
    extras = [k for k in encontrados
              if not any(_en_ventana(k, v) for v in ventanas)]
    redundantes = [k for k in encontrados
                   if k not in extras and k not in primeros]
    return {"ventanas": por_ventana,
            "aciertos": sum(w["acierto"] for w in por_ventana),
            "total_ventanas": len(por_ventana),
            "extras": extras,
            "redundantes": redundantes}


def categoria_por_frame(n_frames, encontrados, ventanas):
    """'normal' | 'esperado' | 'encontrado' por frame; 'encontrado' gana
    sobre 'esperado'. Ventanas y frames fuera de [0, n_frames) se ignoran."""
    cat = ["normal"] * n_frames
    for v in ventanas:
        for i in range(max(v["inicio"], 0), min(v["fin"], n_frames - 1) + 1):
            cat[i] = "esperado"
    for k in encontrados:
        if 0 <= int(k) < n_frames:
            cat[int(k)] = "encontrado"
    return cat


def _indice_frame(ruta):
    m = re.search(r"(\d+)\.png$", ruta.name, re.IGNORECASE)
    if m is None:
        raise ValueError(f"nombre sin sufijo numerico: {ruta.name}")
    return int(m.group(1))


def _contar_frames(ruta_video):
    import cv2
    cap = cv2.VideoCapture(str(ruta_video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    return n


def armar_video(carpeta, destino, fps=25):
    """Junta los PNG de `carpeta` (orden = sufijo numerico del nombre) en un
    video sin perdida (FFV1, o MJPG si el writer no abre). Devuelve la
    cantidad de frames y verifica que el video tenga esa misma cantidad."""
    import cv2
    pngs = sorted(Path(carpeta).glob("*.png"), key=_indice_frame)
    if not pngs:
        raise ValueError(f"sin PNG en {carpeta}")

    primero = cv2.imread(str(pngs[0]))
    if primero is None:
        raise ValueError(f"no se pudo leer {pngs[0]}")
    h, w = primero.shape[:2]

    out = None
    for codec in ("FFV1", "MJPG"):
        out = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*codec),
                              fps, (w, h))
        if out.isOpened():
            break
        out.release()
        out = None
    if out is None:
        raise RuntimeError("cv2.VideoWriter no abrio ni con FFV1 ni con MJPG")

    try:
        for p in pngs:
            img = cv2.imread(str(p))
            if img is None or img.shape[:2] != (h, w):
                raise ValueError(f"frame ilegible o de otro tamano: {p.name}")
            out.write(img)
    finally:
        out.release()

    n_video = _contar_frames(destino)
    if n_video != len(pngs):
        raise RuntimeError(f"el video tiene {n_video} frames pero la carpeta "
                           f"{len(pngs)} PNG")
    return len(pngs)


def _fmt_rango(keys):
    lo, hi = keys
    return str(lo) if lo == hi else f"{lo}-{hi}"


def fila_resumen(carpeta, n_frames, budget, keys_esperados, encontrados,
                 ventanas):
    """Una fila de resumen.png; `encontrados=None` marca una carpeta que
    fallo. `extras` es '-' cuando la carpeta no tiene ventanas (todo lo
    encontrado seria 'extra' y no aporta informacion)."""
    fila = {"carpeta": carpeta, "frames": n_frames, "budget": budget,
            "esperado": _fmt_rango(keys_esperados)}
    if encontrados is None:
        fila.update(encontrados="ERROR", aciertos="-", extras="-",
                    estado="error")
        return fila
    r = clasificar(encontrados, ventanas)
    fila["encontrados"] = len(encontrados)
    if not ventanas:
        fila.update(aciertos="-", extras="-", estado="sin_ventanas")
        return fila
    if r["aciertos"] == r["total_ventanas"]:
        estado = "todos"
    elif r["aciertos"] == 0:
        estado = "ninguno"
    else:
        estado = "algunos"
    fila.update(aciertos=f"{r['aciertos']}/{r['total_ventanas']}",
                extras=len(r["extras"]), estado=estado)
    return fila


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _lineas_pie(encontrados, keys_esperados, ventanas, r):
    """(entradas_ventanas, lineas_texto): entradas = [(texto, ok)]."""
    entradas = [(f"{w['nombre']} [{w['inicio']}-{w['fin']}]: "
                 f"{'OK' if w['acierto'] else 'FALLA'}", w["acierto"])
                for w in r["ventanas"]]
    n = len(encontrados)
    lo, hi = keys_esperados
    veredicto = "dentro del rango" if lo <= n <= hi else "fuera del rango"
    if n:
        lista = ", ".join(str(k) for k in sorted(int(k) for k in encontrados))
        cuerpo = f"encontrados ({n}): {lista}"
    else:
        cuerpo = "0 keyframes encontrados"
    lineas = textwrap.wrap(cuerpo, width=150)
    lineas.append(f"esperado: {_fmt_rango(keys_esperados)} keyframes | "
                  f"encontrados: {n} -> {veredicto}")
    if not ventanas:
        lineas.append("sin ventanas esperadas: solo se espera la cantidad de "
                      "keyframes (flujo continuo / escena sin eventos)")
    elif r["redundantes"]:
        lineas.append("redundantes (2do keyframe en una ventana ya cubierta): "
                      + ", ".join(str(k) for k in r["redundantes"]))
    return entradas, lineas


def dibujar_tabla(carpeta, n_frames, encontrados, ventanas, keys_esperados,
                  budget, destino, columnas=25):
    """Guarda en `destino` la tabla (una celda por frame) de una carpeta y
    devuelve el resultado de `clasificar`."""
    plt = _pyplot()
    from matplotlib.patches import Patch

    r = clasificar(encontrados, ventanas)
    cat = categoria_por_frame(n_frames, encontrados, ventanas)
    filas = max(1, math.ceil(n_frames / columnas))

    texto = [[""] * columnas for _ in range(filas)]
    colores = [["white"] * columnas for _ in range(filas)]
    fondo = {"normal": COLOR_NORMAL, "esperado": COLOR_ESPERADO,
             "encontrado": COLOR_ENCONTRADO}
    for i, c in enumerate(cat):
        texto[i // columnas][i % columnas] = str(i)
        colores[i // columnas][i % columnas] = fondo[c]

    entradas, lineas = _lineas_pie(encontrados, keys_esperados, ventanas, r)
    ancho = max(columnas * ANCHO_CELDA + 0.6, 7.0)
    ancho_util = ancho - 0.6
    ncols = max(1, min(3, int(ancho_util // 3.4)))
    filas_ventanas = math.ceil(len(entradas) / ncols)

    alto_cab = 1.15
    alto_tabla = filas * ALTO_CELDA
    alto_pie = 0.25 + 0.24 * filas_ventanas + 0.22 * len(lineas)
    alto = alto_cab + alto_tabla + alto_pie

    def fy(pulgadas_desde_arriba):
        return 1 - pulgadas_desde_arriba / alto

    fig = plt.figure(figsize=(ancho, alto), dpi=DPI)
    fig.text(0.5, fy(0.12), f"{carpeta} - {n_frames} frames - budget {budget}",
             ha="center", va="top", fontsize=14, fontweight="bold")
    if ventanas:
        sub = (f"aciertos {r['aciertos']}/{r['total_ventanas']} ventanas - "
               f"extras {len(r['extras'])} - "
               f"esperado: {_fmt_rango(keys_esperados)} keyframes")
    else:
        sub = (f"sin ventanas esperadas - esperado: "
               f"{_fmt_rango(keys_esperados)} keyframes - "
               f"encontrados {len(encontrados)}")
    fig.text(0.5, fy(0.46), sub, ha="center", va="top", fontsize=10.5)

    parches = [Patch(facecolor=COLOR_ENCONTRADO, edgecolor=COLOR_BORDE,
                     label="frame encontrado por el algoritmo"),
               Patch(facecolor=COLOR_NORMAL, edgecolor=COLOR_BORDE,
                     label="otro frame")]
    if ventanas:
        parches.insert(0, Patch(facecolor=COLOR_ESPERADO, edgecolor=COLOR_BORDE,
                                label="rango esperado (ventana)"))
    fig.legend(handles=parches, loc="upper center",
               bbox_to_anchor=(0.5, fy(0.72)), ncol=len(parches),
               frameon=False, fontsize=9.5)

    margen = 0.3 / ancho
    ax = fig.add_axes([margen, fy(alto_cab + alto_tabla), 1 - 2 * margen,
                       alto_tabla / alto])
    ax.axis("off")
    tabla = ax.table(cellText=texto, cellColours=colores, cellLoc="center",
                     loc="center", bbox=[0, 0, 1, 1])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    for (fi, ci), celda in tabla.get_celld().items():
        idx = fi * columnas + ci
        if idx >= n_frames:
            celda.visible_edges = "open"
            continue
        celda.set_edgecolor(COLOR_BORDE)
        celda.set_linewidth(0.6)
        if cat[idx] == "encontrado":
            celda.get_text().set_color("white")
            celda.get_text().set_fontweight("bold")
        else:
            celda.get_text().set_color("#333333")

    y = alto_cab + alto_tabla + 0.15
    ancho_col = ancho_util / ncols
    for i, (linea, ok) in enumerate(entradas):
        fila_v, col_v = divmod(i, ncols)
        fig.text((0.3 + col_v * ancho_col) / ancho, fy(y + 0.24 * fila_v),
                 linea, ha="left", va="top", fontsize=9,
                 family="monospace", color=COLOR_OK if ok else COLOR_FALLA,
                 fontweight="bold")
    y += 0.24 * filas_ventanas + (0.04 if entradas else 0)
    for i, linea in enumerate(lineas):
        fig.text(0.3 / ancho, fy(y + 0.22 * i), linea, ha="left", va="top",
                 fontsize=9, family="monospace", color="#222222")

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=DPI, facecolor="white")
    plt.close(fig)
    return r


def dibujar_resumen(filas, destino):
    """Tabla resumen (una fila por carpeta); la celda Aciertos se colorea
    segun `estado`: verde todas, amarillo algunas, rojo ninguna, gris sin
    ventanas."""
    plt = _pyplot()
    from matplotlib.patches import Patch

    cols = ["Carpeta", "Frames", "Budget", "Esperado (keys)",
            "Encontrados (n)", "Aciertos", "Extras"]
    campos = ["carpeta", "frames", "budget", "esperado", "encontrados",
              "aciertos", "extras"]
    texto = [[str(f[c]) for c in campos] for f in filas]
    alto_fila = 0.36
    ancho, alto = 10.5, 1.5 + alto_fila * (len(filas) + 1)

    fig = plt.figure(figsize=(ancho, alto), dpi=DPI)
    fig.text(0.5, 1 - 0.12 / alto, "KeySeer - resultados por carpeta del dataset",
             ha="center", va="top", fontsize=14, fontweight="bold")
    parches = [Patch(facecolor=COLORES_ESTADO["todos"], edgecolor=COLOR_BORDE,
                     label="todas las ventanas"),
               Patch(facecolor=COLORES_ESTADO["algunos"], edgecolor=COLOR_BORDE,
                     label="algunas"),
               Patch(facecolor=COLORES_ESTADO["ninguno"], edgecolor=COLOR_BORDE,
                     label="ninguna / error"),
               Patch(facecolor=COLORES_ESTADO["sin_ventanas"],
                     edgecolor=COLOR_BORDE, label="sin ventanas")]
    fig.legend(handles=parches, loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.5 / alto), ncol=4, frameon=False,
               fontsize=9.5, title="Aciertos", title_fontsize=9.5)

    margen = 0.3 / ancho
    alto_tabla = alto_fila * (len(filas) + 1)
    ax = fig.add_axes([margen, 0.2 / alto, 1 - 2 * margen, alto_tabla / alto])
    ax.axis("off")
    tabla = ax.table(cellText=texto, colLabels=cols, cellLoc="center",
                     loc="center", bbox=[0, 0, 1, 1],
                     colWidths=[0.22, 0.10, 0.10, 0.16, 0.16, 0.13, 0.13])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10.5)
    for (fi, ci), celda in tabla.get_celld().items():
        celda.set_edgecolor(COLOR_BORDE)
        if fi == 0:
            celda.set_facecolor("#3d4451")
            celda.get_text().set_color("white")
            celda.get_text().set_fontweight("bold")
            continue
        celda.set_facecolor("white" if fi % 2 else "#f7f7f7")
        if campos[ci] == "aciertos":
            celda.set_facecolor(COLORES_ESTADO[filas[fi - 1]["estado"]])
            celda.get_text().set_fontweight("bold")

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=DPI, facecolor="white")
    plt.close(fig)


def procesar_carpeta(nombre, dataset, entrada, out, budget=None, columnas=25):
    """Arma el video temporal, corre extract_keyframes, dibuja la tabla y
    devuelve (fila_resumen, encontrados, segundos)."""
    from keyseer.keyframes import extract_keyframes

    keys = entrada["keys_esperados"]
    ventanas = entrada["ventanas"]
    budget = max(1, keys[1]) if budget is None else budget
    tmp = Path(tempfile.mkdtemp(prefix="keyseer_vis_"))
    try:
        video = tmp / f"{nombre}.avi"
        n_frames = armar_video(Path(dataset) / nombre, video)
        t0 = time.time()
        res = extract_keyframes(str(video), budget=budget)
        segundos = time.time() - t0
        encontrados = sorted(int(k) for k in res.keyframe_indices)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    dibujar_tabla(nombre, n_frames, encontrados, ventanas, keys, budget,
                  Path(out) / f"{nombre}.png", columnas=columnas)
    fila = fila_resumen(nombre, n_frames, budget, keys, encontrados, ventanas)
    return fila, encontrados, segundos


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Tablas por carpeta del dataset: frames encontrados vs "
                    "rangos esperados.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--esperado", default=str(DEFAULT_ESPERADO))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", nargs="+", metavar="NOMBRE", default=None)
    ap.add_argument("--budget", type=int, default=None,
                    help="pisa el budget por defecto (extremo superior de "
                         "keys_esperados, minimo 1)")
    ap.add_argument("--columnas", type=int, default=25)
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
                nombre, args.dataset, entrada, args.out, budget=args.budget,
                columnas=args.columnas)
            print(f"{nombre:<17} frames={fila['frames']:<4} "
                  f"budget={fila['budget']:<2} encontrados={encontrados} "
                  f"aciertos={fila['aciertos']} extras={fila['extras']} "
                  f"({seg:.1f}s)", flush=True)
        except Exception as e:  # una carpeta rota no aborta las demas
            errores += 1
            print(f"{nombre:<17} ERROR: {type(e).__name__}: {e}", flush=True)
            budget = (args.budget if args.budget is not None
                      else max(1, entrada["keys_esperados"][1]))
            fila = fila_resumen(nombre, "-", budget,
                                entrada["keys_esperados"], None,
                                entrada["ventanas"])
        filas.append(fila)

    dibujar_resumen(filas, Path(args.out) / "resumen.png")
    print(f"resumen -> {Path(args.out) / 'resumen.png'}")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
