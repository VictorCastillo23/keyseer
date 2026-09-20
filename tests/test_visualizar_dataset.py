"""
tests.test_visualizar_dataset
===============================

Logica pura de evaluacion/visualizar_dataset.py (clasificacion, categorias
por frame, armado del video temporal, tablas matplotlib). Ningun test toca
`dataset/` ni corre el algoritmo: todo se arma con datos sinteticos en
tmp_path.
"""

import json
from pathlib import Path

import numpy as np
import pytest

VENTANAS = [
    {"nombre": "a", "inicio": 10, "fin": 20},
    {"nombre": "b", "inicio": 30, "fin": 40},
]

JSON_ESPERADO = (Path(__file__).resolve().parents[1] / "evaluacion"
                 / "dataset_esperado.json")


# ---------------------------------------------------------------- clasificar

def test_clasificar_acierto_falla_y_extra():
    from evaluacion.visualizar_dataset import clasificar

    r = clasificar([15, 100], VENTANAS)

    assert r["aciertos"] == 1
    assert r["total_ventanas"] == 2
    assert r["ventanas"][0]["acierto"] is True
    assert r["ventanas"][0]["encontrados"] == [15]
    assert r["ventanas"][1]["acierto"] is False
    assert r["ventanas"][1]["encontrados"] == []
    assert r["extras"] == [100]
    assert r["redundantes"] == []


def test_clasificar_dos_keyframes_en_una_ventana_es_un_acierto_y_un_redundante():
    from evaluacion.visualizar_dataset import clasificar

    r = clasificar([12, 18], VENTANAS)

    assert r["aciertos"] == 1
    assert r["ventanas"][0]["encontrados"] == [12, 18]
    assert r["redundantes"] == [18]
    assert r["extras"] == []


def test_clasificar_bordes_inclusivos():
    from evaluacion.visualizar_dataset import clasificar

    r = clasificar([10, 40], VENTANAS)

    assert r["aciertos"] == 2
    assert r["extras"] == []


def test_clasificar_frame_en_borde_compartido_cuenta_para_ambas_ventanas():
    from evaluacion.visualizar_dataset import clasificar

    ventanas = [{"nombre": "p", "inicio": 0, "fin": 10},
                {"nombre": "q", "inicio": 10, "fin": 20}]

    r = clasificar([10], ventanas)

    assert r["aciertos"] == 2
    assert r["redundantes"] == []
    assert r["extras"] == []


def test_clasificar_sin_ventanas_todo_es_extra():
    from evaluacion.visualizar_dataset import clasificar

    r = clasificar([3, 7], [])

    assert r["aciertos"] == 0
    assert r["total_ventanas"] == 0
    assert r["extras"] == [3, 7]


def test_clasificar_sin_encontrados():
    from evaluacion.visualizar_dataset import clasificar

    r = clasificar([], VENTANAS)

    assert r["aciertos"] == 0
    assert r["extras"] == []
    assert r["redundantes"] == []


# ---------------------------------------------------- categoria_por_frame

def test_categoria_por_frame_prioriza_encontrado_sobre_esperado():
    from evaluacion.visualizar_dataset import categoria_por_frame

    cat = categoria_por_frame(50, [15, 45], VENTANAS)

    assert len(cat) == 50
    assert cat[15] == "encontrado"
    assert cat[45] == "encontrado"


def test_categoria_por_frame_ventanas_inclusivas_y_resto_normal():
    from evaluacion.visualizar_dataset import categoria_por_frame

    cat = categoria_por_frame(50, [15], VENTANAS)

    assert cat[10] == "esperado" and cat[20] == "esperado"
    assert cat[30] == "esperado" and cat[40] == "esperado"
    assert cat[9] == "normal" and cat[21] == "normal"
    assert cat[29] == "normal" and cat[41] == "normal"
    assert cat.count("encontrado") == 1


def test_categoria_por_frame_tolera_ventana_y_encontrados_fuera_de_rango():
    from evaluacion.visualizar_dataset import categoria_por_frame

    cat = categoria_por_frame(
        12, [99], [{"nombre": "x", "inicio": 8, "fin": 30}])

    assert len(cat) == 12
    assert cat[8:] == ["esperado"] * 4
    assert "encontrado" not in cat


# -------------------------------------------------------------- fila_resumen

def test_fila_resumen_estados_de_aciertos():
    from evaluacion.visualizar_dataset import fila_resumen

    todos = fila_resumen("X", 50, 4, [3, 4], [15, 35], VENTANAS)
    algunos = fila_resumen("X", 50, 4, [3, 4], [15], VENTANAS)
    ninguno = fila_resumen("X", 50, 4, [3, 4], [1], VENTANAS)
    sin = fila_resumen("X", 50, 4, [3, 4], [1], [])

    assert (todos["aciertos"], todos["estado"]) == ("2/2", "todos")
    assert (algunos["aciertos"], algunos["estado"]) == ("1/2", "algunos")
    assert (ninguno["aciertos"], ninguno["estado"]) == ("0/2", "ninguno")
    assert (sin["aciertos"], sin["estado"]) == ("-", "sin_ventanas")
    assert todos["esperado"] == "3-4"
    assert todos["encontrados"] == 2
    assert algunos["extras"] == 0
    assert ninguno["extras"] == 1


def test_fila_resumen_esperado_fijo_se_muestra_sin_rango():
    from evaluacion.visualizar_dataset import fila_resumen

    fila = fila_resumen("X", 50, 2, [2, 2], [15], VENTANAS)

    assert fila["esperado"] == "2"


def test_fila_resumen_error():
    from evaluacion.visualizar_dataset import fila_resumen

    fila = fila_resumen("X", 0, 4, [3, 4], None, VENTANAS)

    assert fila["estado"] == "error"
    assert fila["encontrados"] == "ERROR"


# --------------------------------------------------------------- armar_video

def _escribir_pngs(carpeta, nombres, w=32, h=24):
    import cv2
    carpeta.mkdir()
    for i, nombre in enumerate(nombres):
        img = np.full((h, w, 3), 10 + i * 20, np.uint8)
        assert cv2.imwrite(str(carpeta / nombre), img)


def _leer_video(ruta):
    import cv2
    cap = cv2.VideoCapture(str(ruta))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def test_armar_video_cuenta_frames_y_es_legible(tmp_path):
    pytest.importorskip("cv2")
    from evaluacion.visualizar_dataset import armar_video

    nombres = [f"X_m1.10_{i:06d}.png" for i in range(12)]
    _escribir_pngs(tmp_path / "png", nombres)
    destino = tmp_path / "salida.avi"

    n = armar_video(tmp_path / "png", destino, fps=25)

    assert n == 12
    assert destino.exists() and destino.stat().st_size > 0
    frames = _leer_video(destino)
    assert len(frames) == 12
    assert frames[0].shape[:2] == (24, 32)


def test_armar_video_ordena_por_sufijo_numerico(tmp_path):
    pytest.importorskip("cv2")
    from evaluacion.visualizar_dataset import armar_video

    # orden lexicografico serian 1, 10, 2: el numerico es 1, 2, 10
    _escribir_pngs(tmp_path / "png", ["X_1.png", "X_2.png", "X_10.png"])
    destino = tmp_path / "salida.avi"

    armar_video(tmp_path / "png", destino)

    niveles = [int(f.mean()) for f in _leer_video(destino)]
    assert niveles == pytest.approx([10, 30, 50], abs=6)


def test_armar_video_carpeta_sin_png_falla(tmp_path):
    from evaluacion.visualizar_dataset import armar_video

    (tmp_path / "vacia").mkdir()

    with pytest.raises(ValueError):
        armar_video(tmp_path / "vacia", tmp_path / "salida.avi")


# ------------------------------------------------------------ dibujar_tabla

def _es_png_no_vacio(ruta):
    ruta = Path(ruta)
    return ruta.exists() and ruta.stat().st_size > 0 and \
        ruta.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_dibujar_tabla_con_ventanas(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.visualizar_dataset import dibujar_tabla

    destino = tmp_path / "con_ventanas.png"
    r = dibujar_tabla("Sintetica", 57, [15, 35, 50], VENTANAS, [3, 4], 4,
                      destino, columnas=10)

    assert _es_png_no_vacio(destino)
    assert r["aciertos"] == 2
    assert r["extras"] == [50]


def test_dibujar_tabla_sin_ventanas(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.visualizar_dataset import dibujar_tabla

    destino = tmp_path / "sin_ventanas.png"
    dibujar_tabla("Sintetica", 40, [12], [], [0, 1], 1, destino, columnas=10)

    assert _es_png_no_vacio(destino)


def test_dibujar_tabla_sin_keyframes(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.visualizar_dataset import dibujar_tabla

    destino = tmp_path / "cero.png"
    dibujar_tabla("Chica", 6, [], [], [2, 3], 3, destino, columnas=25)

    assert _es_png_no_vacio(destino)


def test_dibujar_tabla_crea_carpeta_destino(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.visualizar_dataset import dibujar_tabla

    destino = tmp_path / "no" / "existe" / "t.png"
    dibujar_tabla("S", 20, [3], VENTANAS, [1, 2], 2, destino, columnas=7)

    assert _es_png_no_vacio(destino)


def test_dibujar_resumen(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.visualizar_dataset import dibujar_resumen, fila_resumen

    filas = [
        fila_resumen("A", 50, 4, [3, 4], [15, 35], VENTANAS),
        fila_resumen("B", 50, 4, [3, 4], [15], VENTANAS),
        fila_resumen("C", 50, 4, [3, 4], [1], VENTANAS),
        fila_resumen("D", 50, 1, [0, 1], [], []),
        fila_resumen("E", 0, 1, [0, 1], None, []),
    ]
    destino = tmp_path / "resumen.png"

    dibujar_resumen(filas, destino)

    assert _es_png_no_vacio(destino)


# ---------------------------------------------------------- dataset_esperado

def test_dataset_esperado_json_es_consistente():
    from evaluacion.visualizar_dataset import cargar_esperado

    data = json.loads(JSON_ESPERADO.read_text(encoding="utf-8"))
    assert "inspeccion visual" in data["_nota"]
    carpetas = cargar_esperado(JSON_ESPERADO)
    assert len(carpetas) == 14

    for nombre, entrada in carpetas.items():
        lo, hi = entrada["keys_esperados"]
        assert 0 <= lo <= hi, nombre
        previo_fin = -1
        nombres = set()
        for v in entrada["ventanas"]:
            assert v["inicio"] <= v["fin"], (nombre, v)
            # ordenadas y sin solape; se tolera UN frame de borde compartido
            # (CAVIAR1: mujer_entra termina en 360 y cruce_de_sujetos empieza en 360)
            assert v["inicio"] >= previo_fin, (nombre, v)
            previo_fin = v["fin"]
            assert v["nombre"] not in nombres, (nombre, v)
            nombres.add(v["nombre"])


def test_cargar_esperado_ignora_claves_con_guion_bajo(tmp_path):
    from evaluacion.visualizar_dataset import cargar_esperado

    ruta = tmp_path / "e.json"
    ruta.write_text(json.dumps({"_nota": "x", "carpetas": {
        "A": {"keys_esperados": [1, 2], "ventanas": []}}}), encoding="utf-8")

    assert list(cargar_esperado(ruta)) == ["A"]
