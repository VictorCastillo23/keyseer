"""
tests.test_graficar_dataset
============================

Logica de evaluacion/graficar_dataset.py: preparacion de datos (marcadores
sobre la curva, numeracion de ventanas, niveles de etiquetas), dibujo
matplotlib (grafica por carpeta y resumen x-y) y un test de integracion que
verifica que los frames encontrados coinciden con extract_keyframes. Ningun
test toca `dataset/`.
"""

import json
from pathlib import Path

import numpy as np
import pytest

VENTANAS = [
    {"nombre": "a", "inicio": 10, "fin": 20},
    {"nombre": "b", "inicio": 30, "fin": 40},
]


def _es_png_no_vacio(ruta):
    ruta = Path(ruta)
    return ruta.exists() and ruta.stat().st_size > 0 and \
        ruta.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _curva_sintetica(n_frames=60, paso=3):
    idx = np.arange(0, n_frames, paso)
    sc = np.zeros(len(idx))
    sc[(idx >= 12) & (idx <= 24)] = 0.8
    sc[(idx >= 33) & (idx <= 39)] = 0.4
    return sc, idx


# ---------------------------------------------------- marcadores_encontrados

def test_marcadores_encontrados_busca_el_score_por_frame_real():
    from evaluacion.graficar_dataset import marcadores_encontrados

    idx = np.array([0, 3, 6, 9])
    sc = np.array([0.1, 0.5, 0.9, 0.2])

    m = marcadores_encontrados([9, 3], sc, idx)

    assert [(d["frame"], d["score"]) for d in m] == [(3, 0.5), (9, 0.2)]


def test_marcadores_encontrados_frame_sin_punto_de_score_va_a_cero():
    from evaluacion.graficar_dataset import marcadores_encontrados

    idx = np.array([0, 3, 6, 9])
    sc = np.array([0.1, 0.5, 0.9, 0.2])

    m = marcadores_encontrados([4, 6], sc, idx)

    assert [(d["frame"], d["score"]) for d in m] == [(4, 0.0), (6, 0.9)]


def test_marcadores_encontrados_con_curva_vacia_todo_va_a_cero():
    from evaluacion.graficar_dataset import marcadores_encontrados

    m = marcadores_encontrados([5, 8], np.array([]), np.array([], dtype=int))

    assert [(d["frame"], d["score"]) for d in m] == [(5, 0.0), (8, 0.0)]
    assert marcadores_encontrados([], np.array([]), np.array([], dtype=int)) == []


def test_marcadores_encontrados_devuelve_tipos_python():
    from evaluacion.graficar_dataset import marcadores_encontrados

    m = marcadores_encontrados(np.array([3]), np.array([0.5]), np.array([3]))

    assert type(m[0]["frame"]) is int
    assert type(m[0]["score"]) is float


# -------------------------------------------------------------- preparar_datos

def test_preparar_datos_numera_las_ventanas_desde_uno_en_orden():
    from evaluacion.graficar_dataset import preparar_datos

    sc, idx = _curva_sintetica()

    d = preparar_datos(60, [15, 50], VENTANAS, sc, idx)

    assert [w["numero"] for w in d["ventanas"]] == [1, 2]
    assert [w["nombre"] for w in d["ventanas"]] == ["a", "b"]
    assert [w["acierto"] for w in d["ventanas"]] == [True, False]
    assert d["clasificacion"]["aciertos"] == 1
    assert d["clasificacion"]["extras"] == [50]


def test_preparar_datos_curva_y_marcadores_alineados():
    from evaluacion.graficar_dataset import preparar_datos

    sc, idx = _curva_sintetica()

    d = preparar_datos(60, [15, 50], VENTANAS, sc, idx)

    assert d["x"].tolist() == idx.tolist()
    assert d["y"].tolist() == sc.tolist()
    marcas = {m["frame"]: m["score"] for m in d["encontrados"]}
    assert marcas == {15: 0.8, 50: 0.0}


def test_preparar_datos_sin_ventanas_ni_curva():
    from evaluacion.graficar_dataset import preparar_datos

    d = preparar_datos(6, [], [], np.array([]), np.array([], dtype=int))

    assert d["ventanas"] == []
    assert d["encontrados"] == []
    assert len(d["x"]) == 0 and len(d["y"]) == 0
    assert d["clasificacion"]["total_ventanas"] == 0


def test_preparar_datos_rechaza_curva_desalineada():
    from evaluacion.graficar_dataset import preparar_datos

    with pytest.raises(ValueError):
        preparar_datos(60, [], [], np.array([0.1, 0.2]), np.array([0, 3, 6]))


# ------------------------------------------------------------ asignar_niveles

def test_asignar_niveles_frames_lejanos_quedan_en_el_primer_nivel():
    from evaluacion.graficar_dataset import asignar_niveles

    assert asignar_niveles([10, 200, 400, 600], 740) == [0, 0, 0, 0]


def test_asignar_niveles_frames_cercanos_se_reparten_en_niveles():
    from evaluacion.graficar_dataset import asignar_niveles

    niveles = asignar_niveles([39, 64, 91, 120], 740)

    assert niveles[0] == 0
    assert niveles[1] != niveles[0]     # 64 esta pegado a 39
    assert len(set(niveles)) > 1
    assert all(0 <= n < 3 for n in niveles)


def test_asignar_niveles_nunca_repite_nivel_en_vecinos_pegados():
    from evaluacion.graficar_dataset import asignar_niveles

    frames = [100, 101, 102, 103, 104, 105, 106]
    niveles = asignar_niveles(frames, 740)

    assert len(niveles) == len(frames)
    assert all(a != b for a, b in zip(niveles, niveles[1:]))
    assert all(0 <= n < 3 for n in niveles)


def test_asignar_niveles_vacio():
    from evaluacion.graficar_dataset import asignar_niveles

    assert asignar_niveles([], 100) == []


# ---------------------------------------------------- fila_resumen_grafica

def test_fila_resumen_grafica_agrega_rango_numerico_y_conteo():
    from evaluacion.graficar_dataset import fila_resumen_grafica

    fila = fila_resumen_grafica("X", 50, 4, [3, 4], [15, 35], VENTANAS)

    assert fila["keys_min"] == 3 and fila["keys_max"] == 4
    assert fila["n_encontrados"] == 2
    assert fila["aciertos"] == "2/2" and fila["estado"] == "todos"


def test_fila_resumen_grafica_error_no_tiene_conteo():
    from evaluacion.graficar_dataset import fila_resumen_grafica

    fila = fila_resumen_grafica("X", "-", 4, [3, 4], None, VENTANAS)

    assert fila["estado"] == "error"
    assert fila["n_encontrados"] is None
    assert fila["keys_min"] == 3 and fila["keys_max"] == 4


# ------------------------------------------------------------ dibujar_grafica

def test_dibujar_grafica_con_ventanas(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.graficar_dataset import dibujar_grafica

    sc, idx = _curva_sintetica()
    destino = tmp_path / "con_ventanas.png"

    r = dibujar_grafica("Sintetica", 60, [15, 35, 50], VENTANAS, [3, 4], 4,
                        sc, idx, destino)

    assert _es_png_no_vacio(destino)
    assert r["aciertos"] == 2
    assert r["extras"] == [50]


def test_dibujar_grafica_sin_ventanas(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.graficar_dataset import dibujar_grafica

    sc, idx = _curva_sintetica()
    destino = tmp_path / "sin_ventanas.png"

    dibujar_grafica("Sintetica", 60, [15], [], [0, 1], 1, sc, idx, destino)

    assert _es_png_no_vacio(destino)


def test_dibujar_grafica_sin_keyframes_y_curva_vacia(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.graficar_dataset import dibujar_grafica

    destino = tmp_path / "cero.png"

    dibujar_grafica("Chica", 6, [], [], [2, 3], 3, np.array([]),
                    np.array([], dtype=int), destino)

    assert _es_png_no_vacio(destino)


def test_dibujar_grafica_curva_toda_en_cero(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.graficar_dataset import dibujar_grafica

    destino = tmp_path / "ceros.png"

    dibujar_grafica("Chica", 6, [], [], [2, 3], 3, np.zeros(6), np.arange(6),
                    destino)

    assert _es_png_no_vacio(destino)


def test_dibujar_grafica_crea_carpeta_destino(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.graficar_dataset import dibujar_grafica

    sc, idx = _curva_sintetica()
    destino = tmp_path / "no" / "existe" / "g.png"

    dibujar_grafica("S", 60, [15], VENTANAS, [1, 2], 2, sc, idx, destino)

    assert _es_png_no_vacio(destino)


def test_dibujar_grafica_con_pico_dominante_y_muchos_frames_cercanos(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.graficar_dataset import dibujar_grafica

    idx = np.arange(0, 740)
    sc = np.zeros(740)
    sc[100:130] = 1.0
    sc[300] = 400.0
    destino = tmp_path / "pico.png"

    dibujar_grafica("Grande", 740, [39, 64, 91, 120, 300], VENTANAS,
                    [4, 5], 5, sc, idx, destino)

    assert _es_png_no_vacio(destino)


# ------------------------------------------------------ dibujar_resumen_grafica

def test_dibujar_resumen_grafica(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.graficar_dataset import (dibujar_resumen_grafica,
                                             fila_resumen_grafica)

    filas = [
        fila_resumen_grafica("A", 50, 4, [3, 4], [15, 35], VENTANAS),
        fila_resumen_grafica("B", 50, 4, [3, 4], [15], VENTANAS),
        fila_resumen_grafica("C", 50, 4, [3, 4], [1], VENTANAS),
        fila_resumen_grafica("D", 50, 1, [0, 1], [], []),
        fila_resumen_grafica("E", 50, 1, [2, 2], [4], []),
        fila_resumen_grafica("F", "-", 1, [0, 1], None, []),
    ]
    destino = tmp_path / "resumen.png"

    dibujar_resumen_grafica(filas, destino)

    assert _es_png_no_vacio(destino)


def test_dibujar_resumen_grafica_una_sola_fila_con_error(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluacion.graficar_dataset import (dibujar_resumen_grafica,
                                             fila_resumen_grafica)

    filas = [fila_resumen_grafica("Z", "-", 3, [3, 4], None, VENTANAS)]
    destino = tmp_path / "solo_error.png"

    dibujar_resumen_grafica(filas, destino)

    assert _es_png_no_vacio(destino)


# --------------------------------------------------------- integracion (video)

def test_analizar_video_coincide_con_extract_keyframes(tiny_video):
    pytest.importorskip("cv2")
    from keyseer.keyframes import extract_keyframes
    from evaluacion.graficar_dataset import analizar_video

    video = tiny_video(n_frames=120, event=(20, 100))

    encontrados, scores, frame_indices = analizar_video(video, budget=3)

    esperado = extract_keyframes(video, budget=3)
    assert encontrados == sorted(esperado.keyframe_indices)
    # la curva es la misma que vio el selector: mismos puntos y frames reales
    assert len(scores) == len(frame_indices) == len(esperado.scores)
    assert np.allclose(scores, esperado.scores)
    assert all(k in set(frame_indices.tolist()) for k in encontrados)


def _escribir_pngs_ruidosos(carpeta, n=60, w=64, h=48, evento=(15, 45)):
    import cv2
    carpeta.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        f = np.full((h, w, 3), 40, np.uint8)
        f = cv2.add(f, rng.integers(0, 3, f.shape, dtype=np.uint8))
        if evento[0] <= i < evento[1]:
            cx = 10 + (i - evento[0]) * 1
            cv2.rectangle(f, (cx, 15), (cx + 10, 30), (0, 180, 0), -1)
        assert cv2.imwrite(str(carpeta / f"Sint_{i:06d}.png"), f)


def test_procesar_carpeta_guarda_png_y_devuelve_fila(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    from evaluacion.graficar_dataset import procesar_carpeta

    _escribir_pngs_ruidosos(tmp_path / "dataset" / "Sint")
    entrada = {"keys_esperados": [1, 2],
               "ventanas": [{"nombre": "evento", "inicio": 15, "fin": 45}]}

    fila, encontrados, seg = procesar_carpeta(
        "Sint", tmp_path / "dataset", entrada, tmp_path / "out")

    assert _es_png_no_vacio(tmp_path / "out" / "Sint.png")
    assert fila["carpeta"] == "Sint" and fila["frames"] == 60
    assert fila["budget"] == 2                # extremo superior de keys_esperados
    assert fila["n_encontrados"] == len(encontrados)
    assert all(0 <= k < 60 for k in encontrados)
    assert seg >= 0


# ------------------------------------------------------------------------ main

def test_main_procesa_carpetas_aisla_errores_y_devuelve_1(tmp_path, capsys):
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    from evaluacion.graficar_dataset import main

    _escribir_pngs_ruidosos(tmp_path / "dataset" / "Sint")
    (tmp_path / "dataset" / "Vacia").mkdir()
    esperado = tmp_path / "esperado.json"
    esperado.write_text(json.dumps({"_nota": "x", "carpetas": {
        "Sint": {"keys_esperados": [1, 2], "ventanas": [
            {"nombre": "evento", "inicio": 15, "fin": 45}]},
        "Vacia": {"keys_esperados": [1, 1], "ventanas": []},
    }}), encoding="utf-8")
    out = tmp_path / "out"

    codigo = main(["--dataset", str(tmp_path / "dataset"),
                   "--esperado", str(esperado), "--out", str(out)])

    assert codigo == 1
    assert _es_png_no_vacio(out / "Sint.png")
    assert not (out / "Vacia.png").exists()
    assert _es_png_no_vacio(out / "resumen.png")
    salida = capsys.readouterr().out
    assert "Sint" in salida and "Vacia" in salida and "ERROR" in salida


def test_main_only_filtra_y_carpeta_desconocida_falla(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    from evaluacion.graficar_dataset import main

    _escribir_pngs_ruidosos(tmp_path / "dataset" / "Sint")
    esperado = tmp_path / "esperado.json"
    esperado.write_text(json.dumps({"carpetas": {
        "Sint": {"keys_esperados": [1, 2], "ventanas": []}}}),
        encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["--dataset", str(tmp_path / "dataset"),
              "--esperado", str(esperado), "--out", str(tmp_path / "o"),
              "--only", "NoExiste"])

    codigo = main(["--dataset", str(tmp_path / "dataset"),
                   "--esperado", str(esperado), "--out", str(tmp_path / "o2"),
                   "--only", "Sint", "--budget", "1"])

    assert codigo == 0
    assert _es_png_no_vacio(tmp_path / "o2" / "Sint.png")
    assert _es_png_no_vacio(tmp_path / "o2" / "resumen.png")
