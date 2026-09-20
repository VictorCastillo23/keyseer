# KeySeer

Detección de keyframes por **consolidación de componentes** en modelos de
mezcla gaussiana online.

## La idea

MOG2 (Zivkovic, 2006) mantiene un GMM por píxel y lo actualiza con un prior
de Dirichlet de coeficiente negativo, que poda automáticamente las
componentes espurias. Ese mecanismo de poda es, de hecho, **un test de
hipótesis implícito sobre la novedad**: una componente nacida por ruido se
extingue en pocos frames; una nacida por contenido genuino consolida y crece.

MOG2 descarta esa información al colapsar todo a una máscara binaria.
KeySeer la lee explícitamente:

```
Ψ_t(x) = peso, L frames después, de la componente nacida en t
```

**Proposición 1.** Sean dos eventos idénticos hasta el frame t, uno
transitorio y otro persistente. Todo funcional causal e instantáneo del
modelo en t les asigna el mismo valor. La discriminación transitorio /
persistente por tanto *requiere* latencia.

Verificado empíricamente (`tests/test_core.py`): en el frame de inicio,
nacimientos y KL son idénticos para ambos eventos; Ψ los separa >10x.

## Dos backends

| backend | qué es | cuándo |
|---|---|---|
| `blobtrack` (por defecto) | `cv2.createBackgroundSubtractorMOG2` + edad de blobs rastreados por centroide | camino práctico; el que se evaluó en video real |
| `gmm` | GMM por píxel en NumPy con consolidación explícita (`keyseer/core.py`) | artefacto de investigación; expone Ψ, KL y sorpresa |

En la suite sintética de señuelos `blobtrack` igualó al GMM en recall (1.00) y
capturó 0.00 señuelos (GMM: 1.00), a ~345-455 fps frente a ~17 fps
(`docs/ESTADO.md` §10.1).

## Instalación

```bash
pip install -e ".[all]"
```

Extras: `video` (opencv-python; necesario para leer video y para `blobtrack`),
`peaks` (scipy; necesario para `method="peaks"`). Solo `numpy` es obligatorio
para el núcleo GMM (`keyseer.core`) y las métricas. Los scripts de
visualización sobre el dataset real necesitan además `pip install matplotlib`
(no es un extra declarado).

## Uso

```python
from keyseer import extract_keyframes

res = extract_keyframes(
    "video.mp4",
    budget=8,                # máximo de keyframes
    out_dir="keyframes/",    # opcional: guarda keyframe_NNNNNN.jpg
    method="submodular",     # por defecto; garantía (1 - 1/e)
)
print(res.keyframe_indices)  # números de frame REALES del video
print(res.timestamps())      # segundos (None si no se pudo leer el fps)
```

`extract_keyframes` devuelve un `KeyframeResult`:

| atributo | contenido |
|---|---|
| `keyframe_indices` | números de frame reales del video fuente |
| `dense_positions` | índices densos (frames procesados), alineados con `keyframe_indices`; sirven para indexar `scores` |
| `scores` | señal de novedad, una entrada por frame *procesado* (el motion gate de `blobtrack` salta frames) |
| `n_frames` | cantidad de frames procesados (`len(scores)`) |
| `elapsed`, `fps_source`, `backend`, `saved_paths` | segundos de análisis, fps del video, backend usado, rutas de los .jpg |

```python
res.scores[res.dense_positions]   # score en cada keyframe elegido
```

Parámetros útiles de `extract_keyframes(...)`:

- `backend="blobtrack" | "gmm"`; `backend_kwargs` se pasa a
  `BlobPersistenceTracker` (blobtrack) o a `KeySeerConfig` (gmm), p. ej.
  `extract_keyframes("video.mp4", backend="gmm", backend_kwargs=dict(alpha=0.02, latency=20))`.
- `resize_to=(alto, ancho)`: `None` (por defecto) usa el tamaño por backend,
  hoy `(180, 320)` en ambos (`keyseer.blobtrack.DEFAULT_RESIZE_TO`,
  `keyseer.keyframes.GMM_RESIZE_TO`). **`None` no significa "sin reducir"**:
  para resolución nativa hay que pasar el tamaño del video.
- `grayscale`, `motion_gate`, `motion_gate_kwargs`: solo `blobtrack`;
  `grayscale` y `motion_gate` están activos por defecto (gris + stride
  adaptativo por movimiento).
- `budget` es un máximo, pero no hay criterio de parada por contenido: en el
  dataset real el conteo lo fija el presupuesto (`docs/ESTADO.md` §13).

Acceso de bajo nivel al modelo GMM:

```python
from keyseer import KeySeer, KeySeerConfig
m = KeySeer(H, W, 3, KeySeerConfig(alpha=0.02, latency=20))
for frame in frames:         # frames (H, W, 3)
    r = m.update(frame)      # r.surprisal, r.kl, r.births, r.consolidation
```

`r.consolidation` es `None` durante los primeros `latency` frames; después es
el mapa Ψ del frame `r.consolidation_index`.

## Señales del backend GMM

| señal | tipo | qué mide |
|---|---|---|
| `surprisal` | instantánea | −log p(I_t \| θ_{t−1}), bits de sorpresa |
| `kl` | instantánea | KL(θ_t ‖ θ_{t−1}), sorpresa bayesiana (Itti–Baldi) |
| `births` | instantánea | nacimiento de componente (cambio de soporte) |
| `consolidation` | latencia L | Ψ: masa de novedad que sobrevivió |
| `consolidation_ratio` | latencia L | Ψ por nacimiento — **independiente del área** |
| `coherence` | latencia L | estructura espacial del mapa Ψ |

`blobtrack` no expone estas señales: su score suma, sobre los tracks que
superan una edad mínima (`min_age_to_count`), el área relativa × la edad
(acotada por `age_cap`).

## Propiedades

- Una sola pasada, online. GMM: memoria O(H·W·M), independiente de la
  duración; `blobtrack`: O(tracks activos).
- Sin datos de entrenamiento.
- GMM: 47 fps a 120×160 / 17 fps a 180×320 en NumPy puro; **por debajo de
  tiempo real** a resoluciones útiles (`docs/ESTADO.md` §4.3). `blobtrack`:
  del orden de 200-450 fps a 180×320 en la suite sintética (medición
  ruidosa; `docs/ESTADO.md` §10.1 y §12).
- Latencia acotada L en el GMM (requerida por la Proposición 1, no un defecto).

## Diferencias deliberadas respecto a OpenCV MOG2

La actualización de varianza del GMM usa el estimador MLE isotrópico
`‖δ‖²/C`. OpenCV usa `‖δ‖²` sin dividir entre C, lo que sesga σ² por un
factor C en vídeo a color.

`blobtrack` fuerza `learningRate=1/history` en `apply()`: con la tasa
automática de OpenCV (~1/frames_vistos) un objeto que aparece al inicio del
video se absorbe al fondo mucho más rápido que uno idéntico que aparece tarde
(`docs/ESTADO.md` §10.2).

## Evaluación

`evaluacion/` no forma parte del paquete instalable. Sus comandos
(`python -m evaluacion...`) se corren desde la raíz del repo.

Tests:

```bash
pytest tests/          # 116 tests con matplotlib instalado; sin él se saltan los que dibujan
```

Suite sintética de señuelos (`evaluacion/harness.py`):

```bash
python -m evaluacion.benchmark --seeds 8                          # baseline / grayscale_only / aggressive
python -m evaluacion.benchmark --experiment weighting --seeds 8   # ponderación identity vs binary
python -m evaluacion.benchmark --experiment resolution --seeds 8  # barrido de resize_to
```

Dataset real, **solo local**: `dataset/` (14 carpetas de PNG consecutivos, con
`dataset/analisis-frames.md`) está en `.gitignore` (unos 620 MB) y no se
versiona ni se distribuye. Los scripts esperan
`dataset/<Carpeta>/<Carpeta>_NNNNNN.png` (ordenan por el sufijo numérico; en
`Candela_m1_10` los archivos se llaman `Candela_m1.10_NNNNNN.png`). Las
ventanas de eventos y los rangos de keyframes esperados están en
`evaluacion/dataset_esperado.json` (ventanas fijadas por inspección visual
antes de correr el algoritmo).

```bash
pip install matplotlib
python -m evaluacion.visualizar_dataset [--only NOMBRE ...] [--budget N] [--out DIR]  # tablas -> evaluacion/resultados_dataset/
python -m evaluacion.graficar_dataset   [--only NOMBRE ...] [--budget N] [--out DIR]  # gráficas x-y -> evaluacion/resultados_grafica/
```

Cada carpeta se ensambla en un video temporal sin pérdida y se procesa con el
camino por defecto (`blobtrack`, gris + motion gate, 180×320, submodular).
Las imágenes de `resultados_dataset/` y `resultados_grafica/` sí están
versionadas y se regeneran con esos comandos (sin `--out` las sobrescriben).

Protocolo y resultados: `evaluacion/PROTOCOLO.md`, `evaluacion/RESULTADOS.md`.

## Estado: prototipo de investigación, NO listo para uso

La contribución teórica (Ψ separa transitorio de persistente) está validada.
La métrica compuesta del backend GMM **no** separa eventos reales de
señuelos: la separación es negativa en las cuatro variantes probadas. El
camino práctico es `blobtrack`: rechaza los señuelos de la suite sintética
(0.00 capturados) y se probó de punta a punta sobre 14 carpetas de video real
(dataset local). En las 9 carpetas con ventanas de evento (31 ventanas)
acertó 23 con 8 keyframes extra; el muestreo uniforme también acertó 23, con
13 extras. La diferencia es de pocos keyframes y las ventanas son anchas:
**no es concluyente**.

Limitaciones verificadas, con datos y agenda en
**[`docs/ESTADO.md`](docs/ESTADO.md)**:

1. **El GMM no rechaza ráfagas de ruido denso sostenido** (limitación
   bloqueante del backend de investigación). `blobtrack` las rechaza en la
   suite sintética.
2. **El término de coherencia espacial es contraproducente** para destellos
   globales — refutado por ablación, pendiente de reemplazo.
3. **Latencia del GMM.** Debe cumplir `L ≥ ln(c_T/(α+c_T)) / ln(1−α)`: ≈69
   frames con los defaults de `KeySeerConfig` (`alpha=0.01`, `c_T=0.01`;
   `latency=12`), ≈54 con `alpha=0.02`. Probar L=54 en lugar de 20 mejoró el
   recall pero no el rechazo de ruido (`evaluacion/RESULTADOS.md` §5).
4. **Validación en video real limitada**: 14 carpetas, casi todas de ≤384 px
   de ancho, ventanas de evento anchas (los frames esperados son
   aproximados). Sin video de alta resolución.
5. **Requiere cámara fija.** En `Foliage` y `PeopleAndFoliage` la cámara se
   mueve (el encuadre cambia), lo que viola el supuesto.
6. **El número de keyframes lo fija `budget`, no el contenido**, y el método no
   puede proponer el frame 0 ni un estado inicial/"escena vacía" (resta de
   fondo). Ver `docs/ESTADO.md` §13.
7. **Los primeros ~1/α frames** del GMM son de calentamiento y no son fiables.

## Referencias

- Stauffer & Grimson (1999), *Adaptive background mixture models*.
- Zivkovic (2004); Zivkovic & van der Heijden (2006).
- Itti & Baldi (2009), *Bayesian surprise attracts human attention*.
- Nemhauser et al. (1978), garantía voraz para submodulares monótonas.
