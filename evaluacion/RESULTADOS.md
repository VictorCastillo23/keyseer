# Resultados — Estudio comparativo (Opción C)

**Fecha:** 8 de septiembre de 2026 · **Última actualización:** 20 de septiembre de 2026 (§8-§9)
**Protocolo:** `PROTOCOLO.md` · **Código:** `evaluacion/baselines.py`, `evaluacion/harness.py`, `evaluacion/benchmark.py`
**Pregunta central:** ¿la consolidación de componentes (un modelo) iguala al
modelo dual (dos modelos) en calidad de detección, con menos memoria?

> **Alcance (20 sep).** Las §1-§7 son el estudio original sobre el **backend
> GMM** y la suite sintética `trap` (8 semillas); no se rehicieron. El camino
> práctico hoy es `blobtrack`: sus resultados sintéticos están en la §8 y los
> de video real (dataset local) en la §9.

**Respuesta corta: sí, en esta suite, y por una razón mecanicista específica
— no por ser "mejor" en general.** El modelo dual está diseñado para un
problema más estrecho (objetos que se detienen) que el que se evaluó
(eventos relevantes en general, incluyendo movimiento puro). Ver §3.

---

## 1. Resultado principal (8 semillas, mismo método de selección)

| método | recall eventos reales | señuelos capturados | redundancia | memoria |
|---|---|---|---|---|
| uniforme | 0.67 ± 0.00 | 1.00 ± 0.00 | 0.735 ± 0.008 | 0 |
| diferencia de frames | 0.33 ± 0.00 | 2.38 ± 0.48 | 0.635 ± 0.134 | 36.0 KB |
| MOG2 + ratio (v1, turno 1) | 0.67 ± 0.00 | 2.00 ± 0.00 | 0.460 ± 0.006 | n/d¹ |
| **modelo dual (fast/slow)** | 0.67 ± 0.00 | 1.25 ± 0.43 | 0.668 ± 0.121 | **720.0 KB** |
| Jacobs–Pless (aproximación)² | 0.67 ± 0.00 | 3.00 ± 0.00 | 0.462 ± 0.007 | 144.0 KB |
| **GMM, picos** | **1.00 ± 0.00** | 2.00 ± 0.00 | 0.462 ± 0.007 | **360.0 KB** |
| **GMM, submodular** | **1.00 ± 0.00** | **1.00 ± 0.00** | 0.694 ± 0.039 | **360.0 KB** |

¹ cv2 no expone el tamaño de su modelo interno vía Python.
² reimplementación aproximada, no validada contra las ecuaciones originales
del paper — ver `PROTOCOLO.md` §2.

**Lectura directa:** el backend GMM es el único método con recall perfecto en las 8
semillas, y con submodular iguala el mejor rechazo de señuelos, usando
**la mitad de memoria que el modelo dual**.

## 2. Por qué el modelo dual pierde: mecanismo verificado, no supuesto

El modelo dual (`slow_fg & ~fast_fg`) falla estructuralmente en el evento de
**movimiento puro** (un círculo cruzando la escena) en todas las semillas.
Verificado directamente sobre la señal:

| región | señal dual_bg (máx) |
|---|---|
| movimiento puro (evento real) | 0.016–0.019 |
| destello global (señuelo) | 0.017 |
| **ruido denso (señuelo)** | **0.78–0.98** |
| rectángulo persistente (evento real) | 0.094 |

El objeto en movimiento nunca permanece en un píxel el tiempo suficiente
para que el modelo rápido lo absorba de forma diferenciada del modelo
lento — ambos lo marcan como foreground de manera casi idéntica, así que la
resta se anula. **Esto no es un defecto de mi implementación: es lo que el
modelo dual está diseñado para hacer.** Su literatura de origen (detección
de objetos abandonados) define el problema exactamente como "objetos que
dejan de moverse", no como "eventos relevantes en general".

**El hallazgo más incómodo, verificado con trazas frame a frame:** el ruido
denso genera la señal *más fuerte de todas* en el modelo dual (hasta 0.98,
frente a 0.094 del objeto persistente real). Mecanismo confirmado paso a
paso: al llegar el ruido, la varianza del modelo rápido se infla en 2–3
frames (155→249 en las unidades del modelo) lo bastante para que el ruido
quede dentro del umbral de Mahalanobis y se reclasifique como "ya explicado
por el fondo" — precisamente *porque* es rápido. El modelo lento no alcanza
a hacer esa reclasificación en la misma ventana, y la resta se dispara. Es
un modo de fallo genuino y probablemente general de cualquier par
rápido/lento con diferencia de tasas de adaptación suficientemente grande,
no un artefacto de los hiperparámetros elegidos — se confirmó estable ante
seis combinaciones distintas de α_rápido/α_lento.

## 3. Interpretación correcta (y la incorrecta que hay que evitar)

**Incorrecta:** "el backend GMM es mejor que el modelo dual." El modelo dual no fue
diseñado para detectar movimiento puro; evaluarlo ahí y declarar victoria
sería medir con la vara equivocada.

**Correcta:** cuando el objetivo es "eventos relevantes para resumen" —que
incluye tanto movimiento como persistencia— **leer la dinámica de un solo
modelo generaliza mejor que restar dos modelos especializados en
detenerse**, y lo hace con la mitad de memoria. Esa es la afirmación que el
experimento sostiene, y es más estrecha y más defendible que la original.

## 4. Lo que se confirma sin ambigüedad: escalamiento de memoria

| T (frames) | GMM | VSUMM (k-means) | diferencia de frames |
|---|---|---|---|
| 500 | 360.0 KB | 2 000.0 KB | 36.0 KB |
| 2 000 | 360.0 KB | 8 000.0 KB | 36.0 KB |
| 8 000 | 360.0 KB | 32 000.0 KB | 36.0 KB |

VSUMM crece exactamente proporcional a T (debe guardar un descriptor por
frame para poder aplicar k-means al final); el backend GMM y frame-diff son O(1),
verificado empíricamente y no solo argumentado.

## 5. Amenaza a la validez que no se resolvió

Todo lo anterior es sobre la **suite de señuelos sintética**, diseñada por
mí conociendo de antemano las debilidades documentadas del backend GMM. El sesgo
potencial corre en ambas direcciones — también conocía sus fortalezas— pero
sigue siendo una suite pequeña (6 eventos, 8 semillas) y un solo generador
*(al 8 sep; desde entonces se añadió un segundo generador, `scale_contrast`, y
un dataset real local: §8-§9)*.
El resultado del vídeo real (`vtest.avi`) es puramente cualitativo, sin
ground truth, y no puede usarse para las cifras de recall.

**No se resolvió el Paso 2 pendiente de `ESTADO.md`**: el rechazo de ruido
denso sostenido sigue fallando en el backend GMM también (destello y ruido siguen
colándose como 1 de 6 keyframes con selección submodular). La corrección de
latencia (Paso 1) mejoró el recall de eventos reales pero **no** mejoró el
rechazo de ruido — se verificó explícitamente que `ruido_max` sigue en 1.000
con L=20 y con L=54 por igual.

## 6. Qué falta para que esto sea un paper

1. **Repetir con vídeo real con ground truth.** Sin acceso a PETS2006/ABODA
   en este entorno (§`PROTOCOLO.md` §4). Es el hueco más grave. *(Parcialmente
   atendido: la §9 usa video real con ventanas aproximadas, solo con
   `blobtrack` y muestreo uniforme; sigue sin haber ground truth por frame ni
   comparación con el resto de baselines.)*
2. **Ampliar la suite de señuelos** más allá de 6 eventos — idealmente
   generada por alguien que no conozca las debilidades del backend GMM, para
   eliminar el sesgo de diseño de la §5.
3. **Probar selección submodular también sobre los baselines** (requiere
   descriptores por frame para cada uno; no implementado aquí — actualmente
   solo el backend GMM tiene esa comparación).
4. **Resolver o acotar formalmente** el modo de fallo de ruido denso antes
   de afirmar robustez general.

## 7. Reproducibilidad

Todo el código está en `evaluacion/baselines.py` (métodos comparados) y
`evaluacion/harness.py` (generador de la suite y funciones de puntuación). El
vídeo real usado es público: `raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi`.
Las semillas 0–7 están fijadas; el experimento es reproducible exactamente
con ese rango.

---

## 8. Resultados sintéticos posteriores (backend `blobtrack`)

Todo lo de esta sección es **sintético**. Ver `PROTOCOLO.md` §7 para los
escenarios; las decisiones que se derivaron están en `docs/ESTADO.md` §10-§12.

### 8.1 `blobtrack` frente a GMM (suite `trap`, 8 semillas, submodular)

| | GMM NumPy (`core.py`) | `blobtrack` |
|---|---|---|
| recall de eventos reales | 1.00 ± 0.00 | 1.00 ± 0.00 |
| señuelos capturados | 1.00 ± 0.00 | **0.00 ± 0.00** |
| velocidad | ~17 fps (180×320) | ~345-455 fps |
| memoria de estado | 360 KB fijo | ~200 B-2 KB (crece con los tracks activos) |

### 8.2 Configuraciones de preprocesamiento (`trap`, 16 semillas)

`python -m evaluacion.benchmark --seeds 16` (ESTADO §11.2):

| config | recall_real | decoy_hits | elapsed (s) | state_bytes |
|---|---|---|---|---|
| baseline (color, sin gate) | 1.00 | 0.00 | 1.30 | 400 |
| grayscale_only | 1.00 | 0.00 | 1.23 | 250 |
| aggressive (gris + gate) | 1.00 | 0.00 | 0.45 | 312.5 |

### 8.3 Ponderación del score con señuelos que pasan los gates

`scale_contrast` con `gate_passing_decoys=True` a 1280×720, 8 semillas;
idéntico en los 4 tamaños de resize. Celdas = recall_real / decoy_hits por
presupuesto (b):

| config | pesos | b=3 | b=5 | b=8 |
|---|---|---|---|---|
| baseline | identity | 0.50/1.0 | 0.50/1.0 | 1.00/1.0 |
| baseline | binary | 1.00/0.0 | 1.00/1.0 | 1.00/1.0 |
| aggressive | identity | 0.50/0.0 | 0.50/1.0 | 1.00/2.0 |
| aggressive | binary | 1.00/0.0 | 1.00/0.0 | 1.00/2.0 |

Con `identity` el parche de luz grande domina (score pico 9.0 frente a 1.7 del
círculo y 0.19 del objeto chico); con `binary` entra la mota (0.056) en b=5
baseline. `binary` nunca captura más señuelos que `identity` y recupera el
evento chico, pero no es gratis; no está adoptado en `selection.py`. Sobre
`scale_contrast` sin esos señuelos, `binary` subió el recall de 0.5 a 1.0 en
b=3 y 5 con decoy 0.0, y todos los señuelos de `trap` tienen score exactamente
0 (ESTADO §12).

### 8.4 Barrido de tamaño de resize (`trap` 320×240 nativo + `scale_contrast_hd`)

| tamaño | trap recall/decoy (b=5, 8) | fps trap base/agg | fps HD base/agg |
|---|---|---|---|
| 180×320 | 1.00/0.00 | 311/218 | 151/77 |
| 270×480 | 1.00/0.00 | 205/150 | 116/63 |
| 360×640 | 1.00/0.00 | 138/104 | 91/54 |
| 540×960 (informativo) | 1.00/0.00 | 61/47 | 49/34 |

`trap` con b=3 da 0.667 en todos los tamaños (límite de presupuesto). La suite
no distingue tamaños; un cuadrado estático de 20 px en 1280×720 (1.6% del
ancho, 40 frames) desaparece a 320 y 480 de ancho y se detecta desde 640. Este
barrido motivó subir el default a (360, 640), decisión **revertida** a
(180, 320) por los datos reales de la §9 (ESTADO §12, "Reversion"). El fps es
ruidoso.

## 9. Resultados con datos reales (dataset local, no versionado)

Protocolo en `PROTOCOLO.md` §8. Camino por defecto: `blobtrack`, gris + motion
gate, 180×320, `submodular`, pesos `identity`; presupuesto = extremo superior
de `keys_esperados`. Aciertos = ventanas con ≥ 1 keyframe; extras = keyframes
fuera de toda ventana. **Ventanas anchas y aproximadas; ninguna cifra de esta
sección es concluyente.**

### 9.1 Por carpeta

| carpeta | budget | keyframes encontrados | aciertos | extras |
|---|---|---|---|---|
| Board | 4 | 15, 31, 130, 197 | 3/3 | 0 |
| Candela_m1_10 | 4 | 40, 97, 151, 320 | 2/3 | 2 |
| CAVIAR1 | 6 | 86, 164, 314, 357, 480, 560 | 4/4 | 1 |
| CAVIAR2 | 4 | 30, 64, 130, 428 | 2/3 | 1 |
| CaVignal | 2 | 166, 238 | 1/2 | 1 |
| HallAndMonitor | 5 | 39, 64, 91, 120, 290 | 3/4 | 0 |
| HumanBody2 | 7 | 52, 71, 304, 346, 495, 547, 578 | 3/5 | 2 |
| IBMtest2 | 3 | 35, 50, 65 | 3/3 | 0 |
| PeopleAndFoliage | 4 | 65, 99, 211, 275 | 2/4 | 1 |
| **total (9 con ventanas)** | | | **23/31** | **8** |
| Foliage | 1 | 113 | rango 0-1: dentro | |
| HighwayI | 6 | 22, 49, 66, 149, 256, 321 | rango 4-6: dentro | |
| HighwayII | 6 | 22, 50, 66, 117, 259, 385 | rango 4-6: dentro | |
| Snellen | 2 | 19, 104 | rango 1-2: dentro | |
| Toscana | 3 | (ninguno) | 2-3 esperados | |

- Candela: la ventana final (325-349) se pierde por 5 frames (el 320 es el
  hombre yéndose).
- HallAndMonitor: la ventana 3 (175-225) falla porque el score es ~0 ahí (sin
  tracks contados, no hay frames candidatos).
- Toscana: 0 keyframes sin error; 6 frames < `min_age_to_count=8`.
- Las discrepancias entre `dataset/analisis-frames.md` y los frames están en
  `docs/ESTADO.md` §13.2.

### 9.2 Comparación de configuraciones (9 carpetas, 31 ventanas)

| configuración | aciertos /31 | extras |
|---|---|---|
| **180×320 (default)** | **23** | **8** |
| 360×640 | 20 | 13 |
| nativo, sin ampliar | 19 | 11 |
| muestreo uniforme | 23 | 13 |
| selector por evento (experimento fuera del repo) | 15-16 | 4-5 |

Diferencias de 3-4 ventanas sobre 31. Casi todas las fuentes miden ≤ 384 px de
ancho (solo Toscana es 800×600), así que 360×640 es una prueba de ampliación.
El default (180, 320) se ratificó con estos mismos datos: no es un holdout.

### 9.3 Lo que se observa sobre el conteo de keyframes

- Con `budget=8`, Foliage (esperado 0-1), Snellen (1-2) y CaVignal (2) devuelven
  8 keyframes: **el conteo lo fija el presupuesto, no el contenido**.
- En HighwayI/II (flujo, presupuesto 6) la mayor separación entre keyframes
  consecutivos es 45% / 41% del video a 360×640 y 27% / 28% a 180×320, frente a
  20% del muestreo uniforme.
- La resta de fondo no puede proponer el frame 0, el estado inicial estático ni
  la "escena vacía"; "objeto abandonado" solo se aproxima por el evento de
  salida.

### 9.4 `min_gain_ratio` (3 clips, 180×320, budget 8, `identity`)

| ratio | CaVignal | Candela | HallAndMonitor |
|---|---|---|---|
| 0 (default) | 7 keyframes: 2/2 aciertos, 5 extras | 5 keyframes: 2/3, 2 extras | 8 keyframes: 3/4, 1 extra |
| 0.1 | 4 keyframes: 1/2, 3 extras | 2 keyframes: 1/3 | 2 keyframes: 2/4 |
| 0.2 | 1 keyframe, 0-1 aciertos | 1 keyframe, 0-1 aciertos | 1 keyframe, 0-1 aciertos |

Ningún umbral global lleva CaVignal a ~2 keyframes sin perder eventos reales en
otras carpetas; confirma la advertencia de `keyseer/selection.py`.

### 9.5 Selector por evento (experimento descartado)

Un criterio de parada por evento (clusters de tracks; no está en el repo) logró
en sintético recall 1.00 / 0.00 señuelos con 3.0 keyframes frente a 6.2 del
presupuesto 8 (8 semillas, `trap`), pero en real dio 15-16/31 aciertos frente a
23/31 y fusiona escenas largas en eventos gigantes (HallAndMonitor a 2/4;
CaVignal a 0/2 a 360×640). No adoptado; detalle en `docs/ESTADO.md` §13.7. Las
tablas de tracks reales se vieron antes de fijar el umbral de área, así que no
es un holdout limpio.

### 9.6 Reproducibilidad

Los PNG de `dataset/` no están en el repo. Las imágenes resultantes sí:
`evaluacion/resultados_dataset/` (tablas) y `evaluacion/resultados_grafica/`
(gráficas x-y), regenerables con `python -m evaluacion.visualizar_dataset` y
`python -m evaluacion.graficar_dataset` (requieren `pip install matplotlib`).
