# Protocolo — Estudio comparativo de métodos clásicos de keyframes/resumen
## (Opción C: la contribución es la evidencia, no el mecanismo)

**Fecha:** 8 de septiembre de 2026 · **Última actualización:** 20 de septiembre de 2026 (§7-§8)

> **Alcance (20 sep).** Las §1-§6 son el protocolo original (Opción C, sintético
> más `vtest.avi`) y se conservan. Después se añadieron: experimentos sintéticos
> sobre el backend `blobtrack` (§7) y una evaluación sobre un dataset real local
> (§8). Los resultados están en `RESULTADOS.md` y `docs/ESTADO.md` §10-§13.

## 1. Pregunta de investigación

No "¿es el backend GMM mejor?" — esa pregunta ya perdió con la revisión de literatura.
La pregunta honesta es:

> **¿Leer la dinámica interna de un único modelo de mezcla gaussiana
> (consolidación de componentes) iguala la calidad de detección de un modelo
> de fondo dual (fast/slow), con menos memoria?**

Esta es la comparación que la revisión de literatura identificó como la
correcta y que nunca se hizo. Es una pregunta empírica acotada, no una
afirmación de novedad de mecanismo.

Pregunta secundaria, más barata de responder y útil como contexto:

> ¿Cómo se comparan estos métodos clásicos basados en modelos de fondo
> (consolidación, dual, Jacobs–Pless multiescala, MOG2+ratio) contra
> baselines simples (muestreo uniforme, diferencia de frames, clustering
> VSUMM) en robustez a señuelos, redundancia y memoria?

## 2. Métodos comparados

| método | memoria | fuente |
|---|---|---|
| Uniforme | O(1) | trivial |
| Diferencia de frames | O(1) | clásico |
| MOG2 + ratio de foreground (mi v1, turno 1) | O(HW) | cv2 |
| VSUMM (histograma + k-means) | **O(T)** — guarda un descriptor por frame | Avila et al. 2011 |
| Modelo dual fast/slow | O(2·HWM) | Park et al. 2019; patrón estándar de objetos abandonados |
| Jacobs–Pless multiescala (aproximación) | O(HWK) | Jacobs & Pless 2006/2008 — **reimplementación aproximada**, no validada contra las ecuaciones originales del paper |
| Consolidación (backend GMM) | O(HWM) | este trabajo |

**Nota de honestidad metodológica:** no tengo el texto completo de Jacobs &
Pless, solo resúmenes. Mi implementación es un banco de medias móviles
exponenciales a escalas temporales espaciadas geométricamente — captura la
idea (multiescala, causal, memoria constante) pero no reproduce sus filtros
exactos. Se reporta como aproximación, no como reproducción.

## 3. Métricas

1. **Robustez a señuelos**: recall sobre eventos reales (¿al menos un
   keyframe cae dentro de cada evento real?) y violaciones sobre señuelos
   (¿algún keyframe cae dentro de una trampa?).
2. **Redundancia**: correlación de histograma media entre pares de keyframes
   seleccionados — más bajo es mejor (menos repetición).
3. **Memoria**: tamaño del estado que el método necesita mantener, medido
   empíricamente en función de la duración del vídeo T. La afirmación a
   verificar es O(1)/O(HW) vs O(T).
4. **Throughput**: fps.

**Deliberadamente fuera de alcance en este protocolo**: F1 contra resúmenes
de referencia humanos (TVSum/SumMe) y contra datasets estándar de objetos
abandonados (PETS2006, ABODA) — inaccesibles en este entorno de ejecución
(sin acceso de red a los hosts donde viven). Esta es una limitación real del
estudio, no una omisión oculta; se documenta en la sección de limitaciones
del informe de resultados.

## 4. Datos

- **Suite de señuelos (sintética, controlada)**: generador parametrizado con
  eventos reales (objeto en movimiento, objeto estático persistente, doble
  evento) y señuelos (destello global, ráfaga de ruido denso, ráfaga de
  ruido disperso, transitorio breve de la misma apariencia que un evento
  real). Permite conocer el ground truth exacto.
- **Escalamiento de memoria (sintética)**: vídeos de duración creciente
  (500 / 2 000 / 8 000 / 20 000 frames) con contenido repetitivo simple —
  aquí no importa el contenido, importa cómo crece el estado de cada método.
- **Vídeo real (cualitativo, sin ground truth)**: `vtest.avi` (OpenCV,
  repositorio público, 768×576, 795 frames, peatones, cámara fija). Sirve
  para observar comportamiento en ruido e iluminación reales, no para F1.
- **Dataset real local (ventanas aproximadas)**: 14 carpetas de PNG
  consecutivos, no versionadas; ver §8. No es ground truth preciso.

## 5. Qué contaría como resultado interesante

- Si el modelo dual y la consolidación logran recall de eventos reales
  comparable (diferencia no significativa) con la consolidación usando ~50%
  de la memoria → resultado publicable a favor de la consolidación como
  alternativa eficiente.
- Si el modelo dual rechaza señuelos claramente mejor → resultado honesto en
  contra, también publicable (la nota técnica pasa a ser sobre cuándo *no*
  usar consolidación de un solo modelo).
- Si ninguno de los métodos clásicos supera a la diferencia de frames simple
  en la suite de señuelos → resultado incómodo pero informativo: indicaría
  que la complejidad añadida del GMM no se traduce en robustez práctica
  frente a señuelos simples, y habría que decirlo.

## 6. Amenazas a la validez

- La suite de señuelos la diseñé yo mismo conociendo las debilidades de
  el backend GMM (§3-4 de `ESTADO.md`). Puede estar sesgada a favor de exponer
  precisamente esas debilidades. Se mitiga reportando también condiciones
  donde el backend GMM debería tener ventaja (persistencia genuina), no solo trampas.
- Sin datos reales con ground truth, el recall/precisión son solo sobre
  sintético. El resultado cualitativo en `vtest.avi` no sustituye esto.
  *(Actualización: el dataset local de §8 aporta ventanas de evento reales,
  pero aproximadas; no equivale a ground truth por frame.)*
- La reimplementación de Jacobs–Pless es una aproximación no validada.

## 7. Experimentos sintéticos posteriores (backend `blobtrack`)

Estos experimentos miden decisiones sobre el camino por defecto (`blobtrack`)
y no forman parte de la comparación de la §2. Se corren con
`python -m evaluacion.benchmark` desde la raíz del repo.

### 7.1 Escenarios (`evaluacion/harness.py`, `benchmark.SCENARIOS`)

| escenario | generador | contenido |
|---|---|---|
| `trap` | `generate_trap_video` | la suite de señuelos de la §4 |
| `scale_contrast` | `generate_scale_contrast_video` | un evento real pequeño y corto (cuadrado estático, 40 frames) frente a uno grande y largo (círculo en movimiento continuo, 160 frames), más un destello global como señuelo |
| `scale_contrast_decoys` | `generate_scale_contrast_video(gate_passing_decoys=True)` | lo anterior más dos señuelos que **sí** pasan los gates de `blobtrack` (score > 0): `mota_pequena_transitoria` (32 frames) y `parche_de_luz_grande` (24 frames) |

Motivo de `scale_contrast`: en `trap` los eventos reales tienen tamaño parecido,
así que el sesgo de escala del score de `blobtrack` (área × edad) y de
`select_submodular` (pondera la cobertura por score) es invisible. Los señuelos
de `trap` tienen score exactamente 0, por lo que tampoco exponen el riesgo de
descartar la magnitud del score. El escenario se escribe en coordenadas de
referencia 320×240 y escala a cualquier (W, H).

### 7.2 Experimentos (`--experiment`)

| comando | qué varía | qué mide |
|---|---|---|
| `--experiment weighting` | escenario × ponderación del score {`identity`, `binary`} × presupuesto {3, 5, 8} × config {`baseline`, `aggressive`} | si descartar la magnitud del score (`binary`: score > 0) rescata el evento pequeño sin subir la captura de señuelos (`benchmark.apply_weighting`) |
| `--experiment resolution` | `trap` (320×240 nativo) y `scale_contrast_hd` (1280×720 con señuelos que pasan los gates) × tamaño de resize {180×320, 270×480, 360×640, 540×960} × ponderación × presupuesto × config | efecto del tamaño de resize sobre recall, señuelos y fps |

El tracker corre una vez por (video, config[, tamaño]); ponderación y
presupuesto solo repiten la selección. Las métricas son las de la §3
(recall de eventos reales y señuelos capturados) más fps; el fps incluye
decode y es ruidoso. Resultados: `docs/ESTADO.md` §12.

## 8. Protocolo con datos reales (dataset local)

### 8.1 Datos

`dataset/` en la raíz del repo: 14 carpetas de PNG consecutivos
(`dataset/<Carpeta>/<Carpeta>_NNNNNN.png`; en `Candela_m1_10` los archivos son
`Candela_m1.10_NNNNNN.png`) y `dataset/analisis-frames.md` (descripción de
eventos y cantidad esperada de keyframes). **Está en `.gitignore` (~622 MB):
son datos locales, no versionados**; el protocolo solo se reproduce con esa
misma disposición de carpetas.

`evaluacion/dataset_esperado.json` fija, por carpeta, el rango
`keys_esperados` (del md) y las ventanas de evento (rangos inclusivos de
frames, base 0). 9 carpetas tienen ventanas (31 en total); Foliage, Snellen,
HighwayI, HighwayII y Toscana solo tienen el rango de cantidad. **Las ventanas
se fijaron por inspección visual de los frames antes de correr el algoritmo.**

### 8.2 Método y presupuesto

Se corre el camino por defecto de la librería: `extract_keyframes` con
`blobtrack`, gris + motion gate, 180×320, `submodular`, pesos `identity`.
Presupuesto por carpeta = extremo superior de `keys_esperados` (mínimo 1).
Comparaciones sobre las mismas 9 carpetas: 360×640, nativo sin ampliar, muestreo
uniforme y un selector por evento (experimento fuera del repo; `ESTADO.md` §13.7).

### 8.3 Métricas

(`visualizar_dataset.clasificar`)

- **Acierto**: una ventana con ≥ 1 keyframe.
- **Extra**: keyframe fuera de toda ventana.
- **Redundante**: segundo keyframe dentro de una ventana ya cubierta.
- Carpetas sin ventanas: solo se compara la cantidad con el rango esperado.

Como el md da frames aproximados, las ventanas son anchas y el recall casi no
discrimina; extras y redundancia sí.

### 8.4 Cómo correrlo

```bash
pip install matplotlib   # no es un extra declarado
python -m evaluacion.visualizar_dataset [--only NOMBRE ...] [--budget N] [--out DIR]
python -m evaluacion.graficar_dataset   [--only NOMBRE ...] [--budget N] [--out DIR]
```

Cada carpeta se ensambla en un video temporal sin pérdida (FFV1; MJPG si el
writer no abre) y se borra al terminar. Salidas: tablas por frame en
`evaluacion/resultados_dataset/` y gráficas x-y en
`evaluacion/resultados_grafica/` (una imagen por carpeta más `resumen.png`).
Estas imágenes sí están versionadas y se regeneran con esos comandos.

### 8.5 Amenazas a la validez

- Ventanas anchas y aproximadas; hay discrepancias entre el md y los frames
  (`ESTADO.md` §13.2).
- Casi todas las fuentes miden ≤ 384 px de ancho: no hay video de alta
  resolución real.
- Foliage y PeopleAndFoliage tienen cámara móvil, lo que viola el supuesto de
  cámara fija.
- El default (180, 320) se ratificó con estos mismos datos, así que la
  comparación de configuraciones no es un holdout.
- Diferencias de 3-4 ventanas sobre 31: no concluyente.
