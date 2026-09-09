# Resultados — Estudio comparativo (Opción C)

**Fecha:** 8 de septiembre de 2026
**Protocolo:** `PROTOCOLO.md` · **Código:** `evaluacion/baselines.py`, `evaluacion/harness.py`
**Pregunta central:** ¿la consolidación de componentes (un modelo) iguala al
modelo dual (dos modelos) en calidad de detección, con menos memoria?

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
sigue siendo una suite pequeña (6 eventos, 8 semillas) y un solo generador.
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
   en este entorno (§`PROTOCOLO.md` §4). Es el hueco más grave.
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
