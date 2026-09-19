# KeySeer — Estado del trabajo, limitaciones y agenda

**Versión:** 0.1.0 · **Fecha:** 8 de septiembre de 2026
**Propósito:** documento interno de estado. Registra qué está demostrado, qué
está refutado, qué está sin verificar, y qué sigue. Escrito para ser usado
como base de la sección de limitaciones del paper.

---

## 1. Resumen ejecutivo

La contribución teórica central está **validada en condiciones controladas**.
El sistema **no está listo para publicación**: no ha visto un solo vídeo real,
y la métrica compuesta no logra separar eventos genuinos de señuelos en el
banco sintético de prueba.

| | estado |
|---|---|
| Proposición 1 (la latencia es necesaria) | demostrada y verificada |
| Señal Ψ separa transitorio de persistente | verificada, 11.9x |
| Hipótesis de sorpresa bayesiana (KL) | **refutada**, 1.01x |
| Métrica compuesta rechaza señuelos | **falla**, separación negativa |
| Validación en vídeo real | **inexistente** |
| Comparación contra baselines | **inexistente** |

---

## 2. Lo que está demostrado

### 2.1 Proposición 1

> Sean dos eventos idénticos hasta el frame *t*, uno transitorio y otro
> persistente. Todo funcional causal e instantáneo del modelo en *t* les
> asigna el mismo valor. La discriminación transitorio/persistente requiere
> latencia.

La prueba es inmediata: si las observaciones $I_1..I_t$ coinciden y el modelo
es una función determinista de ellas, entonces $\theta_t$ coincide, y también
cualquier funcional de $\theta_t$.

**Verificación empírica** (magnitud, extensión y posición idénticas; solo
cambia la persistencia; medido en el frame de inicio):

| señal | A: transitorio | B: persistente | razón |
|---|---|---|---|
| nacimientos | 0.2500 | 0.2500 | **1.00x** |
| KL (sorpresa bayesiana) | 0.15768 | 0.15862 | **1.01x** |
| consolidación Ψ | 0.01005 | 0.11948 | **11.89x** |

Nacimientos y KL son ciegos, como predice la proposición. Ψ separa un orden
de magnitud. Codificado en `tests/test_core.py::test_instantaneous_signals_are_blind_at_onset`.

### 2.2 Dinámica de poda: regla de diseño para L

Para una componente **no emparejada** (transitoria), el peso evoluciona como

$$w_{t+1} = w_t(1-\alpha) - \alpha c_T$$

cuya solución es $w_t = (w_0 + c_T)(1-\alpha)^t - c_T$. El tiempo de extinción es

$$t_{\text{muerte}} = \frac{\ln\big(c_T/(w_0+c_T)\big)}{\ln(1-\alpha)}$$

Para una componente **emparejada** (persistente), $w_t \to 1-c_T$ con constante
de tiempo $1/\alpha$.

**Consecuencia práctica no implementada:** con $\alpha=0.02$, $c_T=0.01$,
$w_0=\alpha$, resulta $t_{\text{muerte}} \approx 54$ frames. Los experimentos
usaron $L=20$, es decir **la latencia era menor que el tiempo de extinción**.
Las componentes transitorias todavía conservaban ~50% de su peso al momento de
medirlas. Esto explica parte del fallo de rechazo de señuelos y es corregible
de forma analítica (§5.1).

---

## 3. Lo que está refutado

### 3.1 La sorpresa bayesiana no discrimina

La hipótesis inicial era que $D_t = \mathrm{KL}(\theta_t \| \theta_{t-1})$
(Itti–Baldi) distinguiría ruido de contenido genuino, bajo el argumento de que
"el ruido sorprende pero no informa". **Es falsa en este modelo**: 1.01x.

La razón es la Proposición 1 aplicada a la propia KL — es un funcional
instantáneo de $\theta_t$ y $\theta_{t-1}$, ambos causales. No puede ver el
futuro, y la distinción transitorio/persistente vive en el futuro.

**Implicación para el paper:** la KL debe presentarse como señal descartada
con evidencia, no omitirse. Un revisor que conozca Itti–Baldi preguntará por
ella; tener el número negativo medido es más fuerte que no mencionarla.

### 3.2 El término de coherencia espacial es contraproducente para destellos

Ablación sobre el vídeo de evaluación (3 eventos reales, 2 señuelos):

| variante | quieto | EVT1 | DESTELLO\* | RUIDO\* | EVT2 | EVT3 |
|---|---|---|---|---|---|---|
| A: solo masa | 0.113 | 0.379 | 1.000 | 1.000 | 0.799 | 0.897 |
| B: masa × ratio | 0.113 | 0.596 | **0.613** | 1.000 | 0.799 | 0.898 |
| C: masa × ratio × coherencia | 0.113 | 0.772 | **0.998** | 1.000 | 0.799 | 0.898 |
| D: masa × coherencia | 0.113 | 0.762 | 1.000 | 1.000 | 0.799 | 0.898 |

\* = señuelo, debería puntuar bajo.

El paso de B a C **empeora el destello de 0.613 a 0.998**. La causa es
estructural: un destello global es un mapa uniformemente elevado, y mi medida
$\mathrm{coh} = E[\mathrm{smooth}(m)^2]/E[m^2]$ asigna coherencia 1.0 a
cualquier mapa uniforme. La medida no distingue "estructurado" de "uniforme";
solo penaliza alta frecuencia espacial.

La coherencia sí ayuda a EVT1 (0.596 → 0.772), por lo que el margen agregado
mejora, pero por la razón equivocada. **No debe presentarse como un
componente que funciona.**

---

## 4. Limitaciones verificadas

### 4.1 La métrica compuesta no separa señuelos (crítico)

Margen de separación = mín(eventos reales) − máx(señuelos):

| variante | real mín | señuelo máx | **separación** |
|---|---|---|---|
| A: solo masa | 0.379 | 1.000 | **−0.621** |
| B: masa × ratio | 0.596 | 1.000 | **−0.404** |
| C: masa × ratio × coherencia | 0.772 | 1.000 | **−0.228** |
| D: masa × coherencia | 0.762 | 1.000 | **−0.238** |

**Las cuatro variantes tienen separación negativa.** En toda configuración
probada, al menos un señuelo puntúa por encima del evento real más débil. La
selección submodular con presupuesto 6 escoge el frame 200 — el centro de la
ráfaga de ruido.

La causa dominante es la ráfaga de ruido (15 frames), que satura en 1.000 en
todas las variantes. El ruido sostenido durante más tiempo que la constante de
adaptación *sí* consolida, y ninguna de las señales actuales lo distingue de
contenido real.

Esta es la limitación bloqueante. Ψ resuelve el problema para el que fue
diseñada (transitorio vs. persistente) pero el ruido denso sostenido **no es
transitorio**, y por tanto cae fuera de su alcance por construcción.

### 4.2 Confusión área/persistencia (corregida, documentada)

La masa de consolidación cruda mezcla cuánta área cambió con qué tan
persistente fue el cambio:

| región | masa | nacimientos | ratio por nacimiento |
|---|---|---|---|
| DESTELLO (global) | 0.02626 | 1.0000 | **0.026** |
| EVT2 (objeto 8% del área) | 0.02868 | 0.0828 | **0.346** |

El destello obtenía masa comparable a un objeto persistente **solo por cubrir
12x más área**. Corregido con `consolidation_ratio`. Se documenta porque es un
modo de fallo no obvio que otros reimplementadores encontrarán.

### 4.3 Rendimiento por debajo de tiempo real

NumPy puro, sin optimizar:

| resolución | fps | ms/frame | memoria del modelo |
|---|---|---|---|
| 120×160 | 47.2 | 21.2 | 2.3 MB |
| 180×320 | 16.7 | 59.7 | 6.9 MB |
| 240×426 | 10.4 | 96.3 | 12.3 MB |
| 360×640 | 4.4 | 228.2 | 27.6 MB |

**La afirmación de "viable en tiempo real / embebidos" no está sostenida.**
Hay que separar dos cosas que no deben confundirse en el paper:

- *Complejidad algorítmica*: favorable y demostrable — una pasada, memoria
  $O(HWM)$ independiente de la duración $T$ del vídeo. Esto sí es publicable.
- *Rendimiento de la implementación*: pobre. No es un argumento hasta que
  exista una implementación optimizada.

### 4.4 Validación exclusivamente sintética

**Ningún vídeo real ha sido procesado.** Todos los resultados provienen de
vídeos generados con formas geométricas sobre fondo plano. Esto invalida
cualquier afirmación sobre desempeño práctico. Es la brecha más grande entre
el estado actual y un manuscrito enviable.

---

## 5. Limitaciones estructurales (supuestos del método)

| supuesto | consecuencia si se viola | ¿mitigable? |
|---|---|---|
| Cámara fija | el modelado por píxel colapsa; todo es foreground | estabilización previa, o cambio de método |
| Latencia aceptable | inaplicable a alertas en vivo de latencia cero | no — la Proposición 1 lo prohíbe |
| Calentamiento ~1/α frames | los primeros frames no son fiables | inicialización por lote |
| Iluminación estable | cambios globales generan nacimientos masivos | normalización cromática |
| Contenido informativo = contenido nuevo | falla si lo importante es lo que *no* cambia | fuera de alcance |

---

## 6. Riesgos para publicación

**RESUELTO — novedad refutada en su formulación original.** La revisión se
hizo (ver `LITERATURA.md`). El mecanismo central **no es novedoso**: la
persistencia de componentes como criterio está en Stauffer-Grimson (1999), y
la lectura de edad/actividad de modos en un modelo multimodal para decidir
estacionariedad está patentada (US8305440, Canon). La comunidad de detección
de objetos abandonados resuelve el mismo problema con modelos de fondo duales.

Lo que sobrevive es mucho más estrecho: la regla analítica de latencia, el
argumento de modelo-único frente a modelo-dual, y el hueco de aplicación en
resumen de vídeo. **La agenda de la §7 debe leerse a la luz de
`LITERATURA.md` §4, que la modifica.**

**Riesgo alto — posicionamiento.** El nombre original (previo a KeySeer) prometía un modelo de
fondo nuevo. La contribución real está en la *lectura* del modelo; las
ecuaciones de actualización son de Zivkovic con una corrección menor de
varianza. Recomendación: renombrar a algo descriptivo del mecanismo
(consolidación) y presentar el trabajo como método de detección, no como
sucesor de MOG2.

**Riesgo medio — eje de comparación.** Contra métodos entrenados en
TVSum/SumMe se perderá en F1. Los ejes defendibles son: cero datos de
entrenamiento, operación en streaming, memoria constante en $T$,
interpretabilidad. Requieren protocolo experimental propio, no solo la tabla
de F1 estándar.

---

## 7. Agenda de trabajo

Ordenada por relación valor/riesgo. Los pasos 0 y 1 son bloqueantes.

### Paso 0 — Revisión de literatura — **COMPLETADO, resultado negativo**
Ver `LITERATURA.md`. Existe trabajo previo equivalente. Se activó el criterio
de detención: **replantear el ángulo antes de continuar con la ingeniería.**
Nuevo baseline obligatorio en el Paso 5: modelo de fondo dual (fast/slow),
que es el competidor directo real y no estaba contemplado.

### Paso 1 — Corregir la latencia con la regla analítica
Fijar $L \geq t_{\text{muerte}} = \ln(c_T/(\alpha+c_T))/\ln(1-\alpha)$ en vez
de un valor arbitrario. Con la configuración actual esto sube $L$ de 20 a ~54.
Barato, principiado, y probablemente mejora el rechazo del destello de forma
sustancial. **Repetir la ablación de §3.2 y §4.1 con la L corregida antes de
cualquier otra cosa** — parte del fallo puede desaparecer aquí.

### Paso 2 — Rechazo de ruido por trayectoria de varianza
Hipótesis: las componentes nacidas de ruido **no estrechan su varianza**
(siguen ajustando datos incoherentes, $\sigma^2$ permanece alta), mientras que
las componentes de objeto convergen a la varianza del sensor. La señal
candidata es $\sigma^2_{t+L}/\sigma^2_{\text{init}}$ para la componente nacida
en $t$.

Esto es *ortogonal* al peso y por tanto no redundante con Ψ. Es la vía más
prometedora para §4.1.

**Criterio de éxito medible:** separación (§4.1) positiva en el vídeo de
evaluación existente. Debe probarse como hipótesis falsable, igual que se hizo
con la KL — y descartarse con el número si falla.

### Paso 3 — Reemplazar o eliminar la coherencia espacial
La medida actual está refutada (§3.2). Alternativas: energía de gradiente
normalizada, estructura de componentes conexas, o entropía espacial del mapa.
Debe distinguir *uniforme* de *estructurado*, cosa que la actual no hace.
Si ninguna alternativa supera la ablación, eliminar el término.

### Paso 4 — Validación en vídeo real
Primero cualitativa (5–10 vídeos de cámara fija: vigilancia, timelapse) para
detectar modos de fallo que lo sintético no expone. Después cuantitativa sobre
benchmark. Es probable que esto genere una nueva lista de limitaciones; ese es
su propósito.

### Paso 5 — Protocolo experimental
Baselines: muestreo uniforme, k-means sobre histogramas, umbral de diferencia
de frames, MOG2+conteo de foreground (el método de la primera iteración), y
al menos un método reciente. Ablación de cada señal. Pruebas de significancia.
Sin esto no hay manuscrito.

### Paso 6 — Optimización (solo si el rendimiento es un eje de la tesis)
Numba o Cython sobre el bucle de `update`. Postergable: no afecta la validez
científica, solo la fuerza de un argumento secundario.

---

## 8. Criterios de decisión

Puntos donde conviene detenerse y reevaluar en vez de seguir por inercia:

1. **Tras el Paso 0.** Si el mecanismo ya está publicado, replantear.
2. **Tras el Paso 2.** Si la trayectoria de varianza no da separación
   positiva, la métrica no distingue ruido de contenido, y eso limita el
   método a entornos de bajo ruido. Es publicable, pero hay que declararlo
   como alcance, no esconderlo.
3. **Tras el Paso 4.** Si el desempeño en vídeo real es malo, la contribución
   podría ser teórica (la Proposición 1 y el principio de consolidación) más
   que práctica. Es una publicación distinta, en un lugar distinto, y más
   corta.

---

## 9. Nota sobre método

Tres de los hallazgos de este documento (§3.1, §3.2, §4.3) son **correcciones
de afirmaciones que inicialmente se dieron por buenas** sin medir. En los tres
casos, la medición las contradijo:

- se asumió que la KL bayesiana discriminaría → 1.01x;
- se asumió que la coherencia ayudaba → empeora el destello;
- se asumió throughput de tiempo real → 16.7 fps a 180×320.

La primera versión de la métrica también parecía funcionar hasta que se
introdujeron señuelos deliberados en el vídeo de evaluación. Recomendación
operativa: **todo test sintético nuevo debe incluir señuelos diseñados para
engañar a la métrica propuesta**, no solo ejemplos que la favorezcan.


---

## 10. Modulo A vs Modulo B — pausa de publicacion, foco en herramienta personal (turno 8)

**Decision del usuario:** pausar la publicacion, separar explicitamente
**Modulo A** (deteccion de persistencia) de **Modulo B** (algoritmo de
keyframes), y optimizar por la mejor herramienta con la menor complejidad
para uso personal, dejando de lado la novedad.

### 10.1 Modulo A: dos backends comparados con evidencia, no por argumento

| | GMM NumPy (`core.py`) | cv2 MOG2 + edad de blob (`blobtrack.py`) |
|---|---|---|
| recall eventos reales (submodular, 8 semillas) | 1.00 ± 0.00 | 1.00 ± 0.00 |
| señuelos capturados | 1.00 ± 0.00 | **0.00 ± 0.00** |
| velocidad | ~17 fps (180×320) | **~345-455 fps** |
| memoria de estado | 360 KB fijo (O(H·W)) | **~200 B-2 KB (O(n objetos))** |
| lineas de codigo del nucleo | ~300 | **~150** |

**Decision: `blobtrack` es el backend por defecto.** Gana o empata en todos
los ejes medidos. `core.py` (GMM) se conserva como backend alternativo
(`backend="gmm"`) para quien necesite las señales fina (Ψ, KL, sorpresa)
que el blob tracker no expone -- no se elimina, no se recomienda por
defecto.

### 10.2 Un fallo critico encontrado y corregido: el mismo problema que
motivo todo el proyecto, resurgiendo en el reemplazo "simple"

Al probar `blobtrack` con un objeto estatico AISLADO (sin eventos previos
en el video), el resultado fue **0 keyframes, score maximo 0.0** -- fallo
total. Se rastreo el mecanismo: cv2 MOG2 usa por defecto una tasa de
aprendizaje AUTOMATICA que es mucho mas rapida al principio del video
(~1/frames_vistos) que su tasa nominal (1/history). Un objeto que aparece
temprano en el video (frame 30) se absorbe al fondo en ~9 frames; el MISMO
objeto apareciendo tarde (frame 280, con 250+ frames de "calentamiento"
previo) sobrevive >50 frames. Verificado directamente:

| calentamiento previo | edad maxima alcanzada por el track |
|---|---|
| 0 frames | 7 (muere antes del umbral minimo) |
| 250 frames | 55 |

Esto es, literalmente, el mismo problema que origino la busqueda de
Ψ/modelo dual en primer lugar (MOG2 solo no detecta persistencia) --
resurgiendo dentro del backend que se proponia como reemplazo simple,
oculto porque la suite de señuelos usada para validarlo (`PROTOCOLO.md`)
tenia sus eventos reales bien entrados en el video, con calentamiento de
sobra.

**Correccion:** forzar una tasa de aprendizaje EXPLICITA
(`learningRate=1/history` en `back_sub.apply()`) en vez de la automatica.
Verificado: con la tasa fija, calentamiento=0 y calentamiento=250 alcanzan
la MISMA edad maxima (55). Se revalido toda la suite de 8 semillas tras la
correccion: recall y rechazo de señuelos identicos a los reportados en la
tabla de 10.1. Implementado en `blobtrack.py`.

**Leccion:** un backend "mas simple" hereda las trampas de la libreria que
envuelve si no se audita con el mismo rigor que el codigo propio. La
prueba que lo destapo fue deliberadamente distinta a la suite de
validacion existente (objeto aislado, sin eventos previos) -- las
suites de prueba existentes no lo habrian detectado nunca.

### 10.3 Falsa alarma investigada y descartada: relleno de presupuesto

Se sospechaba que la seleccion submodular rellenaba el presupuesto con
keyframes redundantes cuando el video tiene menos eventos genuinos que
`budget` (observado: 3 eventos reales, budget=6, 2-3 keyframes por
evento). Investigacion:

1. La metrica de redundancia usada (`pairwise_redundancy`) resulto estar
   rota (bug propio, reincidente: el mismo problema de histograma
   dominado por fondo ya corregido en el turno 1, reintroducido aqui).
   Corregida con mascara de primer plano independiente por frame.
2. Con la metrica corregida, se confirmo visualmente que los keyframes
   señalados como "redundantes" en realidad mostraban contenido
   significativamente distinto (mismo objeto, posiciones distintas) -- la
   alarma original era un artefacto de medicion, no un defecto real.
3. Se probo directamente si el algoritmo, SIN ningun criterio de parada
   adicional, rellena el presupuesto para un evento VERDADERAMENTE
   estatico: no lo hace. Con budget=1,3,6,10 siempre devuelve exactamente
   1 keyframe, porque la cobertura del objetivo submodular satura de
   forma natural. Codificado como test de regresion
   (`test_static_object_no_padding_regardless_of_budget`).
4. Se implemento un criterio de parada adicional (`min_gain_ratio`) de
   todas formas, y se encontro que activarlo INTRODUCE un problema nuevo:
   descarta eventos reales genuinamente distintos pero de menor
   intensidad (verificado, se pierde un evento real completo con
   ratio=0.05). Se deja como parametro opcional, desactivado por defecto,
   con esta advertencia documentada en el codigo.

**Conclusion: no hacia falta arreglar nada.** El algoritmo base ya se
comportaba correctamente; el problema estaba en como se media.

### 10.4 Entregable: API unica para uso personal

`keyseer/keyframes.py` -- `extract_keyframes(video_path, budget=8,
out_dir=...)`: una sola llamada, backend `blobtrack` por defecto, guarda
los .jpg elegidos en disco. Probado extremo a extremo en `vtest.avi`
(video real): 8 keyframes, 2.36s de procesamiento, archivos verificados en
disco.

### 10.5 Lo que sigue pendiente

- El punto ciego del filtro de area maxima (`max_area_frac`, rechaza
  objetos reales que cubran >50% del frame) sigue sin resolver mas alla de
  dejarlo configurable -- ver conversacion, turno 7.
- El rechazo de ruido denso del backend GMM (Paso 2, seccion 7 de este
  documento) nunca se completo; con el cambio de backend por defecto a
  `blobtrack` (que SI rechaza el ruido denso de la suite, seccion 10.1)
  esto pierde prioridad para el uso personal, pero sigue abierto si se
  retoma `backend="gmm"`.
- No se ha probado el pipeline completo con video real filmado a mano
  (camara no fija) -- todos los backends asumen camara estatica.

## 11. Preprocesamiento agresivo para uso real (backend blobtrack)

Objetivo: preparar `blobtrack` (backend por defecto) para video real de
camara fija, donde la mayor parte del tiempo no pasa nada. Tres cambios,
todos activados por defecto en `analyze_video_blobtrack` /
`extract_keyframes_blobtrack` / `keyseer.keyframes.extract_keyframes`:

1. **Escala de grises antes de MOG2.** `_detect`/`update` nunca leen
   color, solo la mascara binaria resultante -- confirmado en tiempo de
   ejecucion, no solo por lectura de codigo. Sin cambio medible en
   recall/decoy sobre la suite de señuelos.
2. **Decode real de frames descartados.** `_iter_video` ahora usa
   `cap.grab()` (sin decodificar) para los frames que el stride/gate va a
   descartar, y solo `cap.retrieve()` para los que se procesan -- antes se
   decodificaba TODO el video sin importar el stride.
3. **Stride adaptativo por movimiento (`MotionGatedStride`).** Un
   `cv2.absdiff(...).mean()` barato sobre el frame gris ya reducido decide
   si hace falta correr el detector completo. Durante tramos quietos
   sostenidos el espaciado crece geometricamente (`1,2,4,8,...`) hasta un
   tope (`max_quiet_stride`).

### 11.1 Calibracion: un solo heartbeat no alcanza

Primer intento: un heartbeat unico (recorre todo el video cada
`heartbeat_interval` frames reales como maximo, sin importar el estado)
para acotar el riesgo de que el modelo de fondo quede desactualizado.
Resultado contra la suite de 8 semillas (`evaluacion/harness.py`):
recall_real cayo a **0.5** (decoy_hits se mantuvo en 0).

Causa raiz: el rectangulo persistente-pero-ESTATICO de la suite (la señal
central que este proyecto existe para detectar, ver seccion 1 y el
hallazgo de Ψ) solo dispara `motion` en su frame de aparicion. Despues de
eso la escena esta quieta, y el `age` de un track SOLO avanza en las
pasadas donde `tracker.update()` corre -- no por tiempo real transcurrido.
Con un heartbeat calibrado para "escena vacia, es seguro saltar mucho"
(8-16 frames), un objeto asi tarda hasta `min_age_to_count *
heartbeat_interval` frames reales en alcanzar el gate duro de
`min_age_to_count=8` -- mas que la ventana tipica de un evento corto de la
suite. Verificado con barridos de `motion_threshold` (0.05 a 2.0) y de una
metrica de deteccion alternativa (fraccion de pixeles cambiados en vez de
media global): el sintoma persistia identico sin importar la sensibilidad
del gate, lo que descarto "el umbral esta mal calibrado" y confirmo "el
mecanismo esta incompleto."

**Correccion:** `MotionGatedStride.step()` recibe ahora un tercer
parametro, `tracks_active` (que `analyze_video_blobtrack` alimenta con
`len(tracker.tracks) > 0`), y usa un `active_heartbeat_interval` mucho mas
chico que `heartbeat_interval` cuando hay tracks vivos. Esto desacopla dos
preguntas distintas que el diseño original conflaba en un solo numero:
"cada cuanto es seguro revisar si volvio a pasar algo" (escena vacia,
heartbeat largo) vs. "cada cuanto hace falta re-observar algo que ya esta
ahi para que su edad siga contando" (heartbeat corto).

**Defaults calibrados y verificados en 24 semillas** (`base_stride=1,
max_quiet_stride=16, growth_factor=2, motion_threshold=2.0,
active_heartbeat_interval=2`): recall_real=1.0, decoy_hits=0.0 -- igual
que sin gate -- procesando en promedio ~25% de los frames del video
(131-143 de 541), ~2.7-2.9x mas rapido en reloj de pared. Robusto: el
resultado se mantuvo identico barriendo `motion_threshold` de 0.5 a 3.0,
o sea que la ganancia no depende de un ajuste fragil de un solo numero.

### 11.2 Resultado (`evaluacion/benchmark.py`, 16 semillas)

Nuevo driver ejecutable (antes no existia ninguno en el repo -- toda la
evaluacion se corria a mano). `python -m evaluacion.benchmark --seeds 16`:

| config | recall_real | decoy_hits | elapsed (s) | state_bytes |
|---|---|---|---|---|
| baseline (color, sin gate) | 1.00 | 0.00 | 1.30 | 400 |
| grayscale_only | 1.00 | 0.00 | 1.23 | 250 |
| aggressive (gris + gate) | 1.00 | 0.00 | 0.45 | 312.5 |

`aggressive` iguala recall/decoy de `baseline` con ~2.9x menos tiempo de
procesamiento y sin peor memoria de estado pico.

### 11.3 Cambio de comportamiento no silencioso

`resize_to` por defecto de `keyframes.extract_keyframes` bajo de
`(240,320)` a `(180,320)` (unificado con `blobtrack.py`/`pipeline.py`,
revalidado contra la suite). `grayscale=True` y `motion_gate=True` son
ahora el default en todos los puntos de entrada respaldados por
`blobtrack`; `extract_keyframes_blobtrack` devuelve indices de frame
REALES en `"keyframes"` (antes devolvia posiciones densas sin aplicar
`stride`, inconsistente con `keyframes.py`). `analyze_video_blobtrack` /
`extract_keyframes_blobtrack` ganan una clave nueva `"frame_indices"` en
el dict de resultado.

**Leccion:** un gate de movimiento "barato y generico" que no sabe nada
del estado del tracker es, por diseño, incompatible con detectar
persistencia estatica -- que es literalmente la razon de ser de este
proyecto. La correccion no fue "calibrar mejor un numero", fue notar que
el gate necesitaba una segunda señal (¿hay algo que ya estoy seguimiento?)
ademas del movimiento crudo.

## 12. Tamano de resize de blobtrack y señuelos que pasan los gates (turno 9)

Los parametros del tracker estan en PIXELES (`min_area=60`, kernel 5,
`max_match_dist=40`): en video real de alta resolucion, reducir de mas borra
objetos chicos. Se midio antes de subir el default (`--experiment resolution`).

**Señuelos con score > 0.** Antes: pesos binarios (`score > 0`) subian
recall_real 0.5 -> 1.0 en `scale_contrast` (b=3,5; decoy 0.0), pero todos los
señuelos del trap suite tienen score exactamente 0, asi que no exponian el
riesgo. `generate_scale_contrast_video(gate_passing_decoys=True)` agrega
`mota_pequena_transitoria` (1.6e-3 del frame, 32 frames) y `parche_de_luz_grande`
(37.5%, 24 frames); con 12 frames ninguno pasaba `aggressive` (hacen falta >=28
y >=20). 1280x720, 8 semillas, identico en los 4 tamanos (recall/decoy):

| config | pesos | b=3 | b=5 | b=8 |
|---|---|---|---|---|
| baseline | identity | 0.50/1.0 | 0.50/1.0 | 1.00/1.0 |
| baseline | binary | 1.00/0.0 | 1.00/1.0 | 1.00/1.0 |
| aggressive | identity | 0.50/0.0 | 0.50/1.0 | 1.00/2.0 |
| aggressive | binary | 1.00/0.0 | 1.00/0.0 | 1.00/2.0 |

Ambos riesgos son reales: con identity el parche domina (score pico 9.0 vs 1.7
del circulo y 0.19 del objeto chico); con binary entra la mota (0.056) en b=5
baseline. binary nunca captura mas señuelos que identity y recupera el evento
chico: los señuelos NO revierten la conclusion, pero no es gratis. Adoptarlo en
`selection.py` sigue siendo decision aparte.

**Barrido de tamano** (trap 320x240 nativo + scale_contrast 1280x720):

| tamano | trap recall/decoy (b=5,8) | fps trap base/agg | fps HD base/agg |
|---|---|---|---|
| 180x320 | 1.00/0.00 | 311/218 | 151/77 |
| 270x480 | 1.00/0.00 | 205/150 | 116/63 |
| 360x640 | 1.00/0.00 | 138/104 | 91/54 |
| 540x960 (info) | 1.00/0.00 | 61/47 | 49/34 |

(trap b=3 da 0.667 en todos, tambien a 180x320: limite de presupuesto.) Regla:
el mayor de {270x480, 360x640} que iguale la referencia en trap con baseline y
aggressive -> **`DEFAULT_RESIZE_TO = (360, 640)`**. Costo del benchmark por
defecto (8 semillas): elapsed 1.70 -> 3.82 s baseline, 0.59 -> 1.31 s
aggressive (~2.2x); recall 1.00 / decoy 0.00 sin cambio.

**Por que sube.** La suite no lo muestra (recall/decoy identico en todos los
tamanos). Un cuadrado estatico en 1280x720 (40 frames): lado 20 px (1.6% del
ancho) desaparece a 320 y 480 de ancho y se detecta desde 640 (baseline y
aggressive); 32 px falla en aggressive a 320 y anda desde 480; contraste tenue
(+25 gris) vs normal: sin efecto. El gate de movimiento nunca dispara en el
onset (diff medio 0.01-0.17 vs umbral 2.0): aggressive solo detecta por
heartbeat; la resolucion ayuda, el cuello del gate es la DURACION.

**Por backend.** El GMM es NumPy por pixel (costo ~ H*W): mantiene (180,320).
`extract_keyframes(resize_to=None)` resuelve por backend (`DEFAULT_RESIZE_TO` /
`GMM_RESIZE_TO`); `None` ya no significa "sin reducir". Reemplaza 11.3.

Abierto: eventos estaticos <~28 frames invisibles para `aggressive`; con fuente
nativa 320x240 aggressive+binary+b=3 pasa de 1.0 a 0.5 (cluster del objeto 11
frames vs 13-14 del circulo; knife-edge, no perdida de deteccion); sin probar
objetos <1.6% del ancho ni fuentes distintas de 720p; fps es ruidoso.

**Reversion del default a (180, 320).** Evaluacion sobre `dataset/` real (14
carpetas; 9 con eventos = 31 ventanas fijadas mirando los frames antes de
correr; presupuesto = extremo superior de "keys esperados" en
`dataset/analisis-frames.md`). Aciertos / extras: 180x320 23/31, 8; 360x640
20/31, 13; nativo sin ampliar 19/31, 11; muestreo uniforme 23/31, 13. Casi todas
las fuentes miden <=384 px de ancho, asi que 360x640 es una ampliacion. La
diferencia es de 3-4 ventanas y las ventanas son anchas: no es concluyente,
pero ningun dato respalda el aumento. `DEFAULT_RESIZE_TO` vuelve a (180, 320);
la constante unica y la resolucion por backend de `resize_to=None` se mantienen
(`None` sigue sin significar "sin reducir"). La tabla y la regla de arriba se
conservan como registro del barrido: su decision de default queda reemplazada
por esta. Sigue sin probarse en video real de alta resolucion, que es donde
aplica la justificacion original (objetos <2% del ancho).
