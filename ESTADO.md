# MOG3 — Estado del trabajo, limitaciones y agenda

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

**Riesgo alto — novedad no verificada.** No se ha hecho revisión de
literatura. Es posible que la lectura de la dinámica de nacimiento/muerte de
componentes ya exista. Búsquedas pendientes: *component birth-death dynamics
background subtraction*, *mixture model novelty persistence*, *Bayesian
surprise video summarization*, *GMM component lifetime video*. **Debe hacerse
antes de invertir más esfuerzo de ingeniería.** Si el resultado ya existe, la
agenda cambia por completo.

**Riesgo alto — posicionamiento.** El nombre "MOG3" promete un modelo de
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

### Paso 0 — Revisión de literatura (bloqueante, sin código)
Establecer si la contribución es nueva antes de seguir construyendo.
**Criterio de decisión:** si existe trabajo previo equivalente, detener el
desarrollo y replantear el ángulo.

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
