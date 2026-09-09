# Protocolo — Estudio comparativo de métodos clásicos de keyframes/resumen
## (Opción C: la contribución es la evidencia, no el mecanismo)

**Fecha:** 8 de septiembre de 2026

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
- La reimplementación de Jacobs–Pless es una aproximación no validada.
