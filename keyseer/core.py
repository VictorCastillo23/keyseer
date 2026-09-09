"""
keyseer.core
============

Modelo de mezcla de gaussianas por pixel, reimplementado desde las ecuaciones
(no envuelve cv2.createBackgroundSubtractorMOG2), con lecturas
information-theoretic expuestas por frame.

Base teorica
------------
Cada pixel x se modela como una mezcla de M gaussianas isotropicas:

    p(I(x) | theta(x)) = sum_m  w_m  N(I(x); mu_m, sigma_m^2 I_C)

La estimacion recursiva MAP con prior de Dirichlet de coeficiente negativo
(Zivkovic & van der Heijden, 2006) da la actualizacion de pesos:

    w_m <- w_m + alpha (o_m - w_m) - alpha c_T

donde o_m = 1 para la componente emparejada y 0 para el resto, y el termino
-alpha c_T empuja a cero las componentes espurias (seleccion automatica del
numero de componentes). Las componentes con peso negativo se podan.

Para la componente emparejada:

    rho    = alpha / w_m
    mu_m   <- mu_m   + rho (I(x) - mu_m)
    sig2_m <- sig2_m + rho ( ||I(x)-mu_m||^2 / C  - sig2_m )

NOTA DE CORRECCION: OpenCV actualiza la varianza con ||delta||^2 sin dividir
entre C, lo que sesga sigma^2 por un factor C en video a color. Aqui se usa
el estimador MLE isotropico correcto ||delta||^2 / C. Esto se documenta como
una diferencia deliberada respecto a la implementacion de referencia.

Lecturas expuestas (la contribucion de KeySeer)
------------------------------------------------
1. Sorpresa predictiva (Shannon):
       S_t(x) = -log p(I_t(x) | theta_{t-1}(x))
   Instantanea. Se evalua ANTES de actualizar, asi que es genuinamente
   predictiva y no circular.

2. Sorpresa bayesiana (Itti & Baldi):
       D_t(x) = KL( theta_t(x) || theta_{t-1}(x) )
   Integrativa: mide cuanto CAMBIO el modelo, no cuanto sorprendio.
   Calculada con la cota superior por emparejamiento de componentes.

3. Novedad estructural:
       B_t(x) = 1 si nacio una componente nueva en x en t
   Discreta: el soporte del modelo cambio.

Referencias
-----------
Stauffer & Grimson (1999); Zivkovic (2004); Zivkovic & van der Heijden (2006);
Itti & Baldi (2009), "Bayesian surprise attracts human attention".
"""

from dataclasses import dataclass, field
import numpy as np

__all__ = ["KeySeerConfig", "KeySeer", "FrameReadout"]

_EPS = 1e-12


@dataclass
class KeySeerConfig:
    """Hiperparametros del modelo."""

    n_components: int = 5          # M: componentes por pixel
    alpha: float = 0.01            # tasa de aprendizaje (1/history)
    c_T: float = 0.01              # coeficiente del prior de Dirichlet negativo
    var_init: float = 225.0        # sigma^2 inicial para componentes nuevas (15^2)
    var_min: float = 4.0           # cota inferior de sigma^2 (evita colapso)
    var_max: float = 5000.0        # cota superior de sigma^2
    mahalanobis_thresh: float = 3.0  # umbral de emparejamiento en sigmas
    bg_ratio: float = 0.9          # fraccion de peso acumulado que define fondo
    latency: int = 12              # L: frames de espera para medir consolidacion
    dtype: type = np.float32

    @property
    def alpha_init(self) -> float:
        """Peso asignado a una componente recien nacida."""
        return self.alpha


@dataclass
class FrameReadout:
    """Salidas por frame del modelo."""

    index: int
    surprisal: np.ndarray      # (H,W) sorpresa predictiva en nats
    kl: np.ndarray             # (H,W) sorpresa bayesiana (cota superior)
    births: np.ndarray         # (H,W) bool, nacimiento de componente
    fg_mask: np.ndarray        # (H,W) bool, foreground clasico (para comparar)
    n_active: np.ndarray       # (H,W) int, componentes activas
    warmup: bool = False       # True si el modelo aun no esta estabilizado

    # --- salida retardada (latencia L) -------------------------------
    # consolidation es el mapa Psi para el frame `consolidation_index`,
    # disponible L frames despues de aquel. None durante los primeros L.
    consolidation: np.ndarray = None
    consolidation_index: int = -1


class KeySeer:
    """
    Modelo KeySeer online. Una pasada, memoria O(H*W*M) independiente de la
    longitud del video.

    Uso
    ---
        model = KeySeer(height, width, channels)
        for frame in video:
            readout = model.update(frame)   # frame: (H,W,C) float
    """

    def __init__(self, height, width, channels=3, config: KeySeerConfig = None):
        self.cfg = config or KeySeerConfig()
        self.H, self.W, self.C = height, width, channels
        M = self.cfg.n_components
        dt = self.cfg.dtype

        # Parametros del modelo
        self.w = np.zeros((height, width, M), dtype=dt)          # pesos
        self.mu = np.zeros((height, width, M, channels), dtype=dt)  # medias
        self.var = np.full((height, width, M), self.cfg.var_init, dtype=dt)

        # Instante de nacimiento de cada componente. -1 = nunca nacio.
        # Si un slot se recicla, su birth_time se sobrescribe, lo que
        # automaticamente invalida la consolidacion pendiente de la
        # componente anterior (murio antes de consolidar -> aporta 0).
        self.birth_time = np.full((height, width, M), -1, dtype=np.int32)

        self.n_seen = 0

    # ------------------------------------------------------------------ #
    # utilidades numericas
    # ------------------------------------------------------------------ #
    def _log_gaussian(self, d2, var):
        """log N(x; mu, var*I) dado d2 = ||x-mu||^2. Formas (H,W,M)."""
        C = self.C
        return -0.5 * C * np.log(2.0 * np.pi * var) - d2 / (2.0 * var)

    @staticmethod
    def _logsumexp(a, axis, mask=None):
        """logsumexp estable. `mask` marca entradas validas."""
        if mask is not None:
            a = np.where(mask, a, -np.inf)
        amax = np.max(a, axis=axis, keepdims=True)
        amax_safe = np.where(np.isfinite(amax), amax, 0.0)
        s = np.sum(np.exp(a - amax_safe), axis=axis, keepdims=True)
        out = np.log(np.maximum(s, _EPS)) + amax_safe
        return np.squeeze(out, axis=axis)

    def _kl_gaussian_iso(self, mu1, var1, mu2, var2):
        """
        KL( N(mu1, var1 I) || N(mu2, var2 I) ) en R^C, forma (H,W,M).

            = C/2 [ log(var2/var1) + var1/var2 - 1 ] + ||mu1-mu2||^2 / (2 var2)
        """
        C = self.C
        var1 = np.maximum(var1, _EPS)
        var2 = np.maximum(var2, _EPS)
        dmu2 = np.sum((mu1 - mu2) ** 2, axis=-1)
        return (0.5 * C * (np.log(var2 / var1) + var1 / var2 - 1.0)
                + dmu2 / (2.0 * var2))

    # ------------------------------------------------------------------ #
    # paso principal
    # ------------------------------------------------------------------ #
    def update(self, frame) -> FrameReadout:
        """Procesa un frame (H,W,C) y devuelve las lecturas."""
        cfg = self.cfg
        dt = cfg.dtype
        frame = np.asarray(frame, dtype=dt)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        assert frame.shape == (self.H, self.W, self.C), \
            f"esperaba {(self.H, self.W, self.C)}, recibi {frame.shape}"

        M = cfg.n_components
        active = self.w > _EPS                                   # (H,W,M)

        # --- distancias a cada componente --------------------------------
        diff = frame[:, :, None, :] - self.mu                    # (H,W,M,C)
        d2 = np.sum(diff * diff, axis=-1)                        # (H,W,M)

        # --- 1) SORPRESA PREDICTIVA (antes de actualizar) ----------------
        log_pdf = self._log_gaussian(d2, np.maximum(self.var, _EPS))
        log_w = np.log(np.maximum(self.w, _EPS))
        any_active = np.any(active, axis=2)
        log_px = self._logsumexp(log_w + log_pdf, axis=2, mask=active)
        # pixeles sin modelo aun: sorpresa maxima acotada por prior uniforme
        uniform_surprisal = self.C * np.log(256.0)
        surprisal = np.where(any_active, -log_px, uniform_surprisal).astype(dt)
        surprisal = np.clip(surprisal, 0.0, None)

        # --- emparejamiento (test de Mahalanobis a 3 sigma) --------------
        thr = (cfg.mahalanobis_thresh ** 2) * self.var * self.C
        matches = active & (d2 < thr)                            # (H,W,M)
        has_match = np.any(matches, axis=2)                      # (H,W)

        # componente emparejada = la de mayor peso entre las que pasan
        w_masked = np.where(matches, self.w, -np.inf)
        m_idx = np.argmax(w_masked, axis=2)                      # (H,W)

        onehot = np.zeros((self.H, self.W, M), dtype=bool)
        np.put_along_axis(onehot, m_idx[:, :, None], True, axis=2)
        onehot &= has_match[:, :, None]

        # --- guardar estado previo para la KL ----------------------------
        w_old = self.w.copy()
        mu_old = self.mu.copy()
        var_old = self.var.copy()
        active_old = active.copy()

        # --- 2) actualizacion de pesos (Zivkovic) ------------------------
        o = onehot.astype(dt)
        w_new = self.w + cfg.alpha * (o - self.w) - cfg.alpha * cfg.c_T
        w_new = np.where(active, w_new, self.w)   # inactivas no se tocan aun

        # poda: peso negativo -> componente muere
        died = active & (w_new <= 0.0)
        w_new = np.where(died, 0.0, w_new)
        w_new = np.maximum(w_new, 0.0)

        # --- actualizacion de mu y sigma^2 de la emparejada --------------
        w_matched = np.sum(np.where(onehot, w_new, 0.0), axis=2)  # (H,W)
        rho = np.where(w_matched > _EPS, cfg.alpha / np.maximum(w_matched, _EPS), 0.0)
        rho = np.clip(rho, 0.0, 1.0)[:, :, None]                  # (H,W,1)

        mu_new = self.mu.copy()
        var_new = self.var.copy()

        upd = onehot[:, :, :, None]                               # (H,W,M,1)
        mu_new = np.where(upd, self.mu + rho[:, :, :, None] * diff, mu_new)

        # MLE isotropico correcto: ||delta||^2 / C  (ver nota del modulo)
        d2_over_C = d2 / self.C
        var_new = np.where(onehot,
                           self.var + rho * (d2_over_C - self.var),
                           var_new)
        var_new = np.clip(var_new, cfg.var_min, cfg.var_max)

        # --- 3) NACIMIENTO de componente ---------------------------------
        births = ~has_match                                        # (H,W)
        if np.any(births):
            # elegir slot: uno inactivo si existe, si no el de menor peso
            slot_score = np.where(w_new > _EPS, w_new, -1.0)
            birth_slot = np.argmin(slot_score, axis=2)             # (H,W)

            bh = np.zeros((self.H, self.W, M), dtype=bool)
            np.put_along_axis(bh, birth_slot[:, :, None], True, axis=2)
            bh &= births[:, :, None]

            w_new = np.where(bh, cfg.alpha_init, w_new)
            var_new = np.where(bh, cfg.var_init, var_new)
            mu_new = np.where(bh[:, :, :, None],
                              np.broadcast_to(frame[:, :, None, :], mu_new.shape),
                              mu_new)
            # registrar nacimiento (sobrescribe cualquier pendiente en ese slot)
            self.birth_time = np.where(bh, self.n_seen, self.birth_time)

        # las componentes podadas dejan de tener nacimiento valido
        self.birth_time = np.where(w_new <= _EPS, -1, self.birth_time)

        # --- renormalizar pesos ------------------------------------------
        wsum = np.sum(w_new, axis=2, keepdims=True)
        w_new = np.where(wsum > _EPS, w_new / np.maximum(wsum, _EPS), w_new)

        # --- 4) SORPRESA BAYESIANA: KL(theta_t || theta_{t-1}) -----------
        # cota superior por emparejamiento de componentes (correspondencia
        # 1-a-1 garantizada: es el mismo GMM actualizado incrementalmente)
        active_new = w_new > _EPS
        both = active_new & active_old

        kl_gauss = self._kl_gaussian_iso(mu_new, var_new, mu_old, var_old)
        w_ratio = np.log(np.maximum(w_new, _EPS) / np.maximum(w_old, _EPS))
        kl_terms = w_new * (w_ratio + kl_gauss)
        kl_terms = np.where(both, kl_terms, 0.0)

        # componentes nacidas: aportan un termino de novedad acotado
        born = active_new & ~active_old
        kl_terms = kl_terms + np.where(born, w_new * uniform_surprisal, 0.0)

        kl = np.sum(kl_terms, axis=2).astype(dt)
        kl = np.clip(kl, 0.0, None)

        # --- mascara de foreground clasica (para comparacion) ------------
        order = np.argsort(-w_new, axis=2)
        w_sorted = np.take_along_axis(w_new, order, axis=2)
        cum = np.cumsum(w_sorted, axis=2)
        is_bg_rank = cum - w_sorted < cfg.bg_ratio
        is_bg_sorted = np.zeros_like(is_bg_rank)
        np.put_along_axis(is_bg_sorted, order, is_bg_rank, axis=2)
        matched_is_bg = np.sum(np.where(onehot, is_bg_sorted, False), axis=2) > 0
        fg_mask = ~(has_match & matched_is_bg)

        # --- commit -------------------------------------------------------
        self.w, self.mu, self.var = w_new, mu_new, var_new
        idx = self.n_seen
        self.n_seen += 1

        # --- 5) CONSOLIDACION (salida retardada L frames) ----------------
        # Psi_t(x) = peso actual de la componente nacida en t, si sobrevivio.
        # El prior de Dirichlet negativo ya podo las espurias: lo que queda
        # con peso apreciable es novedad genuina. Es un test de hipotesis
        # gratuito, implicito en la dinamica del modelo.
        consolidation = None
        cons_idx = -1
        target = self.n_seen - 1 - cfg.latency
        if target >= 0:
            survived = (self.birth_time == target)
            consolidation = np.sum(np.where(survived, self.w, 0.0),
                                   axis=2).astype(dt)
            cons_idx = target

        warmup = self.n_seen < int(1.0 / max(cfg.alpha, _EPS))

        return FrameReadout(
            index=idx,
            surprisal=surprisal,
            kl=kl,
            births=births,
            fg_mask=fg_mask,
            n_active=np.sum(active_new, axis=2).astype(np.int16),
            warmup=warmup,
            consolidation=consolidation,
            consolidation_index=cons_idx,
        )

    # ------------------------------------------------------------------ #
    def background_image(self):
        """Imagen de fondo estimada: media de la componente dominante."""
        idx = np.argmax(self.w, axis=2)
        return np.take_along_axis(self.mu, idx[:, :, None, None], axis=2)[:, :, 0, :]
