import numpy as np
from mog3 import MOG3, MOG3Config
from mog3.metrics import spatial_coherence, spatial_pool


def _bg(H=16, W=16, C=3, rng=None):
    rng = rng or np.random.default_rng(0)
    return np.full((H, W, C), 50.0) + rng.standard_normal((H, W, C)) * 2


def test_weights_normalized():
    m = MOG3(16, 16, 3, MOG3Config(alpha=0.05))
    rng = np.random.default_rng(0)
    for _ in range(30):
        m.update(_bg(rng=rng))
    assert np.allclose(m.w.sum(axis=2), 1.0, atol=1e-4)


def test_variance_bounded():
    cfg = MOG3Config(alpha=0.05)
    m = MOG3(16, 16, 3, cfg)
    rng = np.random.default_rng(1)
    for _ in range(40):
        m.update(_bg(rng=rng) * rng.uniform(0.5, 2.0))
    assert m.var.min() >= cfg.var_min - 1e-5
    assert m.var.max() <= cfg.var_max + 1e-5


def test_surprisal_nonneg_and_drops_when_stable():
    m = MOG3(16, 16, 3, MOG3Config(alpha=0.1))
    rng = np.random.default_rng(2)
    first = m.update(_bg(rng=rng)).surprisal.mean()
    for _ in range(60):
        r = m.update(_bg(rng=rng))
    assert r.surprisal.min() >= 0.0
    assert r.surprisal.mean() < first


def test_consolidation_separates_transient_from_persistent():
    """Claim central: Psi distingue novedad transitoria de persistente."""
    L = 15
    cons = {}

    def run(persist_frames):
        m = MOG3(16, 16, 3, MOG3Config(alpha=0.04, latency=L))
        rng = np.random.default_rng(3)
        out = {}
        for _ in range(80):
            m.update(_bg(rng=rng))
        onset = m.n_seen
        for _ in range(persist_frames):
            f = _bg(rng=rng); f[4:12, 4:12] = 200.0
            r = m.update(f)
            if r.consolidation is not None:
                out[r.consolidation_index] = r.consolidation.mean()
        for _ in range(L + 5):
            r = m.update(_bg(rng=rng))
            if r.consolidation is not None:
                out[r.consolidation_index] = r.consolidation.mean()
        return out.get(onset, 0.0)

    transient = run(2)
    persistent = run(40)
    assert persistent > 5 * transient, (transient, persistent)


def test_instantaneous_signals_are_blind_at_onset():
    """Proposicion 1: births no puede distinguir en el frame de inicio."""
    def onset_births(persist):
        m = MOG3(16, 16, 3, MOG3Config(alpha=0.04))
        rng = np.random.default_rng(4)
        for _ in range(80):
            m.update(_bg(rng=rng))
        f = _bg(rng=rng); f[4:12, 4:12] = 200.0
        return m.update(f).births.mean()
    assert abs(onset_births(2) - onset_births(40)) < 1e-9


def test_coherence_ranks_structure_above_sparse_noise():
    rng = np.random.default_rng(5)
    blob = np.zeros((40, 40)); blob[12:28, 12:28] = 1.0
    sparse = (rng.random((40, 40)) < 0.3).astype(float)
    assert spatial_coherence(blob) > spatial_coherence(sparse)
    # un corte de escena es global pero coherente: no debe penalizarse
    assert spatial_coherence(np.ones((40, 40))) > 0.95


def test_spatial_pool_scalar_and_finite():
    v = spatial_pool(np.random.default_rng(6).random((64, 64)))
    assert np.isfinite(v) and v >= 0.0
