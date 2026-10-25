"""Estimating the prevalence of referable DR in an unlabelled deployment sample.

All estimators assume label shift: p(x | y) is the same in the source and the
deployment population, only p(y) changes. Scores are classifier outputs that
were calibrated on the source population.
"""
import numpy as np


def pacc_prevalence(s_target, s_cal, y_cal):
    """Probabilistic adjusted classify-and-count (Bella et al., 2010).

    Under label shift E_t[s] = pi * m1 + (1 - pi) * m0, where m_y = E[s | y] is
    measured on the labelled calibration set. Solving for pi gives a moment
    estimator that does not need the scores to be calibrated.
    """
    m1 = s_cal[y_cal == 1].mean()
    m0 = s_cal[y_cal == 0].mean()
    return float(np.clip((np.mean(s_target) - m0) / (m1 - m0), 0, 1))


def pacc_upper_bound(s_target, s_cal, y_cal, delta, n_boot=1000, rng=None):
    """PACC estimate and its (1 - delta) percentile-bootstrap upper bound.

    The unlabelled sample and both classes of the calibration set are resampled,
    so the uncertainty in m0 and m1 is carried into the bound.
    """
    rng = np.random.default_rng(rng)
    s_target = np.asarray(s_target, float)
    pos, neg = s_cal[y_cal == 1], s_cal[y_cal == 0]

    est = pacc_prevalence(s_target, s_cal, y_cal)
    mt = s_target[rng.integers(0, len(s_target), (n_boot, len(s_target)))].mean(1)
    m1 = pos[rng.integers(0, len(pos), (n_boot, len(pos)))].mean(1)
    m0 = neg[rng.integers(0, len(neg), (n_boot, len(neg)))].mean(1)
    boots = np.clip((mt - m0) / (m1 - m0), 0, 1)
    return est, float(np.quantile(boots, 1 - delta))


def _mlls_batch(S, pi_source, n_bisect=60):
    # The EM fixed point of Saerens et al. (2002) maximises a concave log-likelihood
    # in pi, so we find it directly by bisection on the derivative.
    S = np.clip(np.asarray(S, float), 1e-6, 1 - 1e-6)
    a = S / pi_source
    b = (1 - S) / (1 - pi_source)
    d = a - b

    def grad(pi):
        return (d / (pi[:, None] * a + (1 - pi[:, None]) * b)).sum(axis=1)

    lo, hi = np.zeros(S.shape[0]), np.ones(S.shape[0])
    g0, g1 = grad(lo), grad(hi)
    for _ in range(n_bisect):
        mid = (lo + hi) / 2
        up = grad(mid) > 0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    est = (lo + hi) / 2
    est = np.where(g0 <= 0, 0.0, est)
    est = np.where(g1 >= 0, 1.0, est)
    return est


def em_prevalence(s_target, pi_source):
    """Maximum-likelihood label-shift estimate (EM / MLLS)."""
    return float(_mlls_batch(np.asarray(s_target, float)[None, :], pi_source)[0])


def em_upper_bound(s_target, pi_source, delta, n_boot=200, rng=None):
    """EM estimate and its (1 - delta) percentile-bootstrap upper bound."""
    rng = np.random.default_rng(rng)
    s_target = np.asarray(s_target, float)
    n = len(s_target)
    est = em_prevalence(s_target, pi_source)
    boots = _mlls_batch(s_target[rng.integers(0, n, (n_boot, n))], pi_source)
    return est, float(np.quantile(boots, 1 - delta))


def bbse_prevalence(s_target, s_cal, y_cal, thr=0.5):
    """Black-box shift estimation (Lipton et al., 2018) with a hard decision at `thr`."""
    yhat = (s_cal >= thr).astype(int)
    conf = np.array([[np.mean((yhat == a) & (y_cal == b)) for b in (0, 1)] for a in (0, 1)])
    q = np.array([np.mean(s_target < thr), np.mean(s_target >= thr)])
    pi_s = y_cal.mean()
    try:
        w = np.linalg.solve(conf, q)  # p_t(y) / p_s(y)
        return float(np.clip(w[1] * pi_s, 0, 1))
    except np.linalg.LinAlgError:
        return float(pi_s)
