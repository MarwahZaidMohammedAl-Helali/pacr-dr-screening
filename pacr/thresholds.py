"""Release thresholds for rule-out triage.

An image is released without human grading when its score s is below the
threshold t, and referred otherwise. A threshold of -inf releases nothing.

For a population with prevalence pi:
    F1(t)  = P(s < t | y = 1)            referable cases released (1 - sensitivity)
    F0(t)  = P(s < t | y = 0)            non-referable cases released
    WR(t)  = pi F1 + (1 - pi) F0         workload reduction
    FOR(t) = pi F1 / WR(t)               false-omission rate among released cases
"""
import numpy as np
from scipy.stats import beta

# Candidate thresholds, fixed before seeing any data: 40 points equally spaced on
# the logit scale between -6 and 2 (probabilities 0.0025 to 0.88).
GRID = 1 / (1 + np.exp(-np.linspace(-6, 2, 40)))


def clopper_pearson_upper(k, n, alpha):
    k = np.asarray(k, float)
    return np.where(k >= n, 1.0, beta.ppf(1 - alpha, k + 1, np.maximum(n - k, 1e-12)))


def clopper_pearson_lower(k, n, alpha):
    k = np.asarray(k, float)
    return np.where(k <= 0, 0.0, beta.ppf(alpha, np.maximum(k, 1e-12), n - k + 1))


def false_omission_rate(pi, f1, f0):
    den = pi * f1 + (1 - pi) * f0
    return np.where(den > 0, pi * f1 / np.maximum(den, 1e-300), 0.0)


def for_upper_bound(s_cal, y_cal, pi_upper, delta, grid=GRID):
    """Simultaneous upper bound on FOR(t) for every t in the grid.

    Uses a Clopper-Pearson upper bound on F1 and lower bound on F0, each at level
    delta / (2 * len(grid)). FOR increases with pi and F1 and decreases with F0,
    so plugging in pi_upper gives a valid bound whenever pi <= pi_upper.
    """
    pos, neg = np.sort(s_cal[y_cal == 1]), np.sort(s_cal[y_cal == 0])
    a = delta / (2 * len(grid))
    k1 = np.searchsorted(pos, grid, side="left")
    k0 = np.searchsorted(neg, grid, side="left")
    f1_hi = clopper_pearson_upper(k1, len(pos), a)
    f0_lo = clopper_pearson_lower(k0, len(neg), a)
    return np.where(f0_lo > 0, false_omission_rate(pi_upper, f1_hi, f0_lo), 1.0)


def pacr_threshold(s_cal, y_cal, eps, delta, pi_upper, t_max=np.inf, grid=GRID):
    """Largest grid threshold whose FOR bound is <= eps, capped at t_max.

    t_max is the sensitivity-floor threshold (class-conditional conformal).
    """
    ok = (for_upper_bound(s_cal, y_cal, pi_upper, delta, grid) <= eps) & (grid <= t_max)
    return grid[ok].max() if ok.any() else -np.inf


def plugin_for_threshold(s_cal, y_cal, eps, pi_hat, t_max=np.inf, grid=GRID):
    """Same rule with point estimates instead of confidence bounds (ablation)."""
    pos, neg = s_cal[y_cal == 1], s_cal[y_cal == 0]
    f1 = np.array([(pos < t).mean() for t in grid])
    f0 = np.array([(neg < t).mean() for t in grid])
    fr = np.where(f0 > 0, false_omission_rate(pi_hat, f1, f0), 1.0)
    ok = (fr <= eps) & (grid <= t_max)
    return grid[ok].max() if ok.any() else -np.inf


def split_conformal_threshold(s_cal, y_cal, alpha):
    """Marginal split conformal with the LAC score 1 - p_y.

    The prediction set is {0} exactly when s < 1 - qhat (for alpha < 0.5).
    """
    n = len(s_cal)
    scores = np.where(y_cal == 1, 1 - s_cal, s_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return -np.inf
    return 1 - np.sort(scores)[k - 1]


def weighted_conformal_threshold(s_cal, y_cal, alpha, pi_target):
    """Label-shift weighted split conformal (Podkopaev and Ramdas, 2021)."""
    pi_s = y_cal.mean()
    w = np.where(y_cal == 1, pi_target / pi_s, (1 - pi_target) / (1 - pi_s))
    scores = np.where(y_cal == 1, 1 - s_cal, s_cal)
    order = np.argsort(scores)
    scores, w = scores[order], w[order]
    w_test = max(pi_target / pi_s, (1 - pi_target) / (1 - pi_s))
    cdf = np.cumsum(w) / (w.sum() + w_test)
    idx = np.searchsorted(cdf, 1 - alpha)
    if idx >= len(scores):
        return -np.inf
    return 1 - scores[idx]


def class_conditional_threshold(s_cal, y_cal, alpha):
    """Mondrian conformal on the referable class: P(s < t | y = 1) <= alpha."""
    pos = np.sort(s_cal[y_cal == 1])
    k = int(np.floor(alpha * (len(pos) + 1)))
    return -np.inf if k < 1 else pos[k - 1]


def sensitivity_threshold(s, y, sens):
    """Operating point with the given sensitivity on a labelled tuning set."""
    pos = np.sort(s[y == 1])
    k = int(np.floor((1 - sens) * len(pos)))
    return -np.inf if k < 1 else pos[k - 1]
