"""Run the experiments of the paper.

    python scripts/run_experiments.py base controlled real ncal eps

Settings were fixed before looking at any external-cohort result:
referable DR = ICDR grade >= 2, FOR target 2%, confidence 90%, sensitivity
floor 90%, 200 repetitions. Results are written to results/.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from paths import FEATURES, RESULTS
import pacr

EPS = 0.02          # target false-omission rate
DELTA = 0.10        # 1 - confidence
SENS_FLOOR = 0.90
BACKBONE = "dinov2_s"
COHORTS = {"eyepacs": "EyePACS", "aptos": "APTOS", "ddr": "DDR", "idrid": "IDRiD"}
REPS = 200
ALL_EXPERIMENTS = ["base", "controlled", "real", "ncal", "eps"]


def load(cohort):
    d = np.load(FEATURES / f"{cohort}_{BACKBONE}.npz")
    df = pd.DataFrame({"image_id": d["image_id"], "patient_id": d["patient_id"],
                       "split": d["split"], "grade": d["label"]})
    df["y"] = (df["grade"] >= 2).astype(int)
    df["cohort"] = cohort
    return d["feats"].astype(np.float32), df


def shuffled_patients(df, rng):
    # object dtype so that shuffling never mixes up pandas' Arrow-backed strings
    pats = np.array(df["patient_id"].unique(), dtype=object)
    rng.shuffle(pats)
    return pats


def split_by_patient(df, fractions, seed):
    pats = shuffled_patients(df, np.random.default_rng(seed))
    cuts = np.cumsum([int(round(f * len(pats))) for f in fractions.values()])[:-1]
    part = {p: name for name, chunk in zip(fractions, np.split(pats, cuts)) for p in chunk}
    return df["patient_id"].map(part).values


def fit_head(X, y, groups):
    """Logistic regression on frozen features; C chosen by grouped 5-fold CV AUC."""
    best_auc, best_c = -1, None
    for c in [0.003, 0.01, 0.03, 0.1]:
        aucs = []
        for tr, va in GroupKFold(5).split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        if np.mean(aucs) > best_auc:
            best_auc, best_c = np.mean(aucs), c
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=5000)).fit(X, y)
    return model, best_auc, best_c


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(raw_tune, y_tune):
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(logit(raw_tune)[:, None], y_tune)
    return lambda raw: lr.predict_proba(logit(raw)[:, None])[:, 1]


def ece(s, y, bins=15):
    idx = np.clip(np.digitize(s, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(sum(abs(s[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def bootstrap_auc(s, y, groups, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        take = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
        if y[take].min() != y[take].max():
            vals.append(roc_auc_score(y[take], s[take]))
    return float(roc_auc_score(y, s)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def thresholds(s_cal, y_cal, s_unl, s_tune, y_tune, pi_true, seed, eps=EPS, ablations=True):
    """Release threshold of every method for one calibration set and deployment sample."""
    alpha = 1 - SENS_FLOOR
    t_floor = pacr.class_conditional_threshold(s_cal, y_cal, alpha)
    pi_s = y_cal.mean()
    pi_hat, pi_ub = pacr.pacc_upper_bound(s_unl, s_cal, y_cal, DELTA / 2, rng=seed)
    out = {
        "_pi_hat": pi_hat,
        "_pi_ub": pi_ub,
        "_pi_bbse": pacr.bbse_prevalence(s_unl, s_cal, y_cal),
        "_pi_em": pacr.em_prevalence(s_unl, pi_s),
        "Fixed operating point": pacr.sensitivity_threshold(s_tune, y_tune, SENS_FLOOR),
        "Split conformal": pacr.split_conformal_threshold(s_cal, y_cal, alpha),
        "Class-conditional conformal": t_floor,
        "Label-shift weighted conformal": pacr.weighted_conformal_threshold(s_cal, y_cal, alpha, pi_hat),
        "Source FOR control": pacr.pacr_threshold(s_cal, y_cal, eps, DELTA / 2, pi_s, t_max=t_floor),
        "PACR (ours)": pacr.pacr_threshold(s_cal, y_cal, eps, DELTA / 2, pi_ub, t_max=t_floor),
        "PACR, oracle prevalence": pacr.pacr_threshold(s_cal, y_cal, eps, DELTA / 2, pi_true, t_max=t_floor),
    }
    if ablations:
        out["Ablation: plug-in (no bounds)"] = pacr.plugin_for_threshold(s_cal, y_cal, eps, pi_hat, t_max=t_floor)
        _, em_ub = pacr.em_upper_bound(s_unl, pi_s, DELTA / 2, n_boot=200, rng=seed)
        out["Ablation: EM prevalence"] = pacr.pacr_threshold(s_cal, y_cal, eps, DELTA / 2, em_ub, t_max=t_floor)
        rng = np.random.default_rng(seed + 7)
        boots = [pacr.bbse_prevalence(s_unl[rng.integers(0, len(s_unl), len(s_unl))], s_cal, y_cal)
                 for _ in range(200)]
        bb_ub = float(np.quantile(boots, 1 - DELTA / 2))
        out["Ablation: BBSE prevalence"] = pacr.pacr_threshold(s_cal, y_cal, eps, DELTA / 2, bb_ub, t_max=t_floor)
        out["Ablation: no sensitivity floor"] = pacr.pacr_threshold(s_cal, y_cal, eps, DELTA / 2, pi_ub)
    return out


def population_metrics(t, s, y, pi):
    """FOR, sensitivity and workload reduction in a population with prevalence pi
    whose class-conditional score distributions are those of (s, y)."""
    f1 = (s[y == 1] < t).mean()
    f0 = (s[y == 0] < t).mean()
    wr = pi * f1 + (1 - pi) * f0
    return {"FOR": float(pi * f1 / wr) if wr > 0 else 0.0,
            "sensitivity": float(1 - f1), "workload_reduction": float(wr)}


def records(th, **fields):
    prev = {k[1:]: th[k] for k in ("_pi_hat", "_pi_em", "_pi_ub", "_pi_bbse")}
    for method, t in th.items():
        if not method.startswith("_"):
            yield dict(fields, method=method, threshold=t, **prev)


def build_scores():
    """Train the EyePACS classifier and score every cohort."""
    X, df = load("eyepacs")
    df["part"] = split_by_patient(df, {"train": 0.60, "tune": 0.10, "pool": 0.30}, seed=2026)
    tr, tune = df["part"].values == "train", df["part"].values == "tune"
    head, cv_auc, c = fit_head(X[tr], df["y"].values[tr], df["patient_id"].values[tr])
    calibrate = platt(head.predict_proba(X[tune])[:, 1], df["y"].values[tune])
    df["s"] = calibrate(head.predict_proba(X)[:, 1])
    scores = {"eyepacs": df}
    for cohort in list(COHORTS)[1:]:
        Xc, dc = load(cohort)
        dc["s"] = calibrate(head.predict_proba(Xc)[:, 1])
        dc["part"] = "target"
        scores[cohort] = dc
    return scores, {"cv_auc": cv_auc, "C": c}


def exp_base(scores):
    tune = scores["eyepacs"].query("part == 'tune'")
    rows = []
    for cohort, d in scores.items():
        sub = d[d["part"] == "pool"] if cohort == "eyepacs" else d
        s, y = sub["s"].values, sub["y"].values
        auc, lo, hi = bootstrap_auc(s, y, sub["patient_id"].values)
        rows.append(dict(cohort=COHORTS[cohort], n_images=len(d), n_patients=d["patient_id"].nunique(),
                         n_eval=len(sub), prevalence=y.mean(), auc=auc, auc_lo=lo, auc_hi=hi, ece=ece(s, y),
                         pi_em=pacr.em_prevalence(s, tune["y"].mean()),
                         pi_hat=pacr.pacc_prevalence(s, tune["s"].values, tune["y"].values),
                         grade_counts=json.dumps(d["grade"].value_counts().sort_index().to_dict())))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "base_performance.csv", index=False)
    print(out.drop(columns="grade_counts").round(3).to_string())


def exp_controlled(scores, prevalences=(0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
                   n_cal=None, n_unl=1000, tag="controlled", eps=EPS, ablations=True):
    """Exact label shift: EyePACS held-out patients resampled to each prevalence."""
    d = scores["eyepacs"]
    pool = d[d["part"] == "pool"].reset_index(drop=True)
    tune = d[d["part"] == "tune"]
    rows, t0 = [], time.time()
    for r in range(REPS):
        rng = np.random.default_rng(10_000 + r)
        pats = shuffled_patients(pool, rng)
        k = len(pats) // 3
        role = {**{p: "cal" for p in pats[:k]}, **{p: "unl" for p in pats[k:2 * k]}, **{p: "eval" for p in pats[2 * k:]}}
        role = pool["patient_id"].map(role).values
        cal, unl, ev = pool[role == "cal"], pool[role == "unl"], pool[role == "eval"]
        if n_cal is not None:
            cal = cal.iloc[rng.choice(len(cal), min(n_cal, len(cal)), replace=False)]
        upos, uneg = unl.loc[unl["y"] == 1, "s"].values, unl.loc[unl["y"] == 0, "s"].values
        for pi in prevalences:
            k1 = rng.binomial(n_unl, pi)
            s_unl = np.concatenate([rng.choice(upos, k1, replace=True), rng.choice(uneg, n_unl - k1, replace=True)])
            th = thresholds(cal["s"].values, cal["y"].values, s_unl, tune["s"].values, tune["y"].values,
                            pi, seed=r, eps=eps, ablations=ablations)
            for rec in records(th, rep=r, pi=pi):
                rec.update(population_metrics(rec["threshold"], ev["s"].values, ev["y"].values, pi))
                rows.append(rec)
        if r % 20 == 0:
            print(f"{tag}: rep {r} ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / f"{tag}.csv", index=False)
    return res


def exp_real(scores, deployments=("native", 0.10, 0.05), local_m=(100, 200, 400), n_unl=1000):
    """Deployment on each cohort at its native prevalence and re-weighted to 10% and 5%.

    Re-weighting only changes the class proportions, so all differences in cameras
    and populations between EyePACS and the cohort are kept.
    """
    pool = scores["eyepacs"].query("part == 'pool'")
    tune = scores["eyepacs"].query("part == 'tune'")
    rows = []
    for cohort, dc in scores.items():
        target = pool if cohort == "eyepacs" else dc
        for r in range(REPS):
            rng = np.random.default_rng(20_000 + r)
            pats = shuffled_patients(pool, rng)
            cal_pats = set(pats[: len(pats) // 2])
            cal = pool[pool["patient_id"].isin(cal_pats)]
            s_cal, y_cal = cal["s"].values, cal["y"].values
            tg = target[~target["patient_id"].isin(cal_pats)] if cohort == "eyepacs" else target
            tp = shuffled_patients(tg, rng)
            half = set(tp[: len(tp) // 2])
            unl, ev = tg[tg["patient_id"].isin(half)], tg[~tg["patient_id"].isin(half)]
            upos, uneg = unl.loc[unl["y"] == 1, "s"].values, unl.loc[unl["y"] == 0, "s"].values
            s_ev, y_ev = ev["s"].values, ev["y"].values
            for dep in deployments:
                if dep == "native":
                    pi = float(y_ev.mean())
                    s_unl, y_unl = unl["s"].values, unl["y"].values
                else:
                    pi = float(dep)
                    n = min(n_unl, len(unl))
                    k1 = rng.binomial(n, pi)
                    s_unl = np.concatenate([upos[rng.integers(0, len(upos), k1)], uneg[rng.integers(0, len(uneg), n - k1)]])
                    y_unl = np.r_[np.ones(k1, int), np.zeros(n - k1, int)]
                th = thresholds(s_cal, y_cal, s_unl, tune["s"].values, tune["y"].values, pi, seed=r, ablations=False)
                # Variant: a prevalence bound from m labelled deployment images instead of label shift.
                t_floor = pacr.class_conditional_threshold(s_cal, y_cal, 1 - SENS_FLOOR)
                for m in local_m:
                    if m <= len(y_unl):
                        labelled = y_unl[rng.choice(len(y_unl), m, replace=False)]
                        pi_u = float(pacr.clopper_pearson_upper(labelled.sum(), m, DELTA / 2))
                        th[f"PACR + local prevalence (m={m})"] = pacr.pacr_threshold(
                            s_cal, y_cal, EPS, DELTA / 2, pi_u, t_max=t_floor)
                label = "native" if dep == "native" else f"{round(pi * 100)}%"
                for rec in records(th, cohort=COHORTS[cohort], deploy=label, rep=r, pi_true=pi):
                    rec.update(population_metrics(rec["threshold"], s_ev, y_ev, pi))
                    rows.append(rec)
        print("real:", cohort, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "real.csv", index=False)
    return res


def main():
    global REPS
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", help="any of: " + " ".join(ALL_EXPERIMENTS))
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()
    todo = args.experiments or ALL_EXPERIMENTS
    unknown = set(todo) - set(ALL_EXPERIMENTS)
    if unknown:
        ap.error(f"unknown experiment(s): {', '.join(sorted(unknown))}")
    REPS = args.reps
    RESULTS.mkdir(exist_ok=True)

    scores, info = build_scores()
    json.dump(info, open(RESULTS / "source_model.json", "w"))
    for cohort, d in scores.items():
        d.drop(columns="split").to_csv(RESULTS / f"scores_{cohort}.csv", index=False)
    print("EyePACS head:", info)

    if "base" in todo:
        exp_base(scores)
    if "controlled" in todo:
        exp_controlled(scores)
    if "real" in todo:
        exp_real(scores)
    if "ncal" in todo:
        for n in [250, 500, 1000, 2000]:
            exp_controlled(scores, prevalences=(0.02, 0.05, 0.10), n_cal=n, tag=f"ncal_{n}", ablations=False)
    if "eps" in todo:
        for e in [0.01, 0.05]:
            exp_controlled(scores, tag=f"eps_{e}", eps=e, ablations=False)


if __name__ == "__main__":
    main()
