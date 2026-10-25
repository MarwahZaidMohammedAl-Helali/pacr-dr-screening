"""How often does each prevalence upper bound cover the true prevalence?

Uses the same splits and seeds as the controlled-shift experiment. Needs
results/scores_eyepacs.csv from run_experiments.py.
"""
import numpy as np
import pandas as pd

from paths import RESULTS
import pacr

DELTA = 0.10
PREVALENCES = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
N_UNL = 1000


def main(reps=200):
    d = pd.read_csv(RESULTS / "scores_eyepacs.csv")
    pool = d[d["part"] == "pool"].reset_index(drop=True)
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(10_000 + r)
        pats = np.array(pool["patient_id"].unique(), dtype=object)
        rng.shuffle(pats)
        k = len(pats) // 3
        role = pool["patient_id"].map({**{p: "cal" for p in pats[:k]}, **{p: "unl" for p in pats[k:2 * k]},
                                       **{p: "eval" for p in pats[2 * k:]}}).values
        cal, unl = pool[role == "cal"], pool[role == "unl"]
        s_cal, y_cal = cal["s"].values, cal["y"].values
        upos, uneg = unl.loc[unl["y"] == 1, "s"].values, unl.loc[unl["y"] == 0, "s"].values
        for pi in PREVALENCES:
            k1 = rng.binomial(N_UNL, pi)
            s_unl = np.concatenate([rng.choice(upos, k1, replace=True), rng.choice(uneg, N_UNL - k1, replace=True)])
            pacc, pacc_ub = pacr.pacc_upper_bound(s_unl, s_cal, y_cal, DELTA / 2, rng=r)
            em, em_ub = pacr.em_upper_bound(s_unl, y_cal.mean(), DELTA / 2, n_boot=200, rng=r)
            rb = np.random.default_rng(r + 7)
            boots = [pacr.bbse_prevalence(s_unl[rb.integers(0, N_UNL, N_UNL)], s_cal, y_cal) for _ in range(200)]
            rows.append(dict(rep=r, pi=pi, pacc=pacc, pacc_ub=pacc_ub, em=em, em_ub=em_ub,
                             bbse=pacr.bbse_prevalence(s_unl, s_cal, y_cal),
                             bbse_ub=float(np.quantile(boots, 1 - DELTA / 2))))
        if r % 50 == 0:
            print("rep", r, flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "prev_coverage.csv", index=False)
    summary = res.groupby("pi")[["pacc", "em", "bbse"]].mean()
    for est in ("pacc", "em", "bbse"):
        summary[f"{est}_coverage"] = (res[f"{est}_ub"] >= res["pi"]).groupby(res["pi"]).mean()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
