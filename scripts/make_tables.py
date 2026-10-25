"""LaTeX tables of the paper, generated from the CSV files in results/."""
import json
import os

import numpy as np
import pandas as pd

from paths import RESULTS, TABLES

RES, OUT = str(RESULTS), str(TABLES)
os.makedirs(OUT, exist_ok=True)
EPS = 0.02
COH = ["EyePACS", "APTOS", "DDR", "IDRiD"]
MAIN = ["Fixed operating point", "Split conformal", "Class-conditional conformal", "Label-shift weighted conformal",
        "Source FOR control", "PACR (ours)", "PACR, oracle prevalence"]
INFO = {  # static descriptors from the dataset publications
    "EyePACS": ("USA", "Primary-care screening; multiple cameras", "\\cite{cuadros2009eyepacs,eyepacs2015kaggle}"),
    "APTOS": ("India", "Rural screening (Aravind); multiple cameras", "\\cite{aptos2019}"),
    "DDR": ("China", "147 hospitals; 42 camera models", "\\cite{li2019ddr}"),
    "IDRiD": ("India", "Eye clinic; Kowa VX-10$\\alpha$", "\\cite{porwal2018idrid}"),
}


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)


def tab_data():
    b = pd.read_csv(f"{RES}/base_performance.csv").set_index("cohort")
    rows = []
    for c in COH:
        r = b.loc[c]
        g = {int(k): v for k, v in json.loads(r["grade_counts"]).items()}
        n = sum(g.values())
        grades = " / ".join(f"{g.get(k, 0):,}" for k in range(5))
        rows.append(f"{c} {INFO[c][2]} & {INFO[c][0]} & {INFO[c][1]} & {int(r['n_images']):,} & {grades} & {pct(sum(g.get(k,0) for k in (2,3,4))/n)} \\\\")
    s = ("\\begin{table}[!htbp]\n\\centering\n\\caption{Public datasets used in this study. Grades follow the ICDR scale (0: no DR, 1: mild, 2: moderate, 3: severe non-proliferative, 4: proliferative DR); referable DR is grade $\\geq 2$. Only EyePACS was used to develop the classifier.}\n\\label{tab:data}\n\\footnotesize\n"
         "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}llp{3.6cm}rlr@{}}\n\\toprule\nCohort & Country & Setting / cameras & Images & Grades 0/1/2/3/4 & Referable (\\%) \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    open(f"{OUT}/tab_data.tex", "w").write(s)
    return b


def tab_base():
    b = pd.read_csv(f"{RES}/base_performance.csv").set_index("cohort")
    rows = []
    for c in COH:
        r = b.loc[c]
        role = "internal test" if c == "EyePACS" else "external"
        rows.append(f"{c} & {role} & {int(r['n_eval']):,} & {r['auc']:.3f} ({r['auc_lo']:.3f}--{r['auc_hi']:.3f}) & {r['ece']:.3f} & {pct(r['prevalence'])} & {pct(r['pi_hat'])} & {pct(r['pi_em'])} \\\\")
    s = ("\\begin{table}[!htbp]\n\\centering\n\\caption{Discrimination and calibration of the EyePACS-trained classifier, and prevalence estimated from the unlabelled images of each cohort (all images of the cohort, or all EyePACS held-out images) with the moment-matching estimator used by PACR (PACC) and with the maximum-likelihood EM estimator. AUC: area under the ROC curve with 95\\% patient-level bootstrap confidence interval; ECE: expected calibration error (15 bins).}\n\\label{tab:base}\n\\footnotesize\n"
         "\\begin{tabular}{@{}llrccccc@{}}\n\\toprule\n & & & & & \\multicolumn{3}{c}{Prevalence (\\%)} \\\\\n\\cmidrule(l){6-8}\nCohort & Role & Images & AUC (95\\% CI) & ECE & True & PACC & EM \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(f"{OUT}/tab_base.tex", "w").write(s)


def summarise(res, keys):
    res = res.assign(viol=res["FOR"] > EPS)
    return res.groupby(keys).agg(FOR=("FOR", "mean"), viol=("viol", "mean"), n=("viol", "size"), k=("viol", "sum"),
                                 WR=("workload_reduction", "mean"), sens=("sensitivity", "mean")).reset_index()


def fmt_row(name, r, bold=False):
    lo, hi = wilson(r["k"], r["n"])
    v = f"{pct(r['viol'],0)} ({pct(lo,0)}--{pct(hi,0)})"
    cells = [pct(r["FOR"], 2), v, pct(r["WR"]), pct(r["sens"])]
    if bold:
        cells = [f"\\textbf{{{c}}}" for c in cells]
    return cells


def tab_controlled(tag="controlled", pis=(0.05, 0.10, 0.20, 0.40), label="tab:controlled", methods=MAIN, caption=None):
    s = summarise(pd.read_csv(f"{RES}/{tag}.csv"), ["pi", "method"])
    lines = []
    for pi in pis:
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{Deployment prevalence {pct(pi,0)}\\%}}}} \\\\")
        for m in methods:
            r = s[(np.isclose(s["pi"], pi)) & (s["method"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = fmt_row(m, r, bold=(m == "PACR (ours)"))
            nm = "\\textbf{PACR (ours)}" if m == "PACR (ours)" else m
            lines.append(f"\\quad {nm} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    cap = caption or ("Controlled prevalence shift on EyePACS held-out patients (200 repetitions per prevalence; target FOR $\\varepsilon=2\\%$, sensitivity floor 90\\%, confidence 90\\%). "
                      "FOR: mean false-omission rate among released images; Violations: percentage of repetitions with FOR $>\\varepsilon$ (Wilson 95\\% CI), which should not exceed 10\\%; WR: workload reduction (percentage of images released); Sens.: sensitivity for referable DR.")
    out = ("\\begin{table}[!htbp]\n\\centering\n\\caption{" + cap + "}\n\\label{" + label + "}\n\\footnotesize\n"
           "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nMethod & FOR (\\%) & Violations (\\%) & WR (\\%) & Sens. (\\%) \\\\\n\\midrule\n"
           + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(f"{OUT}/{label.split(':')[1]}.tex", "w").write(out)
    return s


DEPLOY = ["native", "10%", "5%"]
LOCAL = "PACR + local prevalence (m=200)"


def tab_real():
    res = pd.read_csv(f"{RES}/real.csv")
    s = summarise(res, ["deploy", "cohort", "method"])
    nat = res[res["deploy"] == "native"].groupby("cohort")["pi_true"].mean()
    meths = MAIN + ([LOCAL] if LOCAL in set(s["method"]) else [])
    lines = []
    for c in COH:
        lines.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{c} (native prevalence {pct(nat[c],1)}\\%)}}}} \\\\")
        for m in meths:
            cells = []
            for dp in DEPLOY:
                r = s[(s["cohort"] == c) & (s["method"] == m) & (s["deploy"] == dp)]
                if r.empty:
                    cells += ["--"] * 3; continue
                r = r.iloc[0]
                cc = [pct(r["FOR"], 1), pct(r["viol"], 0), pct(r["WR"], 1)]
                if m == "PACR (ours)":
                    cc = [f"\\textbf{{{x}}}" for x in cc]
                cells += cc
            nm = {"PACR (ours)": "\\textbf{PACR (ours)}", LOCAL: "PACR + 200 local labels"}.get(m, m)
            lines.append(f"\\quad {nm} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    cap = ("Deployment of the EyePACS-trained classifier on four cohorts at their native prevalence and re-weighted to screening prevalences of 10\\% and 5\\% "
           "(200 repetitions; in each, half of the cohort's patients supply the unlabelled deployment sample and the other half the evaluation). "
           "FOR: mean false-omission rate among released images (\\%); Viol.: percentage of repetitions with FOR $>\\varepsilon=2\\%$ (should not exceed $\\delta=10\\%$); WR: workload reduction (\\%). "
           "Unlike in Table~\\ref{tab:controlled}, the label-shift assumption is not guaranteed here, because the cohorts also differ from EyePACS in cameras, populations and graders.")
    out = ("\\begin{table}[!htbp]\n\\centering\n\\caption{" + cap + "}\n\\label{tab:real}\n\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n"
           "\\begin{tabular}{@{}lrrrrrrrrr@{}}\n\\toprule\n & \\multicolumn{3}{c}{Native prevalence} & \\multicolumn{3}{c}{10\\% prevalence} & \\multicolumn{3}{c}{5\\% prevalence} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}\n"
           "Method & FOR & Viol. & WR & FOR & Viol. & WR & FOR & Viol. & WR \\\\\n\\midrule\n"
           + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(f"{OUT}/tab_real.tex", "w").write(out)
    return s


def tab_eps():
    rows = []
    for e, tag in [(0.01, "eps_0.01"), (0.02, "controlled"), (0.05, "eps_0.05")]:
        f = f"{RES}/{tag}.csv"
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d = d[d["method"] == "PACR (ours)"].assign(viol=lambda x: x["FOR"] > e)
        g = d.groupby("pi").agg(viol=("viol", "mean"), WR=("workload_reduction", "mean"), sens=("sensitivity", "mean"))
        rows.append((e, g))
    if not rows:
        return
    pis = rows[0][1].index.values
    head = " & ".join(f"{pct(p,0)}\\%" for p in pis)
    lines = []
    for e, g in rows:
        lines.append(f"$\\varepsilon={pct(e,0)}\\%$ & Violations & " + " & ".join(pct(g.loc[p, 'viol'], 0) for p in pis) + " \\\\")
        lines.append(" & WR & " + " & ".join(pct(g.loc[p, 'WR'], 1) for p in pis) + " \\\\")
        lines.append("\\addlinespace")
    out = ("\\begin{table}[!htbp]\n\\centering\n\\caption{Sensitivity of PACR to the FOR target $\\varepsilon$ under controlled prevalence shift (EyePACS held-out patients, 200 repetitions; violations counted against the respective $\\varepsilon$). Values are percentages.}\n\\label{tab:eps}\n\\footnotesize\n"
           f"\\begin{{tabular}}{{@{{}}ll{'r'*len(pis)}@{{}}}}\n\\toprule\n & & \\multicolumn{{{len(pis)}}}{{c}}{{Deployment prevalence}} \\\\\n\\cmidrule(l){{3-{2+len(pis)}}}\nTarget & Metric & " + head + " \\\\\n\\midrule\n"
           + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(f"{OUT}/tab_eps.tex", "w").write(out)


def tab_ablation():
    meths = ["PACR (ours)", "Ablation: plug-in (no bounds)", "Ablation: EM prevalence", "Ablation: BBSE prevalence", "Ablation: no sensitivity floor"]
    return tab_controlled(pis=(0.02, 0.05, 0.10, 0.20), label="tab:ablation", methods=meths,
                          caption="Ablation of PACR components under controlled prevalence shift (EyePACS held-out patients, 200 repetitions). Columns as in Table~\\ref{tab:controlled}.")


def summary_csvs():
    """Per-method averages over repetitions, small enough to keep in the repository."""
    agg = dict(FOR=("FOR", "mean"), violation_rate=("viol", "mean"), workload_reduction=("workload_reduction", "mean"),
               sensitivity=("sensitivity", "mean"), repetitions=("viol", "size"))
    c = pd.read_csv(f"{RES}/controlled.csv")
    c.assign(viol=c["FOR"] > EPS).groupby(["pi", "method"]).agg(**agg).round(4).to_csv(f"{RES}/summary_controlled.csv")
    r = pd.read_csv(f"{RES}/real.csv")
    r.assign(viol=r["FOR"] > EPS).groupby(["cohort", "deploy", "method"]).agg(**agg).round(4).to_csv(f"{RES}/summary_real.csv")


if __name__ == "__main__":
    tab_data(); tab_base()
    if os.path.exists(f"{RES}/controlled.csv"):
        tab_controlled(); tab_ablation()
    if os.path.exists(f"{RES}/real.csv"):
        tab_real()
        summary_csvs()
    tab_eps()
    print(sorted(os.listdir(OUT)))
