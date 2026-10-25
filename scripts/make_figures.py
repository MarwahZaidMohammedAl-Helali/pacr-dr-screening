"""Figures 1 and 3-7 of the paper, made from the CSV files in results/.

    python scripts/make_figures.py          # all figures
    python scripts/make_figures.py 3 5      # only some
"""
import io
import os
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from paths import DATA, FIGURES, RESULTS
import pacr
from pacr.fundus import crop_and_pad

RES, FIG = str(RESULTS), str(FIGURES)
EPS, DELTA = 0.02, 0.10

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
STYLE = {  # method -> (color, marker, linestyle, label)
    "PACR (ours)": ("#2a78d6", "o", "-", "PACR (ours)"),
    "PACR, oracle prevalence": ("#2a78d6", None, (0, (3, 2)), "PACR, oracle prevalence"),
    "Source FOR control": ("#eb6834", "s", "-", "Source FOR control"),
    "Class-conditional conformal": ("#1baf7a", "^", "-", "Class-conditional conformal"),
    "Split conformal": ("#eda100", "D", "-", "Split conformal"),
    "Label-shift weighted conformal": ("#e87ba4", "v", "-", "Label-shift weighted conformal"),
    "Fixed operating point": ("#6250d6", "P", "-", "Fixed operating point"),
}
MAIN = list(STYLE.keys())
COH = ["EyePACS", "APTOS", "DDR", "IDRiD"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "lines.markersize": 4.2, "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def panel_label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=INK, va="bottom")


def method_legend(fig, methods, ncol=4, y=1.0):
    handles = [Line2D([0], [0], color=STYLE[m][0], marker=STYLE[m][1], ls=STYLE[m][2], lw=1.6,
                      markeredgecolor="white", markeredgewidth=0.6, label=STYLE[m][3]) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y),
               handlelength=2.6, columnspacing=1.4)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# Figure 1: motivation
def fig1():
    sc = pd.read_csv(f"{RES}/scores_eyepacs.csv")
    pool = sc[sc["part"] == "pool"]
    s, y = pool["s"].values, pool["y"].values
    base = pd.read_csv(f"{RES}/base_performance.csv")
    pis = np.linspace(0.005, 0.7, 300)
    t_ccp = pacr.class_conditional_threshold(s, y, 0.10)
    t_src = pacr.pacr_threshold(s, y, EPS, DELTA / 2, y.mean(), t_max=t_ccp)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for t, m in [(t_ccp, "Class-conditional conformal"), (t_src, "Source FOR control")]:
        f1, f0 = (s[y == 1] < t).mean(), (s[y == 0] < t).mean()
        fr = pis * f1 / (pis * f1 + (1 - pis) * f0)
        axes[0].plot(pis, fr * 100, color=STYLE[m][0], lw=1.8, label=m)
        axes[1].plot(pis, (pis * f1 + (1 - pis) * f0) * 100, color=STYLE[m][0], lw=1.8)
    # prevalence-adaptive: oracle PACR threshold at each prevalence
    wr_or, for_or = [], []
    for p in pis:
        t = pacr.pacr_threshold(s, y, EPS, DELTA / 2, p, t_max=t_ccp)
        f1, f0 = (s[y == 1] < t).mean(), (s[y == 0] < t).mean()
        wr_or.append((p * f1 + (1 - p) * f0) * 100); for_or.append(p * f1 / max(p * f1 + (1 - p) * f0, 1e-12) * 100)
    m = "PACR, oracle prevalence"
    axes[0].plot(pis, for_or, color=STYLE[m][0], ls=STYLE[m][2], lw=1.8, label="Prevalence-adaptive threshold")
    axes[1].plot(pis, wr_or, color=STYLE[m][0], ls=STYLE[m][2], lw=1.8)
    axes[0].axhline(EPS * 100, color=INK2, lw=0.9)
    axes[0].text(0.69, EPS * 100 + 0.25, "target FOR = 2%", ha="right", va="bottom", fontsize=7, color=INK2)
    for ax in axes:
        for _, r in base.iterrows():
            ax.axvline(r["prevalence"], color=MUTED, lw=0.7, ls=(0, (1, 2)))
        ax.set_xlim(0, 0.7)
        ax.set_xlabel("Prevalence of referable DR in the screened population")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ymax = axes[0].get_ylim()[1]
    for _, r in base.iterrows():
        axes[0].text(r["prevalence"] + 0.006, ymax * 0.97, r["cohort"], rotation=90, va="top", ha="left", fontsize=6.6, color=INK2)
    axes[0].set_ylabel("FOR among released images (%)")
    axes[1].set_ylabel("Workload reduction (%)")
    axes[0].set_ylim(0, ymax)
    axes[1].set_ylim(0, 100)
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    h = [Line2D([0], [0], color=STYLE["Class-conditional conformal"][0], lw=1.8, label="Fixed threshold: class-conditional conformal (90% sensitivity)"),
         Line2D([0], [0], color=STYLE["Source FOR control"][0], lw=1.8, label="Fixed threshold: FOR control at the source prevalence (certifies no release, WR = 0)"),
         Line2D([0], [0], color=STYLE[m][0], ls=STYLE[m][2], lw=1.8, label="Threshold adapted to the true prevalence")]
    fig.legend(handles=h, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    save(fig, "fig1_motivation")


# Figure 3: controlled shift
def fig3(tag="controlled", name="fig3_controlled"):
    res = pd.read_csv(f"{RES}/{tag}.csv")
    res["viol"] = res["FOR"] > EPS
    g = res.groupby(["method", "pi"])
    summ = g.agg(FOR=("FOR", "mean"), FOR_lo=("FOR", lambda v: np.percentile(v, 5)), FOR_hi=("FOR", lambda v: np.percentile(v, 95)),
                 viol=("viol", "mean"), n=("viol", "size"), WR=("workload_reduction", "mean"), sens=("sensitivity", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for m in MAIN:
        d = summ[summ["method"] == m].sort_values("pi")
        if d.empty:
            continue
        col, mk, ls, _ = STYLE[m]
        kw = dict(color=col, marker=mk, ls=ls, markeredgecolor="white", markeredgewidth=0.5, zorder=3 if "ours" in m else 2,
                  lw=2.0 if "ours" in m else 1.4)
        axes[0].plot(d["pi"], d["FOR"] * 100, **kw)
        axes[1].plot(d["pi"], d["viol"] * 100, **kw)
        axes[2].plot(d["pi"], d["WR"] * 100, **kw)
    axes[0].axhline(EPS * 100, color=INK2, lw=0.9); axes[0].text(0.5, EPS * 100 + 0.4, "ε = 2%", fontsize=6.8, color=INK2, ha="right", va="bottom")
    axes[1].axhline(DELTA * 100, color=INK2, lw=0.9); axes[1].text(0.5, DELTA * 100 + 2, "δ = 10%", fontsize=6.8, color=INK2, ha="right")
    axes[0].set_ylabel("Mean FOR among released (%)")
    axes[1].set_ylabel("Runs with FOR > ε (%)")
    axes[2].set_ylabel("Workload reduction (%)")
    axes[1].set_ylim(-3, 103); axes[2].set_ylim(0, 100)
    for i, ax in enumerate(axes):
        ax.set_xlabel("Deployment prevalence")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        panel_label(ax, "(%s)" % "abc"[i])
    fig.tight_layout(w_pad=1.6)
    method_legend(fig, MAIN, ncol=4, y=0.99)
    save(fig, name)
    return summ


# Figure 4: prevalence estimation
EST = {"pi_hat": ("#2a78d6", "o", "PACC, used by PACR"), "pi_em": ("#6250d6", "D", "EM (maximum likelihood)"),
       "pi_bbse": ("#eb6834", "s", "BBSE")}


def q(p):
    return lambda v: np.percentile(v, p)


def fig4():
    ctl = pd.read_csv(f"{RES}/controlled.csv")
    ctl = ctl[ctl["method"] == "PACR (ours)"]
    real = pd.read_csv(f"{RES}/real.csv")
    real = real[(real["method"] == "PACR (ours)") & (real["deploy"] == "native")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 0.95]))
    ax = axes[0]
    g = ctl.groupby("pi").agg(**{k: (k, "mean") for k in EST}, lo=("pi_hat", q(5)), hi=("pi_hat", q(95)), ub=("pi_ub", "mean")).reset_index()
    lim = g["pi"].max() + 0.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.fill_between(g["pi"], g["lo"], g["hi"], color=EST["pi_hat"][0], alpha=0.13, lw=0)
    for k, (c, mk, lab) in EST.items():
        ax.plot(g["pi"], g[k], color=c, marker=mk, markeredgecolor="white", markeredgewidth=0.5, lw=1.8 if k == "pi_hat" else 1.3,
                label={"pi_hat": "PACC (mean, 5–95%)", "pi_em": "EM", "pi_bbse": "BBSE"}[k], zorder=3 if k == "pi_hat" else 2)
    ax.plot(g["pi"], g["ub"], color=EST["pi_hat"][0], ls=(0, (3, 2)), lw=1.4, label="PACC bound $\\pi_U$ (mean)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("True prevalence (controlled shift, EyePACS)"); ax.set_ylabel("Estimated prevalence")
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, loc="upper left", fontsize=6.6)
    panel_label(ax, "(a)")
    ax = axes[1]
    rg = real.groupby("cohort").agg(true=("pi_true", "mean"), **{k: (k, "mean") for k in EST},
                                    lo=("pi_hat", q(5)), hi=("pi_hat", q(95)), ub=("pi_ub", "mean")).reindex(COH)
    x = np.arange(len(COH))
    ax.scatter(x, rg["true"], color=INK, marker="_", s=200, lw=2.2, label="True prevalence", zorder=4)
    for i, (k, (c, mk, lab)) in enumerate(EST.items()):
        xx = x + (i - 1) * 0.2 + 0.0
        if k == "pi_hat":
            ax.errorbar(xx, rg[k], yerr=[rg[k] - rg["lo"], rg["hi"] - rg[k]], fmt=mk, color=c, markeredgecolor="white",
                        capsize=0, lw=1.2, ms=5, label=lab + " (5–95%)", zorder=3)
            ax.scatter(xx, rg["ub"], marker="v", facecolor="white", edgecolor=c, linewidth=1.0, s=24, zorder=3, label="PACC upper bound $\\pi_U$ (mean)")
        else:
            ax.scatter(xx, rg[k], color=c, marker=mk, edgecolor="white", linewidth=0.5, s=26, label=lab, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(COH)
    ax.set_xlim(-0.6, len(COH) - 0.4)
    ax.set_ylabel("Prevalence of referable DR (native)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.06)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.6, ncol=1, handletextpad=0.4, labelspacing=0.9)
    panel_label(ax, "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_prevalence")
    return g, rg


# Figure 5: real cohorts
LOCAL = "PACR + local prevalence (m=200)"
STYLE[LOCAL] = ("#0b0b0b", "*", "-", "PACR + 200 local labels")


def fig5(deploy="5%", name="fig5_real_cohorts"):
    real = pd.read_csv(f"{RES}/real.csv")
    real = real[real["deploy"] == deploy].copy()
    real["viol"] = real["FOR"] > EPS
    summ = real.groupby(["cohort", "method"]).agg(FOR=("FOR", "mean"), FOR_lo=("FOR", q(5)), FOR_hi=("FOR", q(95)), viol=("viol", "mean"),
                                                   WR=("workload_reduction", "mean"), WR_lo=("workload_reduction", q(5)),
                                                   WR_hi=("workload_reduction", q(95)), sens=("sensitivity", "mean")).reset_index()
    meths = [m for m in MAIN + [LOCAL] if m in set(summ["method"])]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.9), sharey=True)
    y = np.arange(len(meths))[::-1]
    fmax = max(4.0, float(np.nanmax(summ["FOR_hi"])) * 100 * 1.08)
    for j, co in enumerate(COH):
        d = summ[summ["cohort"] == co].set_index("method").reindex(meths)
        for row, (col, lo, hi, xlab) in enumerate([("FOR", "FOR_lo", "FOR_hi", "FOR (%)"), ("WR", "WR_lo", "WR_hi", "WR (%)")]):
            ax = axes[row, j]
            for k, m in enumerate(meths):
                c, mk = STYLE[m][0], STYLE[m][1] or "o"
                ax.plot([d.loc[m, lo] * 100, d.loc[m, hi] * 100], [y[k], y[k]], color=c, lw=1.4, solid_capstyle="round")
                ax.plot(d.loc[m, col] * 100, y[k], marker=mk, color=c, markeredgecolor=c if "oracle" in m else "white",
                        markeredgewidth=0.9 if "oracle" in m else 0.6, ms=6.5 if mk == "*" else 5.2, mfc="white" if "oracle" in m else c, ls="none")
                if row == 0:
                    v = d.loc[m, "viol"] * 100
                    ax.text(0.985, y[k], f"{v:.0f}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.2,
                            color=INK if v > DELTA * 100 else MUTED, fontweight="bold" if v > DELTA * 100 else "normal")
            if row == 0:
                ax.axvline(EPS * 100, color=INK2, lw=0.9)
                ax.set_xlim(-0.2, fmax * 1.3); ax.set_xticks(np.arange(0, fmax + 0.01, 1 if fmax <= 5 else 2))
                ax.set_title(co, color=INK, fontweight="bold")
            else:
                ax.set_xlim(-2, 102)
            ax.set_xlabel(xlab)
            ax.grid(axis="y", visible=False)
    for r_ in range(2):
        axes[r_, 0].set_yticks(y); axes[r_, 0].set_yticklabels([STYLE[m][3] for m in meths], fontsize=6.8)
    panel_label(axes[0, 0], "(a)"); panel_label(axes[1, 0], "(b)")
    fig.tight_layout(h_pad=1.4, w_pad=0.5)
    save(fig, name)
    return summ


# Figure 6: local prevalence and calibration size
CCOL = {"EyePACS": "#2a78d6", "APTOS": "#eb6834", "DDR": "#1baf7a", "IDRiD": "#6250d6"}
CMK = {"EyePACS": "o", "APTOS": "s", "DDR": "^", "IDRiD": "D"}


def fig6(deploy="5%"):
    real = pd.read_csv(f"{RES}/real.csv")
    real = real[real["deploy"] == deploy].copy()
    real["viol"] = real["FOR"] > EPS
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    nviol = 0
    xs = [0, 100, 200, 400]
    for co in COH:
        d = real[real["cohort"] == co]
        vs, ws = [], []
        for m in xs:
            lab = "PACR (ours)" if m == 0 else f"PACR + local prevalence (m={m})"
            dd = d[d["method"] == lab]
            vs.append(dd["viol"].mean() * 100); ws.append(dd["workload_reduction"].mean() * 100)
        nviol += int(np.nansum(vs))
        for ax, v in ((axes[0], ws),):
            ax.plot(xs[1:], v[1:], color=CCOL[co], marker=CMK[co], markeredgecolor="white", markeredgewidth=0.5, label=co)
            ax.plot([0], [v[0]], color=CCOL[co], marker=CMK[co], mfc="white", markeredgewidth=1.0, ls="none")
            ax.plot([0, 100], [v[0], v[1]], color=CCOL[co], lw=0.8, ls=(0, (1.5, 1.5)))
    axes[0].set_ylabel("Workload reduction (%)")
    axes[0].text(0.03, 0.97, "5% prevalence deployment\n" + ("no run exceeded ε at any $m_{loc}$" if nviol == 0 else ""),
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=6.6, color=INK2)
    for ax in axes[:1]:
        ax.set_xlabel("Labelled deployment images $m_{loc}$")
        ax.set_xticks(xs); ax.set_xticklabels(["0\n(PACC)", "100", "200", "400"])
    axes[0].set_ylim(bottom=-1)
    axes[0].legend(frameon=False, loc="lower right", fontsize=6.6)
    ax = axes[1]
    rows = []
    for n in [250, 500, 1000, 2000]:
        p = f"{RES}/ncal_{n}.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d["method"] == "PACR (ours)"]
        for pi, dd in d.groupby("pi"):
            rows.append(dict(n=n, pi=pi, WR=dd["workload_reduction"].mean() * 100, viol=(dd["FOR"] > EPS).mean() * 100))
    if rows:
        r = pd.DataFrame(rows)
        pc = {0.02: ("#2a78d6", "o"), 0.05: ("#eb6834", "s"), 0.1: ("#1baf7a", "^")}
        for pi, dd in r.groupby("pi"):
            c, mk = pc.get(round(pi, 2), (MUTED, "o"))
            ax.plot(dd["n"], dd["WR"], color=c, marker=mk, markeredgecolor="white", markeredgewidth=0.5, label=f"prevalence {pi*100:.0f}%")
        ax.set_xscale("log"); ax.set_xticks([250, 500, 1000, 2000]); ax.set_xticklabels(["250", "500", "1000", "2000"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Source calibration images"); ax.set_ylabel("Workload reduction (%)")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=6.6)
    for i, ax in enumerate(axes):
        panel_label(ax, "(%s)" % "abc"[i])
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig6_local_and_ncal")
    return pd.DataFrame(rows) if rows else None


# Figure 7: qualitative examples
def fig7():
    from PIL import Image
    real = pd.read_csv(f"{RES}/real.csv")
    t = real[(real["cohort"] == "EyePACS") & (real["method"] == "PACR (ours)") & (real["deploy"] == "5%")]["threshold"].median()
    sc = pd.read_csv(f"{RES}/scores_eyepacs.csv")
    sc = sc[sc["part"] == "pool"]
    zf = zipfile.ZipFile(DATA / "eyepacs/EYEPACS.zip")
    names = {os.path.splitext(os.path.basename(n))[0]: n for n in zf.namelist() if n.lower().endswith((".jpeg", ".jpg", ".png"))}
    rng = np.random.default_rng(3)
    groups = [("Released, grade 0", sc[(sc.s < t) & (sc.grade == 0)]),
              ("Released, referable (false omission)", sc[(sc.s < t) & (sc.y == 1)]),
              ("Referred, grade 0", sc[(sc.s >= t) & (sc.grade == 0)]),
              ("Referred, referable", sc[(sc.s >= t) & (sc.y == 1)])]
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 6.7))
    for i, (title, g) in enumerate(groups):
        pick = g.iloc[rng.choice(len(g), min(4, len(g)), replace=False)] if len(g) else g
        for j in range(4):
            ax = axes[i, j]; ax.axis("off")
            if j < len(pick):
                r = pick.iloc[j]
                img = crop_and_pad(Image.open(io.BytesIO(zf.read(names[r["image_id"]]))))
                ax.imshow(img)
                ax.text(0.5, -0.04, f"grade {int(r['grade'])} · score {r['s']:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color=INK2)
        axes[i, 0].text(-0.08, 0.5, title.replace(", ", ",\n") + f"\n(n = {len(g):,})", transform=axes[i, 0].transAxes, ha="right", va="center", fontsize=7.4, color=INK, fontweight="bold")
    fig.suptitle(f"EyePACS held-out images at the median PACR threshold for a 5% prevalence deployment (t = {t:.3f})", fontsize=8.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0.13, 0, 1, 0.98))
    save(fig, "fig7_examples")
    return t, {k: len(g) for k, g in groups}


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    figures = {"1": fig1, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7}
    for key in sys.argv[1:] or figures:
        figures[key]()
