"""Figure 2: overview of PACR."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paths import FIGURES

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WASH = {"blue": "#e6f0fb", "orange": "#fdeee7", "aqua": "#e3f6ef", "gray": "#f0efec"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")


def box(x, y, w, h, title, body, edge, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=1.2, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.6, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=6.7, color=INK2, linespacing=1.4)


def arrow(x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.2, color=color))

box(1, 31, 24, 16, "Deployment sample", "unlabelled images\nfrom the new population\n(scores $s_j$ only)", INK2, WASH["gray"])
box(1, 6, 24, 16, "Source calibration set", "labelled images\n(EyePACS held-out patients)\nscores $s_i$, grades $y_i$", INK2, WASH["gray"])
box(31, 31, 24, 16, "Step 1: prevalence", "moment-matching (PACC)\nestimate $\\hat\\pi$, bootstrap\nupper bound $\\pi_U$", ORANGE, WASH["orange"])
box(31, 6, 24, 16, "Step 2: release rates", "Clopper\u2013Pearson bounds\n$\\overline{F}_1(t)$, $\\underline{F}_0(t)$\non a fixed grid of $t$", AQUA, WASH["aqua"])
box(61, 18.5, 22, 16, "Step 3: certify $\\hat t$", "largest $t \\leq t_\\beta$ with\n$\\overline{\\mathrm{FOR}}(t) \\leq \\varepsilon$\n(sensitivity floor $t_\\beta$)", BLUE, WASH["blue"])
ax.text(93, 33.5, "Release\n$s < \\hat t$", ha="center", va="center", fontsize=8.4, color=INK, fontweight="bold")
ax.text(93, 27.6, "no human grading", ha="center", va="center", fontsize=7.2, color=INK2)
ax.text(93, 18.5, "Refer\n$s \\geq \\hat t$", ha="center", va="center", fontsize=8.4, color=INK, fontweight="bold")
ax.text(93, 12.6, "sent to a grader", ha="center", va="center", fontsize=7.2, color=INK2)

arrow(25.8, 39, 30.2, 39)
arrow(25.8, 14, 30.2, 14)
arrow(55.8, 39, 60.4, 30.5)
arrow(55.8, 14, 60.4, 22.5)
arrow(83.8, 29, 87.2, 32.5, BLUE)
arrow(83.8, 24, 87.2, 20, BLUE)
ax.text(50, 1.2, "Guarantee (Proposition 1): under label shift, FOR$(\\hat t) \\leq \\varepsilon$ with probability $\\geq 1-\\delta$",
        ha="center", va="bottom", fontsize=7.6, color=INK2, style="italic")

FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "fig2_method.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig2_method.png", dpi=300, bbox_inches="tight")
