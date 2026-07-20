# -*- coding: utf-8 -*-
"""
Graphical Abstract for the CCA manuscript.
Self-contained (no data dependency): schematic curves + confirmed summary numbers.
Landscape ~2.5:1 (Elsevier GA spec). Simplified text, large fonts for legibility.
Output: graphical_abstract.png / .pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

# ---- colour-blind-safe palette (Okabe-Ito) ----
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
VERM   = "#D55E00"
GRAY   = "#7F7F7F"
LGRAY  = "#D9D9D9"
INK    = "#222222"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.linewidth": 1.0,
    "savefig.dpi": 300,
})

FIG_W, FIG_H = 13.28, 5.31
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")

def figtext(x, y, s, **kw):
    kw.setdefault("transform", fig.transFigure)
    kw.setdefault("color", INK)
    return fig.text(x, y, s, **kw)

def rbox(x, y, w, h, fc="white", ec=INK, lw=1.4, rad=0.02, z=1):
    p = FancyBboxPatch((x, y), w, h, transform=fig.transFigure,
                       boxstyle=f"round,pad=0.006,rounding_size={rad}",
                       fc=fc, ec=ec, lw=lw, zorder=z)
    fig.add_artist(p)
    return p

def arrow(x0, x1, y, color=INK):
    a = FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=30, lw=3.0,
                        color=color, zorder=5)
    fig.add_artist(a)

# ================= Title =================
figtext(0.5, 0.945, "A simple statistical distance beats deep learning",
        ha="center", va="center", fontsize=22, fontweight="bold")
figtext(0.5, 0.885,
        "Reaction-curve anomaly detection  ·  22.8 M curves  ·  3 analyzers  ·  4.7 years",
        ha="center", va="center", fontsize=14.5, color=GRAY)

# column headers
figtext(0.165, 0.815, "INPUT",  ha="center", fontsize=16, fontweight="bold", color=BLUE)
figtext(0.500, 0.815, "MODEL",  ha="center", fontsize=16, fontweight="bold", color=BLUE)
figtext(0.820, 0.815, "RESULT", ha="center", fontsize=16, fontweight="bold", color=BLUE)

# ================= Column A: input =================
rbox(0.035, 0.16, 0.26, 0.61, fc="#F7FAFC", ec=LGRAY, lw=1.2)
axA = fig.add_axes([0.070, 0.315, 0.205, 0.37]); axA.set_zorder(6); axA.patch.set_alpha(0)
t = np.linspace(5, 38, 200)
normal = 1.0 - np.exp(-(t - 5) / 6.0)
band = 0.05 + 0.012 * (t - 5) / 33
axA.fill_between(t, normal - 2 * band, normal + 2 * band, color=LGRAY, alpha=0.9, lw=0, label="normal ±2 SD")
anom = normal + 0.17 * np.exp(-((t - 23) / 3.0) ** 2) + 0.05 * np.exp(-((t - 7) / 1.0) ** 2)
axA.plot(t, anom, color=VERM, lw=3.0, label="anomaly")
axA.set_xlim(5, 38); axA.set_ylim(-0.05, 1.28)
axA.set_ylabel("calc = main − sub", fontsize=13)
axA.tick_params(labelsize=11, length=3)
axA.set_xticks([5, 15, 25, 35])
axA.legend(loc="lower right", fontsize=11.5, frameon=False)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
figtext(0.165, 0.225, "Reaction curve — anomalies\ninvisible to routine QC",
        ha="center", va="center", fontsize=13.5, color=INK)

# ================= Column B: model =================
arrow(0.300, 0.350, 0.46, color=BLUE)
rbox(0.355, 0.16, 0.29, 0.61, fc="#F2F7FB", ec=BLUE, lw=1.6)
figtext(0.500, 0.705, "Empirical\nMahalanobis distance",
        ha="center", va="center", fontsize=17, fontweight="bold", color=BLUE)

axB = fig.add_axes([0.400, 0.315, 0.20, 0.30]); axB.set_zorder(6); axB.patch.set_alpha(0); axB.set_axis_off()
axB.set_xlim(-3.2, 3.4); axB.set_ylim(-2.6, 2.6)
rng = np.random.default_rng(3)
cloud = rng.multivariate_normal([0, 0], [[1.0, 0.7], [0.7, 1.0]], 120)
axB.scatter(cloud[:, 0], cloud[:, 1], s=10, color=GRAY, alpha=0.5, lw=0)
for k, a in ((2, 0.9), (3.2, 0.5)):
    axB.add_patch(Ellipse((0, 0), 2*k*1.05, 2*k*0.55, angle=45,
                          fill=False, ec=BLUE, lw=1.4, ls="--", alpha=a))
axB.scatter([2.6], [-1.9], marker="X", s=150, color=VERM, zorder=6)
axB.annotate("", xy=(2.55, -1.85), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color=VERM, lw=2.2))
figtext(0.500, 0.225, "500 recent samples  ·  no GPU",
        ha="center", va="center", fontsize=14, fontweight="bold", color=INK)

# ================= Column C: result =================
arrow(0.650, 0.700, 0.46, color=BLUE)
rbox(0.705, 0.16, 0.26, 0.61, fc="#F7FAFC", ec=LGRAY, lw=1.2)

axC = fig.add_axes([0.740, 0.535, 0.205, 0.17]); axC.set_zorder(6); axC.patch.set_alpha(0)
labels = ["Maha-\nlanobis", "Deep\nlearning"]
vals = [0.99, 0.51]
bars = axC.bar([0, 1], vals, color=[BLUE, ORANGE], width=0.6)
axC.set_ylim(0, 1.15); axC.set_xticks([0, 1])
axC.set_xticklabels(labels, fontsize=12.5)
axC.set_yticks([0, 0.5, 1.0])
axC.tick_params(labelsize=11, length=3)
axC.set_ylabel("recall", fontsize=13)
for b, v in zip(bars, vals):
    axC.text(b.get_x() + b.get_width()/2, v + 0.04, f"{int(round(v*100))}%",
             ha="center", fontsize=15, fontweight="bold",
             color=BLUE if v == 0.99 else INK)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)

badges = [
    "Detects subtle anomalies\ndeep learning misses",
    "~1000× less compute",
    "Confirmed on a real anomaly",
]
by = 0.40
for b in badges:
    figtext(0.720, by, "✔", fontsize=15, color=GREEN, fontweight="bold", va="center")
    figtext(0.742, by, b, fontsize=13.5, va="center", color=INK)
    by -= 0.085

# ================= Bottom band =================
rbox(0.035, 0.02, 0.93, 0.085, fc=BLUE, ec=BLUE, rad=0.02)
figtext(0.5, 0.062,
        "A low-cost, interpretable tool for continuous laboratory quality assurance",
        ha="center", va="center", fontsize=15.5, color="white", fontweight="bold")

out_png = os.path.join(os.path.dirname(__file__), "graphical_abstract.png")
out_pdf = out_png.replace(".png", ".pdf")
fig.savefig(out_png, dpi=300, facecolor="white")
fig.savefig(out_pdf, facecolor="white")
print("wrote:", out_png)
print("wrote:", out_pdf)
