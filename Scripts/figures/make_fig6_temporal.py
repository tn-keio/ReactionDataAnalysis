# -*- coding: utf-8 -*-
"""
Fig. 6 - Long-term temporal robustness (56 months).
(a) Per-item month-to-month decision consistency (Cohen's kappa) across the 14-item
    panel: empirical Mahalanobis vs PCA.
(b) Representative item (CRE): monthly detection recall over 56 months.
Data: Results/20260706_TemporalRobustness_AllItems/.
Outputs gray (main text) and color (supplementary) variants at 600 DPI.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

BASE = os.path.join(REPO, "\Results\20260706_TemporalRobustness_AllItems")
OUT = os.path.dirname(__file__)
PANEL = ["TP", "ALB", "TB", "DB", "CA", "IP", "CRE", "UA", "FE", "CU", "ZN", "GLU", "GA", "CRP"]

BLUE, ORANGE = "#0072B2", "#E69F00"
INK, GRAY, LGRAY = "#222222", "#7F7F7F", "#BDBDBD"

summary = pd.read_csv(os.path.join(BASE, "all_items_summary.csv"))
summary = summary[summary["item"].isin(PANEL)].copy()
summary = summary.sort_values("Empirical_maha_kappa_bord_mean", ascending=False).reset_index(drop=True)
maha_mean = summary["Empirical_maha_kappa_bord_mean"].mean()
pca_mean = summary["PCA_kappa_bord_mean"].mean()

cre = pd.read_csv(os.path.join(BASE, "CRE", "monthly_metrics.csv"))


def make(color: bool):
    if color:
        c_maha, c_pca = BLUE, ORANGE
        ls_pca = "-"
    else:
        c_maha, c_pca = "#1a1a1a", GRAY
        ls_pca = "--"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.3),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---- (a) per-item kappa ----
    x = np.arange(len(summary))
    w = 0.40
    axA.bar(x - w/2, summary["Empirical_maha_kappa_bord_mean"], w,
            color=c_maha, label=f"Empirical Mahalanobis (mean κ {maha_mean:.2f})")
    axA.bar(x + w/2, summary["PCA_kappa_bord_mean"], w,
            color=c_pca, hatch=None if color else "///",
            edgecolor="white" if color else c_pca, label=f"PCA (mean κ {pca_mean:.2f})")
    axA.axhline(maha_mean, color=c_maha, lw=1.2, ls=":")
    axA.axhline(pca_mean, color=c_pca, lw=1.2, ls=":")
    axA.set_xticks(x)
    axA.set_xticklabels(summary["item"], rotation=45, ha="right", fontsize=9)
    axA.set_ylim(0, 1.05)
    axA.set_ylabel("month-to-month decision\nconsistency (Cohen's κ)", fontsize=10.5)
    axA.set_title("(a) Decision consistency across the 14-item panel", fontsize=11, loc="left")
    legA = axA.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white",
                      edgecolor="#999999", framealpha=1.0, borderpad=0.6)
    legA.set_zorder(20)
    legA.get_frame().set_linewidth(0.8)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # ---- (b) CRE monthly subtle (1 sigma) recall time series ----
    m = np.arange(len(cre))
    axB.plot(m, cre["E_recall_k1"], color=c_maha, lw=2.0, label="Empirical Mahalanobis")
    axB.plot(m, cre["P_recall_k1"], color=c_pca, lw=1.8, ls=ls_pca, label="PCA")
    axB.set_ylim(0.0, 1.03)
    tick = [i for i in range(0, len(cre), 12)]
    axB.set_xticks(tick)
    axB.set_xticklabels([cre["month"].iloc[i] for i in tick], fontsize=9)
    axB.set_ylabel("monthly subtle (1σ) recall", fontsize=10.5)
    axB.set_title("(b) Representative item (CRE), 56 months", fontsize=11, loc="left")
    axB.legend(loc="lower left", fontsize=9, frameon=False)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    fig.tight_layout()
    tag = "color" if color else "gray"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig6_temporal_robustness_{tag}.{ext}"),
                    dpi=600, facecolor="white")
    plt.close(fig)
    print(f"wrote fig6_temporal_robustness_{tag}")


make(color=True)
make(color=False)
print(f"panel means: maha={maha_mean:.3f} pca={pca_mean:.3f}")
