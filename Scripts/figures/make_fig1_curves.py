# -*- coding: utf-8 -*-
"""
Fig. 1 - Reaction-curve morphology, the calc channel, and the synthetic anomaly battery.
(a) A clean endpoint reaction curve: main (RAM) and sub (RAS) wavelength absorbance and
    their difference (calc = main - sub) over the analysis window (points 5-38).
(b) Small multiples of the five synthetic anomaly types (A-E) plus the benign_offset
    negative control, each shown against the clean +/-2 SD band, derived from the SAME
    base curve. Example severity 5 sigma; the battery spans 1-8 sigma.
Data: Data/BigData/synthetic_anomalies/endpoint_synthetic_anomalies_p5_38.parquet
Outputs gray (main text) and color (supplementary) variants at 600 DPI.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
REPO = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

PARQUET = os.path.join(REPO, 'Data', 'BigData', 'synthetic_anomalies',
                       'endpoint_synthetic_anomalies_p5_38.parquet')
OUT = os.path.dirname(__file__)
ITEM = "CRE"
SEV = 5.0
PTS = list(range(5, 39))
RAM = [f"RAM{p}" for p in PTS]
RAS = [f"RAS{p}" for p in PTS]

BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
INK, GRAY, LGRAY = "#222222", "#7F7F7F", "#D9D9D9"

TYPES = [
    ("A_spike",      "A  Spike"),
    ("B_drift",      "B  Drift"),
    ("C_hump",       "C  Hump"),
    ("D_kinetic",    "D  Kinetic lag"),
    ("E_baseline",   "E  Baseline shift"),
    ("benign_offset","Benign offset (control)"),
]

df = pd.read_parquet(PARQUET)
d = df[df["item"] == ITEM].copy()

def calc_of(row):
    return row[RAM].to_numpy(float) - row[RAS].to_numpy(float)

clean = d[d["anomaly_type"] == "clean"]
clean_calc = clean[RAM].to_numpy(float) - clean[RAS].to_numpy(float)
# per-point min-max normalization from the clean pool (as the detector uses)
cmin = clean_calc.min(0)
cmax = clean_calc.max(0)
rng = np.where((cmax - cmin) > 1e-9, cmax - cmin, 1.0)
def norm(x):
    return (np.asarray(x, float) - cmin) / rng
clean_norm = norm(clean_calc)
mean = clean_norm.mean(0)
sd = clean_norm.std(0)

# pick a base_id present for all types (sev 5, benign uses nearest available)
def rows_for(t, sev):
    sub = d[(d["anomaly_type"] == t)]
    if t == "benign_offset":
        svals = sorted(sub["severity"].unique())
        sev = min(svals, key=lambda s: abs(s - sev))
    return sub[sub["severity"] == sev]

# severities shown per type (D_kinetic is a lag in points, not k×σ̄)
SEV_BY_TYPE = {
    "A_spike": [2.0, 5.0, 8.0], "B_drift": [2.0, 5.0, 8.0], "C_hump": [2.0, 5.0, 8.0],
    "E_baseline": [2.0, 5.0, 8.0], "benign_offset": [2.0, 5.0, 8.0], "D_kinetic": [1.0, 3.0, 5.0],
}

sets = {t: set(rows_for(t, SEV)["base_id"]) for t, _ in TYPES}
common = set(clean["base_id"])
for t, _ in TYPES:
    common &= sets[t]
base_id = sorted(common)[0] if common else None

def one_example(t, sev):
    r = rows_for(t, sev)
    rr = r[r["base_id"] == base_id] if base_id is not None else r.iloc[0:0]
    row = rr.iloc[0] if len(rr) else r.iloc[0]
    return calc_of(row)

# examples[t] = list of calc curves, one per severity (low -> high)
examples = {t: [one_example(t, s) for s in SEV_BY_TYPE[t]] for t, _ in TYPES}

# clean base curve for panel (a)
if base_id is not None and (clean["base_id"] == base_id).any():
    crow = clean[clean["base_id"] == base_id].iloc[0]
else:
    crow = clean.iloc[0]
main_a = crow[RAM].to_numpy(float)
sub_a = crow[RAS].to_numpy(float)
calc_a = main_a - sub_a


def make(color: bool):
    if color:
        cmain, csub, ccalc = BLUE, ORANGE, INK
        tcol = {"A_spike": VERM, "B_drift": BLUE, "C_hump": GREEN,
                "D_kinetic": PURPLE, "E_baseline": ORANGE, "benign_offset": GRAY}
    else:
        cmain, csub, ccalc = "#555555", "#AAAAAA", "#000000"
        tcol = {t: "#000000" for t, _ in TYPES}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig = plt.figure(figsize=(12.4, 5.0))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.5, 1, 1, 1], hspace=0.45, wspace=0.32)

    # ---- (a) calc construction ----
    axa = fig.add_subplot(gs[:, 0])
    axa.plot(PTS, main_a, color=cmain, lw=2.0, label="main (RAM)")
    axa.plot(PTS, sub_a, color=csub, lw=2.0, label="sub (RAS)")
    axa.plot(PTS, calc_a, color=ccalc, lw=2.4, ls="--", label="calc = main − sub")
    axa.set_title("(a) Dual-wavelength calc channel", fontsize=11.5, loc="left")
    axa.set_xlabel("reaction point")
    axa.set_ylabel("absorbance (a.u.)")
    axa.legend(loc="center right", fontsize=9.5, frameon=True, facecolor="white",
               edgecolor="#999999", framealpha=1.0).set_zorder(10)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)

    # ---- (b) small multiples: clean base vs injected anomaly at increasing severity ----
    sev_cols = ["#4C9EDB", "#E69F00", "#D55E00"] if color else ["0.62", "0.40", "0.0"]
    sev_lws = [1.3, 1.9, 2.5]
    positions = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
    for (t, title), (r, c) in zip(TYPES, positions):
        ax = fig.add_subplot(gs[r, c])
        ax.plot(PTS, calc_a, color="0.55", lw=1.4, ls="--", label="clean base")
        allc = [calc_a]
        for k, sev in enumerate(SEV_BY_TYPE[t]):
            lab = f"{sev:.0f} pts lag" if t == "D_kinetic" else f"{sev:.0f}σ"
            ax.plot(PTS, examples[t][k], color=sev_cols[k], lw=sev_lws[k], label=lab)
            allc.append(examples[t][k])
        ax.set_title(title, fontsize=10, loc="left", color=tcol[t] if color else INK)
        both = np.concatenate(allc)
        lo, hi = float(both.min()), float(both.max())
        pad = 0.15 * (hi - lo) + 1.0
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(labelsize=8)
        if t == "benign_offset":
            ax.text(0.96, 0.12, "ΔCalc ≈ 0\n(all severities)", transform=ax.transAxes,
                    ha="right", fontsize=8, color=INK, style="italic")
        if t == "A_spike":
            ax.legend(loc="upper left", fontsize=7, frameon=True, facecolor="white",
                      edgecolor="#bbbbbb", framealpha=0.95)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if c == 1:
            ax.set_ylabel("calc (a.u.)", fontsize=9)
        if r != 1:
            ax.set_xticklabels([])
    fig.text(0.665, 0.015, "reaction point (5–38)", ha="center", fontsize=10)
    fig.text(0.665, 0.965,
             "(b) Synthetic anomaly battery vs clean base, at increasing severity "
             "(2σ / 5σ / 8σ; D: 1 / 3 / 5-point lag)",
             ha="center", fontsize=11)

    tag = "color" if color else "gray"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig1_curves_anomalies_{tag}.{ext}"),
                    dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig1_curves_anomalies_{tag}  (item={ITEM}, base_id={base_id})")


make(color=True)
make(color=False)
