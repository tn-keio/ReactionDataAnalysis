"""Method-agnostic Supplementary figure: point-wise variability settles from ~point 5.

Justifies the reaction-point window (using photometric points 5-38) WITHOUT any anomaly-detection
model, using only descriptive statistics of the curves. Answers: "if we must choose the usable data
points not knowing which detector will be used, where does each item's variability settle?"

Metric: within-curve local roughness |2nd diff| (a model-free measure of point-to-point instability =
air-bubble jitter), summarized as the 99th percentile across samples and expressed RELATIVE to each
item's own settled floor (median over points 12-19). value = 1 -> as stable as the settled baseline;
value >> 1 -> unstable (bubble). Roughness ignores smooth real signal (2nd diff removes level & slope),
so it isolates instability from legitimate reaction kinetics.

  Panel A -- heatmap (21 items x points 2-14): the bubble hot-zone is confined to points 2-4 and cools
             to the settled floor from ~point 5, for every item.
  Panel B -- per-item recommended start (= 1 + last point exceeding 2x the settled floor): 20/21 items
             <= 5. The sole exception, LDL-C, is a genuine early kinetic feature (a reproducible
             absorbance hump), not instability.

Inputs (bundled, frozen): Data/reaction_point_window/{pointwise_variability.csv,
recommended_start_by_item.csv}. These are produced by the full analysis (run_pointwise_variability.py)
over the internal dataset; they are shipped here so this figure is reproducible standalone.
Recommended-start robustness columns (p99 / p99.5 / p99.9) live in recommended_start_by_item.csv;
p99.5 matches the P99.5 detection operating point (the conclusion is invariant p99 -> p99.5).

Outputs gray (main-text) and color (Supplemental) versions to Figures/.
Requires: numpy, pandas, matplotlib.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))                 # Export_For_GitHub_CCA/
RES = os.path.join(REPO, 'Data', 'reaction_point_window')       # bundled frozen input CSVs
FIG = os.path.join(REPO, 'Figures')

COLS = list(range(2, 15))            # heatmap points (|2nd diff| undefined at point 1)
START_CHOSEN = 5


def compute():
    # recommended_start (p99 primary + p99.5/p99.9 robustness) is the single source of truth;
    # this figure just reads it.
    p = pd.read_csv(os.path.join(RES, 'pointwise_variability.csv'))
    sdf = pd.read_csv(os.path.join(RES, 'recommended_start_by_item.csv'))
    mats = {}
    for it, g in p.groupby('item'):
        r = g.set_index('point')['rough_p99']           # jitter p99 (% endpoint IQR), pts 2..37
        mats[it] = r / np.median(r.loc[12:19])          # relative to settled floor (heatmap value)
    order = sdf.sort_values(['recommended_start', 'item'])['item'].tolist()
    M = np.vstack([[mats[it].loc[c] for c in COLS] for it in order])
    return order, M, sdf


def draw(order, M, sdf, cmap, path, color):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={'width_ratios': [1.35, 1]})

    # ---- Panel A: heatmap of relative roughness ----
    im = axA.imshow(M, aspect='auto', cmap=cmap, norm=LogNorm(vmin=1, vmax=100),
                    extent=[COLS[0] - 0.5, COLS[-1] + 0.5, len(order) - 0.5, -0.5])
    axA.set_xticks(COLS); axA.set_yticks(range(len(order))); axA.set_yticklabels(order, fontsize=8)
    axA.axvline(START_CHOSEN - 0.5, color=('#00ff88' if color else 'w'), lw=2.2)
    axA.set_xlabel('photometric point'); axA.set_title('(A) Point-wise instability (roughness relative to settled floor)', pad=10)
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.02)
    cb.set_label('|2nd diff| p99  /  item settled floor   (x)')

    # ---- Panel B: recommended start per item ----
    s = sdf.sort_values('recommended_start')
    y = np.arange(len(s))
    hot = s['recommended_start'] > START_CHOSEN
    base_c = '0.3' if not color else '#1f77b4'
    hi_c = '0.0' if not color else '#d62728'
    axB.hlines(y, 0, s['recommended_start'], color=[hi_c if h else base_c for h in hot], lw=1.2)
    axB.scatter(s['recommended_start'], y, s=42, color=[hi_c if h else base_c for h in hot],
                zorder=3, marker='o')
    axB.axvline(START_CHOSEN, color=('#0a7' if color else '0.0'), ls='--', lw=1.4)
    axB.set_yticks(y); axB.set_yticklabels(s['item'], fontsize=8)
    axB.set_xlim(0, max(9, s['recommended_start'].max() + 1)); axB.set_xlabel('recommended start point  (= 1 + last unstable point)')
    axB.set_title('(B) Where each item settles (model-free)')
    axB.grid(axis='x', alpha=0.3); axB.invert_yaxis()
    axB.text(START_CHOSEN + 0.12, 0.35, 'chosen\nstart 5', color=('#0a7' if color else '0.0'),
             fontsize=8, ha='left', va='top')
    ldl = s[s['item'] == 'LDL-C']
    if len(ldl):
        yi = list(s.sort_values('recommended_start')['item']).index('LDL-C')
        axB.annotate('LDL-C: real early kinetic\nhump (signal, not noise)',
                     xy=(ldl['recommended_start'].iloc[0], yi), xytext=(2.2, yi - 1.4),
                     fontsize=7, color=hi_c, va='center', ha='left',
                     arrowprops=dict(arrowstyle='->', color=hi_c, lw=0.8))

    fig.suptitle('Model-free (detector-independent) view: bubble instability is confined to points 1-4 and '
                 'settles by ~point 5 for 20/21 items (LDL-C: real early kinetics)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=600); plt.close(fig)
    print('->', path)


def main():
    os.makedirs(FIG, exist_ok=True)
    order, M, sdf = compute()
    draw(order, M, sdf, 'Greys', os.path.join(FIG, 'pointwise_variability_gray.png'), color=False)
    draw(order, M, sdf, 'magma_r', os.path.join(FIG, 'pointwise_variability_color.png'), color=True)


if __name__ == '__main__':
    main()
