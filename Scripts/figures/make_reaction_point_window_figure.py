"""Supplementary figure: choice of the reaction photometric-point window (start = point 5).

Reproduces a publication-quality 2-panel figure from the frozen results of
`Results/20260716_ReactionPoints_Usage` (no re-computation of the heavy detection run):

  Panel A -- Where bubble noise ends: per-point curvature (|2nd diff|, % of endpoint IQR),
             across-item typical (median of p50) and worst-case (max of p99). Bubble jitter is
             confined to points 1-4 and returns to the flat-baseline level by ~point 5; reaction
             onset is ~point 20.
  Panel B -- What starting earlier gains/costs: specificity, reaction-region recall and
             early-region recall vs the window start point (mean over items, Empirical Mahalanobis).
             Lowering the start from 11 (current) to 5 recovers early-region anomaly detection with
             essentially no loss of specificity or reaction-region recall.

Outputs gray (main-text) and color (Supplemental) versions per ForPaper/FIGURE_RULES.md.
Series are distinguished by line style + marker so the gray version is legible.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.abspath(os.path.join(HERE, '..', '..'))
# ForPaper edition: analysis restricted to the 21 ForPaper=1 items (no non-panel items).
RES = os.path.join(MAIN, 'Results', '20260716_ReactionPoints_Usage_ForPaper')
FIG = os.path.join(HERE, 'Figures')

START_CURRENT = 11
START_CHOSEN = 5
ONSET = 20


def load():
    tails = pd.read_csv(os.path.join(RES, 'noise_profile_tails.csv'))
    summ = pd.read_csv(os.path.join(RES, 'summary_by_start.csv'))
    # Panel A: across-item aggregates per point
    a = (tails.groupby('point')
         .agg(p50_med=('p50', 'median'), p99_max=('p99', 'max')).reset_index())
    # Panel B: mean over items, Empirical
    e = summ[summ.method == 'E']
    b = (e.groupby('start')
         .agg(spec=('spec', 'mean'), recall_react=('recall_react', 'mean'),
              recall_early=('recall_early', 'mean')).reset_index().sort_values('start'))
    return a, b


def draw(a, b, palette, path):
    c = palette
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3))

    # ---- Panel A: per-point jitter ----
    m = a[a.point <= 24]
    axA.axvspan(0.5, 4.5, color=c['bubble'], alpha=0.18, lw=0, zorder=0)
    axA.plot(m.point, np.clip(m.p50_med, 0, 18), color=c['s1'], ls='-', marker='o', ms=3.5,
             lw=1.6, label='typical jitter (median of items, p50)')
    axA.plot(m.point, np.clip(m.p99_max, 0, 18), color=c['s2'], ls='--', marker='^', ms=4,
             lw=1.6, label='worst-case jitter (max of items, p99)')
    base = float(a.loc[a.point == 15, 'p99_max'].iloc[0])
    axA.axhline(base, color=c['ref'], ls=':', lw=1.1, label='baseline level (point 15)')
    axA.axvline(START_CHOSEN, color=c['chosen'], ls='-', lw=1.2, alpha=0.9)
    axA.annotate('onset ~pt20', xy=(ONSET, base + 1.5), fontsize=8, ha='center', color=c['txt'])
    axA.axvline(ONSET, color=c['txt'], ls=':', lw=0.9, alpha=0.6)
    axA.set_xlabel('photometric point'); axA.set_ylabel('curvature  |2nd diff|  (% of endpoint IQR)')
    axA.set_title('(A) Bubble noise settles by ~point 5')
    axA.set_xlim(1, 24); axA.set_ylim(0, 18); axA.grid(alpha=0.3); axA.legend(fontsize=7.5, loc='upper right')
    axA.text(2.6, 16.6, 'bubble', color=c['bubble_txt'], fontsize=8.5, ha='center', style='italic')

    # ---- Panel B: detection vs start point ----
    axB.axvspan(0.5, 4.5, color=c['bubble'], alpha=0.18, lw=0, zorder=0)
    axB.plot(b.start, b.spec, color=c['spec'], ls='-', marker='o', ms=5, lw=1.7, label='specificity (real normals)')
    axB.plot(b.start, b.recall_react, color=c['react'], ls='--', marker='s', ms=5, lw=1.7, label='recall: reaction-region anomalies')
    axB.plot(b.start, b.recall_early, color=c['early'], ls='-.', marker='^', ms=6, lw=1.8, label='recall: early-region anomalies')
    axB.axvline(START_CHOSEN, color=c['chosen'], ls='-', lw=1.2, alpha=0.9)
    axB.annotate('chosen\nstart 5', xy=(START_CHOSEN, 0.03), fontsize=7.5, ha='center', color=c['chosen'])
    axB.set_xlabel('window start point  (end = 38)'); axB.set_ylabel('rate')
    axB.set_title('(B) Starting at point 5 recovers early-region detection')
    axB.set_ylim(0, 1.04); axB.set_xlim(0, 21); axB.grid(alpha=0.3); axB.legend(fontsize=7.5, loc='center right')

    fig.tight_layout()
    fig.savefig(path, dpi=600); plt.close(fig)
    print('->', path)


GRAY = dict(s1='0.15', s2='0.15', ref='0.45', bubble='0.5', bubble_txt='0.25', txt='0.2',
            chosen='0.0', cur='0.55', spec='0.1', react='0.4', early='0.0')
COLOR = dict(s1='#1f77b4', s2='#d62728', ref='#7f7f7f', bubble='#d62728', bubble_txt='#d62728',
             txt='#333333', chosen='#2ca02c', cur='#7f7f7f', spec='#1f77b4', react='#2ca02c', early='#ff7f0e')


def main():
    os.makedirs(FIG, exist_ok=True)
    a, b = load()
    draw(a, b, GRAY, os.path.join(FIG, 'reaction_point_window_gray.png'))
    draw(a, b, COLOR, os.path.join(FIG, 'reaction_point_window_color.png'))


if __name__ == '__main__':
    main()
