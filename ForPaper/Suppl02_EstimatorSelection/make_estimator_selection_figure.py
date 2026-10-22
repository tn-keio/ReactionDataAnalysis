"""Supplementary figures for main-estimator selection (Empirical vs Robust vs E-union-R), points 5-38.

Reads the outputs of ExperimentalCode/Scripts/run_estimator_selection.py and renders, per
ForPaper/FIGURE_RULES.md, a colour version (submitted) and a grayscale version (print fallback).

These become panels (b), (c) and (d) of Supplementary Fig. S2 -- panel (a) is the schematic from
make_estimator_concept_figure.py. The panel letters are assigned when the supplementary document is
assembled (exports/build_suppl.py), so no figure number is drawn inside the images.

  estimator_recall_by_type      (S2b) -- per-anomaly-type recall, WORST item (min across items), for
                                        E / R / OR. B_drift highlighted (it is Robust's structural
                                        blind spot and the discriminator between estimators).
  estimator_severity_worstitem  (S2c) -- severity response: worst-item recall at k=1 (subtle) and
                                        mean specificity, by estimator.
  estimator_spec_vs_bdrift      (S2d) -- specificity vs B_drift recall, one point per item, by
                                        estimator. Shows the trade-off and whether OR reaches the
                                        top-right (high recall AND high specificity) that neither
                                        E nor R does.

*** PLAN STAGE: run only AFTER run_estimator_selection.py has produced the CSVs. ***
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.abspath(os.path.join(HERE, '..', '..'))
RES = os.path.join(MAIN, 'Results', '20260716_EstimatorSelection_E_vs_R')
FIG = os.path.join(HERE, 'Figures')
WINDOW = 'p5_38'                                   # primary window
POS_TYPES = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']
METHODS = ['E', 'R', 'OR']
GRAY = {'E': '0.1', 'R': '0.6', 'OR': '0.82'}
# Colour-vision-safe (FIGURE_RULES.md 2b). The old tab10 trio put R on red and
# OR on green: CIEDE2000 4.6 apart under deuteranopia, i.e. the same colour.
# E and R reuse the codes the method-comparison figures give Empirical and
# Robust Mahalanobis, so an estimator keeps one colour across the paper.
COLOR = {'E': '#0072B2', 'R': '#222222', 'OR': '#D55E00'}
MARK = {'E': 'o', 'R': 's', 'OR': '^'}
HATCH = {'E': None, 'R': '///', 'OR': 'xxx'}          # gray-version fills (distinct without color)


def load():
    bi = pd.read_csv(os.path.join(RES, 'summary_by_item.csv'))
    return bi[bi.window == WINDOW].copy()


def fig1(bi, pal, path, color):
    worst = bi.groupby('method')[[f'recall_{t}' for t in POS_TYPES]].min().reindex(METHODS)
    x = np.arange(len(POS_TYPES)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, m in enumerate(METHODS):
        vals = [worst.loc[m, f'recall_{t}'] for t in POS_TYPES]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=m, color=pal[m],
                      hatch=(HATCH[m] if not color else None), edgecolor='k', linewidth=0.5)
    bi_idx = POS_TYPES.index('B_drift')
    ax.axvspan(bi_idx - 0.45, bi_idx + 0.45, color=('#ffe08a' if color else '0.9'), zorder=0, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(POS_TYPES); ax.set_ylim(0, 1.03)
    ax.set_ylabel('worst-item recall (min across 21 items, k>=2)')
    # No figure number in the title: these are panels of Supplementary Fig. S2,
    # and an embedded number goes stale the moment the numbering changes.
    ax.set_title('Worst-item recall by anomaly type — B_drift discriminates the estimators')
    ax.legend(title='estimator'); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=600); plt.close(fig); print('->', path)


def fig2(bi, pal, path, color):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for m in METHODS:
        s = bi[bi.method == m]
        ax.scatter(s['spec'], s['recall_B_drift'], s=45, marker=MARK[m], color=pal[m],
                   edgecolor='k', linewidth=0.4, alpha=0.85, label=m)
    ax.axhline(0.8, ls=':', color='0.6', lw=1); ax.axvline(0.97, ls=':', color='0.6', lw=1)
    ax.set_xlabel('specificity (real normals)'); ax.set_ylabel('B_drift recall (k>=2)')
    ax.set_title('Specificity vs B_drift recall per item\n(top-right = both good; does OR get there?)')
    ax.set_xlim(right=1.001); ax.set_ylim(-0.02, 1.03); ax.legend(title='estimator'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=600); plt.close(fig); print('->', path)


def fig3(bi, pal, path, color):
    agg = bi.groupby('method').agg(recall_k1_min=('recall_k1', 'min'), recall_k1_mean=('recall_k1', 'mean'),
                                   spec_mean=('spec', 'mean'), recall_main_min=('recall_main', 'min')).reindex(METHODS)
    x = np.arange(len(METHODS)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.9))
    # Colour identifies the estimator; k=1 vs k>=2 is carried by position and
    # shade, so the legend must describe position and shade, not colour. Neutral
    # swatches keep it from looking like the series colour means something.
    ax.bar(x - w / 2, agg['recall_k1_min'], w,
           color=[pal[m] for m in METHODS], edgecolor='k', linewidth=0.5)
    ax.bar(x + w / 2, agg['recall_main_min'], w,
           color=[pal[m] for m in METHODS], alpha=0.5, edgecolor='k', linewidth=0.5)
    handles = [Patch(facecolor='0.35', edgecolor='k', linewidth=0.5,
                     label='left bar (solid): worst-item recall @ k=1 (subtle)'),
               Patch(facecolor='0.78', edgecolor='k', linewidth=0.5,
                     label='right bar (pale): worst-item recall (k>=2)')]

    # mean specificity printed in a header band: inside the bars it was
    # unreadable wherever the bar is dark
    for i, m in enumerate(METHODS):
        ax.annotate(f"spec {agg.loc[m,'spec_mean']:.3f}", (i, 1.085), ha='center',
                    fontsize=9.5, color='#222222')
    ax.axhline(1.045, color='0.75', lw=0.8)
    # headroom for the specificity band above the bars; the legend sits below the
    # axes so it cannot cover those values
    ax.set_xticks(x); ax.set_xticklabels(METHODS); ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('worst-item recall'); ax.set_title('Severity response & worst-item recall by estimator')
    ax.legend(handles=handles, fontsize=8.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=600); plt.close(fig); print('->', path)


def main():
    if not os.path.exists(os.path.join(RES, 'summary_by_item.csv')):
        raise SystemExit(f'Missing {RES}/summary_by_item.csv -- run run_estimator_selection.py first.')
    os.makedirs(FIG, exist_ok=True)
    bi = load()
    for name, fn in [('estimator_recall_by_type', fig1), ('estimator_spec_vs_bdrift', fig2),
                     ('estimator_severity_worstitem', fig3)]:
        fn(bi, GRAY, os.path.join(FIG, f'{name}_gray.png'), color=False)
        fn(bi, COLOR, os.path.join(FIG, f'{name}_color.png'), color=True)


if __name__ == '__main__':
    main()
