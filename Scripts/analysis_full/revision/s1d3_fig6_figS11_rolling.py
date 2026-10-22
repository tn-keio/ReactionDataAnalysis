# Stage 1d-3 (R2, 2026-10-21): Fig. 6 and Fig. S11 redrawn for the rolling evaluation of s1d (each monthly
# rolling recent-500 model scored on the next 500 curves and on anomalies injected into them), replacing the
# fixed-probe kappa figures of the submission.  Data: s1d series.csv, no clipping, unscreened battery.
#   Fig. 6  (a) per assay: median and worst month of k>=2 recall, empirical / robust / PCA, line at 0.875
#           (b) assay x month heatmap of the empirical routine flag rate, known events marked
#   Fig. S11 (a) monthly k>=2 recall per assay (small multiples), three methods
#            (b) monthly routine flag rate per assay (small multiples), three methods
#            (c) panel-mean recall against magnitude k (0.02-8 sigma-bar), three methods
# Colours, labels and fonts follow make_figures (empirical #0072B2, robust #222222, PCA #E69F00).
# Output: Results/20260917_Rev_R2_RollingContemporaneous/figures/  (colour, 600 dpi)
# py3.14 (no TensorFlow).
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import s1d_rolling_contemporaneous as D

OUT = os.path.join(D.OUT, 'figures')
COL = {'E': '#0072B2', 'R': '#222222', 'P': '#E69F00'}
LAB = {'E': 'Mahalanobis (Empirical)', 'R': 'Mahalanobis (Robust)', 'P': 'PCA'}
MRK = {'E': 'o', 'R': 'D', 'P': 's'}
LS = {'E': '-', 'R': '-', 'P': '--'}
ITEMS = D.P.proc_names(21)
EVENTS = [('2021-07', '2021-09', 'optical event'), ('2023-06', '2024-04', 'IgM reagent'),
          ('2024-12', '2025-03', 'IgG reagent')]


def load():
    d = pd.read_csv(os.path.join(D.OUT, 'series.csv'))
    d = d[d['clip'] == 'noclip'].copy()
    d['t'] = pd.PeriodIndex(d['month'], freq='M').to_timestamp()
    return d


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=600, bbox_inches='tight')
    plt.close(fig)


def fig6(d):
    fig = plt.figure(figsize=(10.5, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.62)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(ITEMS))
    for j, m in enumerate('ERP'):
        g = d[d.method == m].groupby('item').unscreened_recall_main
        med, low = g.median().reindex(ITEMS), g.min().reindex(ITEMS)
        xo = x + (j - 1) * 0.25
        ax.vlines(xo, low, med, color=COL[m], lw=1.6, zorder=3)
        ax.plot(xo, med, ls='none', marker=MRK[m], ms=6, color=COL[m], label=f'{LAB[m]}: median month', zorder=4)
        ax.plot(xo, low, ls='none', marker=MRK[m], ms=6, mfc='white', mec=COL[m], mew=1.4,
                label=f'{LAB[m]}: worst month', zorder=4)
    ax.axhline(0.875, color='0.45', ls=':', lw=1.2, zorder=1)
    ax.text(-0.5, 0.885, '0.875', va='bottom', ha='left', fontsize=8, color='0.35')
    ax.set_xticks(x); ax.set_xticklabels(ITEMS, rotation=45, ha='right', fontsize=8)
    ax.set_xlim(-0.6, len(ITEMS) - 0.4); ax.set_ylim(0, 1.05); ax.set_ylabel('Recall (k ≥ 2)')
    ax.grid(alpha=0.3, axis='y')
    ax.set_title('(a) Monthly recall on anomalies injected into the next 500 curves (median and worst month)',
                 fontsize=10, loc='left')
    ax.legend(fontsize=7.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.24), frameon=False)

    ax = fig.add_subplot(gs[1])
    e = d[d.method == 'E']
    months = pd.period_range(e['month'].min(), e['month'].max(), freq='M')
    M = e.pivot(index='item', columns='month', values='routine_flag_rate').reindex(index=ITEMS,
                                                                               columns=[str(p) for p in months])
    bounds = [0, 0.01, 0.02, 0.05, 0.10, 1.0]
    cmap = ListedColormap(['#DCEAF7', '#9DC3E6', '#4A90C8', '#1C5D99', '#08306B'])
    cmap.set_bad('white')
    im = ax.imshow(np.ma.masked_invalid(M.values), aspect='auto', cmap=cmap,
                   norm=BoundaryNorm(bounds, cmap.N), interpolation='nearest')
    ax.set_yticks(range(len(ITEMS))); ax.set_yticklabels(ITEMS, fontsize=7.5)
    jan = [i for i, p in enumerate(months) if p.month == 1]
    ax.set_xticks(jan); ax.set_xticklabels([str(months[i].year) for i in jan], fontsize=8)
    ax.set_xticks(range(len(months)), minor=True)
    ax.tick_params(axis='x', which='minor', length=2)
    for a, b, lab in EVENTS:
        i0 = list(map(str, months)).index(a); i1 = list(map(str, months)).index(b)
        ax.annotate('', xy=(i0 - 0.5, -1.0), xytext=(i1 + 0.5, -1.0), annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color='0.25', lw=1.4))
        ax.text(max((i0 + i1) / 2, 3.0), -1.6, lab, ha='center', va='bottom', fontsize=7.5, color='0.25')
    ax.set_ylim(len(ITEMS) - 0.5, -2.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, spacing='uniform')
    cb.set_ticks(bounds); cb.set_ticklabels(['0', '1%', '2%', '5%', '10%', '100%'])
    cb.ax.tick_params(labelsize=7.5)
    ax.set_title('(b) Routine flag rate of empirical Mahalanobis on the next 500 real curves (white: no model)', fontsize=10, loc='left')
    save(fig, 'fig6_rolling_evaluation_color.png')


def small_multiples(d, col, ylabel, name, title, ylim, ref=None):
    nc = 4; nr = int(np.ceil(len(ITEMS) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 2.6 * nr), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, it in zip(axes, ITEMS):
        for a, b, lab in EVENTS:
            if lab.endswith('reagent') and not lab.startswith(it + ' '):   # reagent episodes only on their own assay
                continue
            ax.axvspan(pd.Period(a, 'M').to_timestamp(), pd.Period(b, 'M').to_timestamp() + pd.offsets.MonthEnd(0),
                       color='0.92', zorder=0)
        for m in 'PRE':
            g = d[(d.method == m) & (d.item == it)].sort_values('t')
            ax.plot(g.t, g[col], color=COL[m], ls=LS[m], lw=1.8 if m == 'E' else 1.2, label=LAB[m],
                    zorder=3 if m == 'E' else 2)
        if ref is not None:
            ax.axhline(ref, color='0.45', ls=':', lw=1.0)
        ax.set_title(it, fontsize=9); ax.set_ylim(*ylim); ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=True, labelsize=7)
    for ax in axes[len(ITEMS):]:
        ax.axis('off')
    for i in range(0, len(ITEMS), nc):
        axes[i].set_ylabel(ylabel, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    axes[len(ITEMS)].legend(h[::-1], l[::-1], fontsize=8, loc='center left', frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, name)


def recall_vs_k(d):
    ks = [0.02, 0.05, 0.1, 0.2, 0.5, 1]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for m in 'ERP':
        g = d[d.method == m]
        ys = [g[f'unscreened_recall_k{k}'].mean() for k in ks[:-1]] + [g.unscreened_recall_k1.mean(),
                                                                         g.unscreened_recall_main.mean()]
        ax.plot(ks + [2], ys, color=COL[m], marker=MRK[m], ls=LS[m], lw=2.2 if m == 'E' else 1.4, ms=6, label=LAB[m])
    ax.set_xscale('log'); ax.set_xticks(ks + [2]); ax.set_xticklabels(['0.02', '0.05', '0.1', '0.2', '0.5', '1', '≥2'])
    ax.set_xlabel('Magnitude k (σ̄)'); ax.set_ylabel('Recall (mean over assay-months)'); ax.set_ylim(0, 1.03)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right', frameon=False)
    ax.set_title('Recall against magnitude, anomalies injected into the next 500 curves', fontsize=9.5)
    fig.tight_layout(); save(fig, 'figS11c_rolling_recall_vs_k.png')


def export_csv(d):
    # machine-readable assay x month x method table for the supplement (aggregates only, no curves)
    cols = {'item': 'assay', 'month': 'month', 'method': 'method', 'test_span_d': 'eval_block_span_days',
            'routine_flag_rate': 'routine_flag_rate', 'unscreened_recall_main': 'recall_k2to8',
            'unscreened_recall_k1': 'recall_k1'}
    cols.update({f'unscreened_recall_k{k}': f'recall_k{k}' for k in [0.5, 0.2, 0.1, 0.05, 0.02]})
    cols.update({f'unscreened_{t}_main': f'recall_{t}_k2to8' for t in ['A_spike', 'B_drift', 'C_hump', 'E_baseline']})
    cols['unscreened_D_lag1'] = 'recall_D_lag1'
    out = d[list(cols)].rename(columns=cols)
    out['method'] = out['method'].map({'E': 'mahalanobis_empirical', 'R': 'mahalanobis_robust', 'P': 'pca'})
    out = out.sort_values(['assay', 'month', 'method'])
    out.round(4).to_csv(os.path.join(OUT, 'rolling_evaluation_by_assay_month.csv'), index=False)
    print('csv rows', len(out))


def main():
    d = load()
    export_csv(d)
    assert d.groupby('method').size().eq(1158).all(), d.groupby('method').size()
    fig6(d)
    small_multiples(d, 'unscreened_recall_main', 'recall (k ≥ 2)', 'figS11a_rolling_recall_bymonth.png',
                    'Monthly recall (k ≥ 2) of each rolling model on anomalies injected into the next 500 curves '
                    '(shaded: optical event; for IgM and IgG also the reagent episode)', (0, 1.03), ref=0.875)
    small_multiples(d, 'routine_flag_rate', 'routine flag rate', 'figS11b_rolling_flagrate_bymonth.png',
                    'Monthly routine flag rate of each rolling model on the next 500 real curves (shaded: optical event; for IgM and IgG also the reagent episode; '
                    'axis truncated at 0.15)', (0, 0.15))
    recall_vs_k(d)
    print('->', OUT, sorted(os.listdir(OUT)))


if __name__ == '__main__':
    main()
