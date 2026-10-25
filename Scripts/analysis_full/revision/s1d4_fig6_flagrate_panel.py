# Stage 1d-4: panel (b) of the s1d3 rolling-evaluation figure replaced.  The assay x month heat map of the empirical
# routine flag rate (s1d3) duplicated Fig. S11(b) and showed one method only, so recall in (a) could not be
# read against its cost.  New (b): per assay, median and 95th-percentile month of the routine flag rate for
# the same three methods, drawn like (a); the dotted line is the nominal 0.5% of the self-calibrated P99.5.
# Monthly values and the event months stay in Fig. S11(b).  s1d3 is not changed; its loader and styles are reused.
# Output: Results/20260917_Rev_R2_RollingContemporaneous/figures/fig6_rolling_evaluation_v2_color.png (600 dpi)
# py3.14 (no TensorFlow).
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import blended_transform_factory

import s1d3_fig6_figS11_rolling as F

ITEMS, COL, LAB, MRK = F.ITEMS, F.COL, F.LAB, F.MRK


def dumbbell(ax, d, col, lo, hi):
    # filled marker = lo statistic, open marker = hi statistic, joined per assay; three methods side by side
    x = np.arange(len(ITEMS))
    for j, m in enumerate('ERP'):
        g = d[d.method == m].groupby('item')[col]
        a, b = lo(g).reindex(ITEMS), hi(g).reindex(ITEMS)
        xo = x + (j - 1) * 0.25
        ax.vlines(xo, a, b, color=COL[m], lw=1.6, zorder=3)
        ax.plot(xo, a, ls='none', marker=MRK[m], ms=6, color=COL[m], zorder=4)
        ax.plot(xo, b, ls='none', marker=MRK[m], ms=6, mfc='white', mec=COL[m], mew=1.4, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(ITEMS, rotation=45, ha='right', fontsize=8)
    ax.set_xlim(-0.6, len(ITEMS) - 0.4)
    ax.grid(alpha=0.3, axis='y')


def refline(ax, y, lab):
    # reference line labelled just outside the right edge, clear of the markers
    ax.axhline(y, color='0.45', ls=':', lw=1.2, zorder=1)
    ax.text(1.005, y, lab, transform=blended_transform_factory(ax.transAxes, ax.transData),
            va='center', ha='left', fontsize=8, color='0.35')


def fig6(d):
    fig = plt.figure(figsize=(10.5, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.8], hspace=0.30)

    ax = fig.add_subplot(gs[0])
    dumbbell(ax, d, 'unscreened_recall_main', lambda g: g.median(), lambda g: g.min())
    refline(ax, 0.875, '0.875')
    ax.set_ylim(0, 1.05); ax.set_ylabel('Recall (k ≥ 2)')
    ax.set_title('(a) Monthly recall on anomalies injected into the next 500 curves (median and worst month)',
                 fontsize=10, loc='left')

    ax = fig.add_subplot(gs[1])
    dumbbell(ax, d, 'routine_flag_rate', lambda g: g.median(), lambda g: g.quantile(0.95))
    refline(ax, 0.005, '0.5%')
    ax.set_ylim(0, 0.06); ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel('Routine flag rate')
    ax.set_title('(b) Monthly routine flag rate on the same 500 real curves (median and 95th-percentile month)',
                 fontsize=10, loc='left')
    # one legend for both panels: colour/shape = method, fill = which month
    h = [Line2D([], [], ls='none', marker=MRK[m], ms=6, color=COL[m], label=LAB[m]) for m in 'ERP']
    h += [Line2D([], [], ls='none', marker='o', ms=6, color='0.5', label='median month'),
          Line2D([], [], ls='none', marker='o', ms=6, mfc='white', mec='0.5', mew=1.4,
                 label='worst month (a) / 95th-percentile month (b)')]
    ax.legend(handles=h, fontsize=7.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.24), frameon=False)
    F.save(fig, 'fig6_rolling_evaluation_v2_color.png')


def main():
    d = F.load()
    assert d.groupby('method').size().eq(1158).all(), d.groupby('method').size()
    hi = d.groupby(['item', 'method']).routine_flag_rate.quantile(0.95)
    assert hi.max() < 0.06, hi.idxmax()          # nothing clipped by the y-axis of (b)
    fig6(d)
    print('->', F.OUT)


if __name__ == '__main__':
    main()
