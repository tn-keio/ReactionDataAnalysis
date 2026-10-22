# Stage 3d: with the s3a set-up (v2 benchmark, unclipped, 21 items, 13 methods,
# simplified style: five methods in colour, the other eight in grey), redraw
#   - Fig. 4(a) operating summary with the legend moved above the bars: in make_figures.fig3_operating_summary the
#     upper-right legend hid the specificity marks of the three right-most methods (already so in the submission);
#   - Fig. S9(a)(b) per-item recall vs N (k >= 2 and k = 1) and Fig. S10 recall by anomaly type vs N, which were
#     still the submitted 11-method, clipped versions; the titles now say 21-item panel.  With 13 methods the
#     shared legends of make_figures (figS_byitem: below the panels; fig4_recall_by_type: inside the last panel)
#     were cut off or covered lines, so both functions are copied here with the legend placed outside the axes
#     and the figure saved with a tight bounding box.  Empirical Mahalanobis is drawn on top (zorder 6): where
#     both estimators reach 1.0 it was hidden under robust covariance.
# fig3_operating_summary, figS_byitem and fig4_recall_by_type of make_figures.py are copied here with only the
# y-range/ticks, legend placement, x tick labels and saving changed; everything else (data, styles) is s3a's.
# Fig. 4(a)+(b) is recomposed by s3c afterwards.
# Output: ForPaper/Rev_Figs2to4_v2/Figures_simplified_s3d/  (colour only)
# py3.14 (no TensorFlow).
import os
import numpy as np
import matplotlib.pyplot as plt

import s3a_figs_2to4_v2 as S

F = S.F
OUT = os.path.join(S.OUT, 'Figures_simplified_s3d')


def fig3_operating_summary_legend_top(gray):
    N = 500
    order = sorted(F.METHODS, key=lambda m: -np.mean(F.item_vals(m, N, 'recall_main')))
    rm = [np.mean(F.item_vals(m, N, 'recall_main')) for m in order]
    worst = [min(F.item_vals(m, N, 'recall_main')) for m in order]
    spec = [np.mean(F.item_vals(m, N, 'specificity_clean')) for m in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bcol = ['0.55' if gray else '#A6C8E0'] * len(order)
    ax.bar(x, rm, color=bcol, width=0.62, label='Mean recall (k≥2)')
    ax.plot(x, worst, ls='none', marker='v', ms=8, color=('0.0' if gray else '#D55E00'), label='Worst-item recall')
    ax.plot(x, spec, ls='none', marker='_', ms=16, mew=2.4, color=('0.25' if gray else '#222222'),
            label='Specificity (clean)')
    ax.set_xticks(x); ax.set_xticklabels([F.LABEL[m] for m in order], rotation=40, ha='right', fontsize=8)
    # changed: head-room above 1.0 for a one-row legend, so no mark is covered
    ax.set_ylim(0, 1.17); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel('Rate')
    ax.set_title(f'Operating point at N=500, P99.5 (points 5–38, {len(F.PANEL)}-item panel)', fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=8, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(); F.save(fig, 'fig3_operating_summary', gray)


def byitem(metric_fn, ylabel, name, gray, title):
    panel = F.PANEL
    ncol = 4; nrow = int(np.ceil(len(panel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, it in zip(axes, panel):
        for m in F.METHODS:
            ys = [metric_fn(it, m, N) for N in F.N_GRID]
            ax.plot(F.N_GRID, ys, label=F.LABEL[m], **F.style(gray, m, emph=True))
        ax.set_xscale('log'); ax.set_xticks(F.N_GRID); ax.set_xticklabels([str(n) for n in F.N_GRID], fontsize=7)
        ax.set_ylim(0, 1.03); ax.set_title(it, fontsize=10); ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=True)   # the last row is incomplete, so label every panel's N axis
    # the empty cells of the last row hold the legend
    for ax in axes[len(panel):]:
        ax.axis('off')
    for i in range(0, len(panel), ncol):
        axes[i].set_ylabel(ylabel, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[len(panel)].legend(handles, labels, fontsize=8, ncol=2, loc='center left', frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); F.save(fig, name, gray, tight_bbox=True)


def recall_by_type(gray):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    for ax, t in zip(axes, F.TYPES):
        for m in F.METHODS:
            ys = [F.mean_ci(F.item_vals(m, N, f'recall_{t}'))[0] for N in F.N_GRID]
            ax.plot(F.N_GRID, ys, label=F.LABEL[m], **F.style(gray, m))
        ax.set_xscale('log'); ax.set_xticks(F.N_GRID); ax.set_xticklabels([str(n) for n in F.N_GRID])
        ax.set_ylim(0, 1.03); ax.set_title(t); ax.set_xlabel('N'); ax.grid(alpha=0.3)
    axes[0].set_ylabel('Recall (k≥2)')
    F.side_legend(axes[-1])
    fig.suptitle('Recall by anomaly type vs training size — points 5–38, 21-item panel', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); F.save(fig, 'fig4_recall_by_type', gray, tight_bbox=True)


def figS9a(gray):
    byitem(lambda it, m, N: F.KEY[(it, m, N)]['budgets']['99.5']['recall_main']['mean'],
                  'recall (k≥2)', 'figS1_recall_vs_N_byitem', gray,
                  title='Detection recall vs training size, by item — points 5–38, 21-item panel')


def figS9b(gray):
    byitem(lambda it, m, N: float(np.nanmean(
        [F.KEY[(it, m, N)]['budgets']['99.5'].get(f'recall_{t}_k1', {'mean': np.nan})['mean'] for t in F.TYPES])),
                  'recall (k=1)', 'figS2_recall_k1_byitem', gray,
                  title='Subtle (1σ) anomaly recall vs training size, by item — points 5–38, 21-item panel')


def main():
    F.R_ALL = F.R = S.load_records()
    F.KEY = {(r['item'], r['method'], r['N']): r for r in F.R}
    miss = [(i, m, n) for i in S.PANEL21 for m in S.METHODS13 for n in F.N_GRID if (i, m, n) not in F.KEY]
    assert not miss, f'missing records: {miss[:5]} ... ({len(miss)})'
    S.setup('simplified')
    base_style = F.style

    def style_maha_on_top(gray, m, emph=True):
        d = base_style(gray, m, emph)
        if m == 'maha':
            d['zorder'] = 6
        return d
    F.style = style_maha_on_top
    F.FIGS = OUT
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig3_operating_summary_legend_top, figS9a, figS9b, recall_by_type):
        fn(False)
    print('->', OUT, sorted(os.listdir(OUT)))


if __name__ == '__main__':
    main()
