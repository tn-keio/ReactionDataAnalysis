# Fig. S6 redrawn from the exact-time S9 run (s1s), for the revised manuscript.
# Reuses event_robustness_figures.py unchanged; only the input, the output folder and the
# heat-map colour range differ (CRP reaches 0.38 on the exact-time data, above the old vmax 0.15).
# Input : Results/20260924_Rev_S9_RawLot/exact_time/event_robustness_results.csv
# Output: Results/20261021_Rev_FigS6_ExactTime/{figA,figB,figC}_*_color.png, event_robustness_summary.csv,
#         surge_localization.csv (baseline vs reaction deviation of the >=5% reagent surges)
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import event_robustness_figures as F  # noqa: E402

F.RESULTS = os.path.join(F.BASE, 'Results', '20260924_Rev_S9_RawLot', 'exact_time', 'event_robustness_results.csv')
F.OUT = os.path.join(F.BASE, 'Results', '20261021_Rev_FigS6_ExactTime')
F.FIGDIR = F.OUT
VMAX = 0.40


def figA(df, summ, gray):
    # same as F.figA, with the colour range widened to the exact-time maximum
    d = df[df['event_type'].isin(F.ETYPES)].assign(surge=df['post_rate'] >= F.SURGE_THRESH)
    piv = d.groupby(['item', 'event_type'])['surge'].mean().unstack().reindex(F.PANEL)[F.ETYPES]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(piv.values, cmap='Greys' if gray else 'OrRd', vmin=0, vmax=VMAX, aspect='auto')
    ax.set_xticks(range(len(F.ETYPES))); ax.set_xticklabels(['cell+lamp', 'mechanism', 'reagent'])
    ax.set_yticks(range(len(F.PANEL)))
    ax.set_yticklabels([f'{it} (m{summ[summ.item == it].module.iloc[0]})' for it in F.PANEL])
    for i in range(len(F.PANEL)):
        for j in range(len(F.ETYPES)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if v > 0.25 else 'black')
    ax.set_title('Fraction of events causing a clear surge (post-flag ≥5%)\n'
                 '(placebo windows reached ≥5% in 0.4%)', fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='fraction of events with post ≥5%')
    fig.tight_layout(); F.save(fig, 'figA_sensitivity_heatmap', gray)


def figB(df, gray):
    # same as F.figB, but the placebo maximum is 5.6% here, so the 5% line is not labelled as above it,
    # and the symlog axis is not extended below zero
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ['placebo'] + F.ETYPES
    data = [df[df['event_type'] == c]['post_rate'].values * 100 for c in cats]
    bp = ax.boxplot(data, labels=['placebo\n(null)', 'cell+lamp', 'mechanism', 'reagent'],
                    showfliers=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('0.8' if gray else '#9ecae1')
    ax.axhline(2.0, color='0.4' if gray else '#ff7f0e', ls=':', lw=1.1, label='2% (within natural drift)')
    ax.axhline(5.0, color='0.1' if gray else '#d62728', ls='--', lw=1.3, label='5% clear surge (placebo 0.4%)')
    ax.set_ylabel('post-event flag rate (%)'); ax.set_yscale('symlog', linthresh=1); ax.set_ylim(0, 110)
    ax.set_title('Post-event flag rate by event type (pooled items/analyzers)', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); F.save(fig, 'figB_postrate_by_event', gray)


def main():
    os.makedirs(F.OUT, exist_ok=True)
    df, plc = F.load()
    summ = F.summary(df)
    figA(df, summ, False); figB(df, False); F.figC(df, False)
    s = df[(df['event_type'] == 'reagent') & (df['post_rate'] >= F.SURGE_THRESH)]
    s[['analyzer', 'item', 'event_date', 'lot', 'post_rate', 'baseline_dev_sigma', 'reaction_dev_sigma']] \
        .to_csv(os.path.join(F.OUT, 'surge_localization.csv'), index=False)
    print('reagent surges >=5%:', len(s),
          '| reaction > baseline deviation:', int((s['reaction_dev_sigma'] > s['baseline_dev_sigma']).sum()),
          '| median baseline / reaction dev (sigma):',
          round(s['baseline_dev_sigma'].median(), 3), '/', round(s['reaction_dev_sigma'].median(), 3))
    for et in ['placebo'] + F.ETYPES:
        g = df[df['event_type'] == et]
        print(f'{et:10s} n={len(g):5d} >=5% {100 * (g.post_rate >= 0.05).mean():.2f}% '
              f'>=2% {100 * (g.post_rate >= 0.02).mean():.1f}% median {100 * g.post_rate.median():.2f}% '
              f'max {100 * g.post_rate.max():.1f}%')
    print('DONE ->', F.OUT)


if __name__ == '__main__':
    main()
