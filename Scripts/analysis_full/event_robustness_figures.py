# Aggregate + figures for event_robustness_analysis.py output.  Reads event_robustness_results.csv.
# Question answered: do maintenance / reagent-lot events invalidate a recently-trained maha model, item-dep?
# Significance (PLAN §11 review): report BOTH
#   - statistical: post_rate > per-item placebo 95th percentile
#   - operational (primary conclusion): post_rate >= 2%  (>=10 alerts / 500 = force-retrain trigger)
# Figures (FIGURE_RULES: gray main / color suppl, 600 dpi):
#   figA_sensitivity_heatmap : item x {cell+lamp, mechanism, reagent} -> fraction of events with post_rate>=2%
#   figB_postrate_by_event   : post_rate distributions per event type (events vs placebo), pooled
#   figC_localization        : baseline-dev vs reaction-dev for significant events (optical vs chemical)
# *** CODE-REVIEW STAGE: created but NOT executed (needs the results CSV). ***
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(BASE, 'Results', '20260718_EventRobustness')
RESULTS = os.path.join(OUT, 'event_robustness_results.csv')
FIGDIR = os.path.join(BASE, 'ForPaper', 'Suppl08_MaintenanceLotRobustness', 'Figures')
PANEL = ['TP', 'ALB', 'TB', 'DB', 'Ca', 'IP', 'CRE', 'UA', 'CRP', 'TC', 'HDL-C', 'LDL-C', 'TG']
ETYPES = ['cell_lamp', 'mechanism', 'reagent']
OP_THRESH = 0.02       # operational: post_rate >= 2%  (note: natural 500-window drift reaches this ~20% of time)
SURGE_THRESH = 0.05    # clear event-driven surge: post_rate >= 5% (above the placebo MAX ~5% -> ~never by drift)


def save(fig, name, gray):
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, f'{name}_{"gray" if gray else "color"}.png')
    fig.savefig(p, dpi=600); plt.close(fig)
    if gray:
        try:
            from PIL import Image
            Image.open(p).convert('L').save(p, dpi=(600, 600))
        except Exception:
            pass


def load():
    df = pd.read_csv(RESULTS)
    # per-item placebo 95th percentile of post_rate (natural drift null), pooled over analyzers
    plc = df[df['event_type'] == 'placebo'].groupby('item')['post_rate'].quantile(0.95)
    df['placebo95'] = df['item'].map(plc)
    df['sig_stat'] = df['post_rate'] > df['placebo95']
    df['sig_op'] = df['post_rate'] >= OP_THRESH
    return df, plc


def summary(df):
    rows = []
    for it in PANEL:
        for et in ETYPES:
            g = df[(df['item'] == it) & (df['event_type'] == et)]
            if len(g) == 0:
                continue
            rows.append(dict(item=it, module=g['module'].iloc[0], event_type=et, n_events=len(g),
                             median_post=round(g['post_rate'].median(), 4),
                             max_post=round(g['post_rate'].max(), 4),
                             frac_ge2pct=round(g['sig_op'].mean(), 3),
                             frac_gt_placebo95=round(g['sig_stat'].mean(), 3),
                             median_base_dev=round(g['baseline_dev_sigma'].median(), 3),
                             median_reac_dev=round(g['reaction_dev_sigma'].median(), 3)))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, 'event_robustness_summary.csv'), index=False)
    return s


def figA(df, summ, gray):
    # heatmap item x event_type -> fraction of events causing a CLEAR SURGE (post_rate >= 5%).
    # 5% is above the placebo MAX (~5%), so it is essentially never reached by natural 500-window drift
    # -> a clean, placebo-baseline-free measure of event-driven model invalidation.
    d = df[df['event_type'].isin(ETYPES)].assign(surge=df['post_rate'] >= SURGE_THRESH)
    piv = d.groupby(['item', 'event_type'])['surge'].mean().unstack().reindex(PANEL)[ETYPES]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    cmap = 'Greys' if gray else 'OrRd'
    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=0.15, aspect='auto')
    ax.set_xticks(range(len(ETYPES))); ax.set_xticklabels(['cell+lamp', 'mechanism', 'reagent'])
    ax.set_yticks(range(len(PANEL))); ax.set_yticklabels([f'{it} (m{summ[summ.item==it].module.iloc[0]})' for it in PANEL])
    for i in range(len(PANEL)):
        for j in range(len(ETYPES)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if (gray and v > 0.08) else 'black')
    ax.set_title('Fraction of events causing a clear surge (post-flag ≥5%)\n'
                 '(≥5% is above natural-drift max ≈5%; = model invalidated)'
                 + ('  [grayscale]' if gray else '  [color]'), fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='fraction of events with post ≥5%')
    fig.tight_layout(); save(fig, 'figA_sensitivity_heatmap', gray)


def figB(df, gray):
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ['placebo'] + ETYPES
    data = [df[df['event_type'] == c]['post_rate'].values * 100 for c in cats]
    bp = ax.boxplot(data, labels=['placebo\n(null)', 'cell+lamp', 'mechanism', 'reagent'],
                    showfliers=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('0.8' if gray else '#9ecae1')
    ax.axhline(2.0, color='0.4' if gray else '#ff7f0e', ls=':', lw=1.1, label='2% (within natural drift)')
    ax.axhline(5.0, color='0.1' if gray else '#d62728', ls='--', lw=1.3, label='5% clear surge (> drift max)')
    ax.set_ylabel('post-event flag rate (%)'); ax.set_yscale('symlog', linthresh=1)
    ax.set_title('Post-event flag rate by event type (pooled items/analyzers)'
                 + ('  [grayscale]' if gray else '  [color]'), fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); save(fig, 'figB_postrate_by_event', gray)


def figC(df, gray):
    # optical (baseline) vs chemical (reaction) localization for significant (op) events
    sig = df[(df['event_type'].isin(ETYPES)) & (df['sig_op'])]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    marks = {'cell_lamp': 'o', 'mechanism': 's', 'reagent': '^'}
    for et in ETYPES:
        g = sig[sig['event_type'] == et]
        ax.scatter(g['baseline_dev_sigma'], g['reaction_dev_sigma'], marker=marks[et], s=45,
                   facecolor=('none' if gray else None), edgecolor='0.1', label=et.replace('_', '+'))
    lim = max(1.0, sig[['baseline_dev_sigma', 'reaction_dev_sigma']].max().max() * 1.1) if len(sig) else 1.0
    ax.plot([0, lim], [0, lim], '0.6', ls=':', lw=1)
    ax.set_xlabel('baseline-point deviation (σ)  [optical]'); ax.set_ylabel('reaction-point deviation (σ)  [chemical]')
    ax.set_title('Where the shift is, for events that invalidated the model\n(above diagonal = chemical/kinetic; below = optical)'
                 + ('  [grayscale]' if gray else '  [color]'), fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); save(fig, 'figC_localization', gray)


def main():
    df, plc = load()
    summ = summary(df)
    print('=== summary (frac of events with post_rate>=2%) ===')
    print(summ.to_string(index=False))
    print('\nplacebo 95th pct post_rate by item:')
    print((plc * 100).round(2).to_string())
    for gray in (True, False):
        figA(df, summ, gray); figB(df, gray); figC(df, gray)
    print('DONE ->', FIGDIR)


if __name__ == '__main__':
    main()
