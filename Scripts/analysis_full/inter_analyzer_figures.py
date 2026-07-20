# Aggregate + figures for inter_analyzer_analysis.py.  Reads inter_analyzer_results.csv.
# Q: are analyzers harmonized enough to pool their data for training?  item-dependent?
# Figures (FIGURE_RULES: gray main / color suppl, 600 dpi):
#   figA_transfer_matrix : train_src {#1,#2,#3,pool} x test_analyzer -> mean flag rate (diag = within)
#   figB_penalty_by_item : transfer penalty (cross-src flag - within) per item  (0 = harmonized)
#   figC_pool_vs_single  : (a) within vs pool flag rate per item;  (b) days to accrue 500 (single vs pooled)
# *** CODE-REVIEW STAGE: created but NOT executed (needs the results CSV). ***
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
OUT = os.path.join(BASE, 'Results', '20260718_InterAnalyzer')
RESULTS = os.path.join(OUT, 'inter_analyzer_results.csv')
FIGDIR = os.path.join(BASE, 'ForPaper', 'Suppl03_InterAnalyzerTransferability', 'Figures')
PANEL = ['TP', 'ALB', 'TB', 'DB', 'Ca', 'IP', 'CRE', 'UA', 'CRP', 'TC', 'HDL-C', 'LDL-C', 'TG']
ANALYZERS = ['#1', '#2', '#3']
HARMON_THRESH = 0.02      # penalty considered negligible if <2% (within natural 500-window drift)


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


def daily_volumes():
    """Per-item total samples/day over the 3-analyzer overlap (2022-06+), for the efficiency panel."""
    vol = {}
    for it in PANEL:
        df = pd.read_parquet(PROC, columns=['S_Date', 'SystemName'], filters=[('A_Name', '==', it)])
        d = pd.to_datetime(df['S_Date'])
        d = d[d >= '2022-06-02']
        if len(d) == 0:
            vol[it] = np.nan; continue
        days = max(1, (d.max() - d.min()).days)
        vol[it] = len(d) / days          # total (all analyzers) per day
    return vol


def figA(df, gray):
    srcs = ANALYZERS + ['pool']
    M = np.full((len(srcs), len(ANALYZERS)), np.nan)
    for i, s in enumerate(srcs):
        for j, b in enumerate(ANALYZERS):
            g = df[(df['train_src'] == s) & (df['test_analyzer'] == b)]
            if len(g):
                M[i, j] = g['flag_rate'].mean() * 100
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(M, cmap=('Greys' if gray else 'OrRd'), vmin=0, vmax=3, aspect='auto')
    ax.set_xticks(range(len(ANALYZERS))); ax.set_xticklabels([f'test {a}' for a in ANALYZERS])
    ax.set_yticks(range(len(srcs))); ax.set_yticklabels([f'train {s}' for s in srcs])
    for i in range(len(srcs)):
        for j in range(len(ANALYZERS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if (gray and M[i, j] > 1.8) else 'black')
    ax.set_title('Mean flag rate (%): train analyzer × test analyzer\n(diagonal = within; off-diag = cross transfer)'
                 + ('  [grayscale]' if gray else '  [color]'), fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='flag rate (%)')
    fig.tight_layout(); save(fig, 'figA_transfer_matrix', gray)


def figB(df, gray):
    cross = df[df['kind'] == 'cross']
    data = [cross[cross['item'] == it]['penalty'].values * 100 for it in PANEL]
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, tick_labels=PANEL, showfliers=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('0.8' if gray else '#9ecae1')
    ax.axhline(0, color='0.4', ls='-', lw=0.8)
    ax.axhline(HARMON_THRESH * 100, color='0.1' if gray else '#d62728', ls='--', lw=1.2, label='2% (harmonized threshold)')
    ax.set_ylabel('cross-analyzer transfer penalty (flag − within, %)')
    ax.set_xticklabels(PANEL, rotation=40, ha='right', fontsize=8)
    ax.set_title('Transfer penalty per item (0 = analyzers harmonized)'
                 + ('  [grayscale]' if gray else '  [color]'), fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); save(fig, 'figB_penalty_by_item', gray)


def figC(df, vol, gray):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # (a) within vs pool flag rate per item
    within = df[df['kind'] == 'within'].groupby('item')['flag_rate'].mean().reindex(PANEL) * 100
    pool = df[df['kind'] == 'pool'].groupby('item')['flag_rate'].mean().reindex(PANEL) * 100
    x = np.arange(len(PANEL)); w = 0.4
    ax1.bar(x - w / 2, within.values, w, label='within (single-analyzer)', color='0.5' if gray else '#4c72b0')
    ax1.bar(x + w / 2, pool.values, w, label='pool (3 analyzers)', color='0.75' if gray else '#dd8452')
    ax1.set_xticks(x); ax1.set_xticklabels(PANEL, rotation=40, ha='right', fontsize=8)
    ax1.set_ylabel('flag rate (%)'); ax1.axhline(0.5, color='0.6', ls=':', lw=1)
    ax1.set_title('(a) pool-trained vs single-analyzer flag rate', fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis='y')
    # (b) days to accrue 500 samples: single (=1/3 of total/day) vs pool (=total/day)
    days_single = np.array([500 / (vol[it] / 3) if vol.get(it) else np.nan for it in PANEL])
    days_pool = np.array([500 / vol[it] if vol.get(it) else np.nan for it in PANEL])
    ax2.bar(x - w / 2, days_single, w, label='single analyzer', color='0.5' if gray else '#4c72b0')
    ax2.bar(x + w / 2, days_pool, w, label='pooled (3 analyzers)', color='0.75' if gray else '#dd8452')
    ax2.set_xticks(x); ax2.set_xticklabels(PANEL, rotation=40, ha='right', fontsize=8)
    ax2.set_ylabel('days to accrue 500 samples'); ax2.set_yscale('log')
    ax2.set_title('(b) collection time to 500 (pool ≈ 1/3)', fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis='y')
    fig.suptitle('Pooling analyzers for training' + ('  [grayscale]' if gray else '  [color]'), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); save(fig, 'figC_pool_vs_single', gray)


def main():
    df = pd.read_csv(RESULTS)
    # summary
    print('=== mean flag rate by kind ===')
    print(df.groupby('kind')['flag_rate'].agg(['mean', 'median', 'max']).round(4).to_string())
    print('\n=== cross transfer penalty by item (mean, 95pct) ===')
    cr = df[df['kind'] == 'cross'].groupby('item')['penalty']
    summ = pd.DataFrame({'mean_penalty': cr.mean().round(4), 'p95_penalty': cr.quantile(0.95).round(4),
                         'frac_penalty_ge2pct': (df[df['kind'] == 'cross'].assign(g=df['penalty'] >= HARMON_THRESH)
                                                 .groupby('item')['g'].mean()).round(3)}).reindex(PANEL)
    print(summ.to_string())
    summ.to_csv(os.path.join(OUT, 'inter_analyzer_summary.csv'))
    vol = daily_volumes()
    for gray in (True, False):
        figA(df, gray); figB(df, gray); figC(df, vol, gray)
    print('DONE ->', FIGDIR)


if __name__ == '__main__':
    main()
