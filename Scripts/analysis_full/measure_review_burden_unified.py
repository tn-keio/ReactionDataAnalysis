# Review burden measured under ONE protocol, for every item, over the whole record.
#
# The first estimate (make_review_burden_table.py) reused the method comparison's per-item flag
# rates. Those are each measured on a different 11-13 day window early in that item's record, with a
# threshold calibrated on a holdout block -- so the spread between items reflected which window each
# happened to land in as much as the assay itself (Results/20260720_ALB_FlagRate_Investigation).
#
# Here every item is measured the same way, and the way the paper actually recommends deploying:
#
#   rolling recent-N : each block of N curves is scored by an empirical-covariance Mahalanobis model
#                      fitted on the N curves immediately before it, threshold self-calibrated as the
#                      P99.5 of those N training scores (no holdout -- deployment has none)
#   whole record     : 2021-04 .. 2025-12, all three analyzers pooled, real curves only
#   tests/day        : N_total / collection span in CALENDAR days (Table S1)
#
# N=500 is the paper's operating point; N=5000 is reported as a sensitivity because the excess over
# the 0.5% nominal budget is dominated by estimation noise in a 500-sample fit of a 34-dim covariance.
#
# Window/channel/estimator follow the method comparison: points 5-38, Calc = RAM - RAS, scalar
# min-max fitted on the training window, empirical covariance.

import gc
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
OUT = os.path.join(BASE, 'Results', '20260720_ReviewBurden_UnifiedProtocol')

# Per-item curve counts. The original lives in the manuscript repository under
# supplementary/tables/. Point PAPER_REPO_DIR at that checkout, or drop a copy of
# the CSV into this repository's Data/ directory.
_paper_repo = os.environ.get('PAPER_REPO_DIR')
COUNTS = (os.path.join(_paper_repo, 'supplementary', 'tables', 'item_counts_ForPaper21.csv')
          if _paper_repo
          else os.path.join(_REPO_ROOT, 'Data', 'item_counts_ForPaper21.csv'))

P0 = 5
RAM = [f'RAM{i}' for i in range(P0, 39)]
RAS = [f'RAS{i}' for i in range(P0, 39)]
BUDGET = 99.5
N_GRID = [500, 5000]
MIN_MONTH_N = 100

# (paper item, A_Name) -- the 13-item method-comparison panel.
# A_Name is the dataset partition value; the comparison's second tuple element is the SYNTHETIC
# battery name (Ca -> 'CA'), which is not what the partition is keyed on.
PANEL = [('TP', 'TP'), ('ALB', 'ALB'), ('TB', 'TB'), ('DB', 'DB'), ('Ca', 'Ca'), ('IP', 'IP'),
         ('CRE', 'CRE'), ('UA', 'UA'), ('Fe', 'Fe'), ('Cu', 'Cu'), ('Zn', 'Zn'), ('GLU', 'GLU'),
         ('CRP', 'CRP')]

# per-item rates from the previous, window-based estimate (supplementary Table S2/S3)
OLD_RATE = {'ALB': 1.67, 'CRE': 0.47, 'UA': 0.67, 'Ca': 0.73, 'TP': 0.40, 'TB': 0.37, 'CRP': 0.30,
            'IP': 0.63, 'DB': 0.53, 'Fe': 0.30, 'GLU': 0.37, 'Cu': 0.73, 'Zn': 0.33}

BLUE, ORANGE, VERM, INK, GRAY = '#0072B2', '#E69F00', '#D55E00', '#222222', '#7F7F7F'


def norm_fit(a):
    return float(a.min()), float(a.max())


def apply_norm(a, lo, hi):
    return (a - lo) / (hi - lo + 1e-8)      # affine, no clipping


def maha(loc, cov, X):
    P = np.linalg.pinv(cov)
    d = X - loc
    return np.sum(d @ P * d, axis=1)


def load(a_name):
    d = ds.dataset(PROC, partitioning='hive')
    tb = d.to_table(columns=['S_Date'] + RAM + RAS,
                    filter=pc.field('A_Name') == a_name).to_pandas()
    tb['S_Date'] = pd.to_datetime(tb['S_Date'])
    tb = tb.sort_values('S_Date').reset_index(drop=True)
    calc = tb[RAM].values.astype('float64') - tb[RAS].values.astype('float64')
    dates = pd.to_datetime(tb['S_Date'].values)
    del tb
    gc.collect()
    return dates, calc


def rolling_flags(calc, n_train):
    """Deployment scheme: score each block of n_train with a model fitted on the preceding block."""
    n = len(calc)
    flag = np.full(n, np.nan)
    for start in range(n_train, n, n_train):
        stop = min(start + n_train, n)
        train = calc[start - n_train:start]
        lo, hi = norm_fit(train)
        W = apply_norm(train, lo, hi)
        e = EmpiricalCovariance().fit(W)
        thr = np.percentile(maha(e.location_, e.covariance_, W), BUDGET)
        flag[start:stop] = maha(e.location_, e.covariance_,
                                apply_norm(calc[start:stop], lo, hi)) >= thr
    return flag


def volumes():
    c = pd.read_csv(COUNTS)
    c['key'] = c['item(paper)'].str.upper()
    out = {}
    for _, r in c.iterrows():
        days = (pd.to_datetime(r['date_max']) - pd.to_datetime(r['date_min'])).days
        out[r['key']] = dict(n_total=int(r['N_total']), days=int(days),
                             per_day=r['N_total'] / days)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    vol = volumes()
    rows, monthly = [], []

    for item, a_name in PANEL:
        dates, calc = load(a_name)
        v = vol[item.upper()]
        rec = dict(item=item, n_curves=len(calc), n_total=v['n_total'],
                   collection_days=v['days'], tests_per_calendar_day=v['per_day'],
                   old_rate_pct=OLD_RATE[item])
        for N in N_GRID:
            if len(calc) < 3 * N:
                rec[f'rate_N{N}_pct'] = np.nan
                continue
            f = rolling_flags(calc, N)
            ok = ~np.isnan(f)
            rec[f'rate_N{N}_pct'] = 100 * float(f[ok].mean())
            if N == N_GRID[0]:
                df = pd.DataFrame({'month': [str(m) for m in np.asarray(
                    dates.to_period('M'))[ok]], 'flag': f[ok]})
                g = df.groupby('month').agg(n=('flag', 'size'),
                                            rate_pct=('flag', lambda x: 100 * x.mean())).reset_index()
                g = g[g['n'] >= MIN_MONTH_N]
                g['item'] = item
                monthly.append(g)
        rec['flags_per_day_N500'] = rec['rate_N500_pct'] / 100 * v['per_day']
        rec['flags_per_day_N5000'] = rec['rate_N5000_pct'] / 100 * v['per_day']
        rec['old_flags_per_day'] = OLD_RATE[item] / 100 * v['per_day']
        rows.append(rec)
        print(f"  {item:4s} n={len(calc):>9,}  N500 {rec['rate_N500_pct']:.2f}%  "
              f"N5000 {rec['rate_N5000_pct']:.2f}%  (old {OLD_RATE[item]:.2f}%)  "
              f"-> {rec['flags_per_day_N500']:.1f}/day", flush=True)
        del dates, calc
        gc.collect()

    df = pd.DataFrame(rows).sort_values('flags_per_day_N500', ascending=False)
    df.to_csv(os.path.join(OUT, 'review_burden_unified.csv'), index=False, float_format='%.4f')
    mon = pd.concat(monthly)
    mon.to_csv(os.path.join(OUT, 'monthly_flagrate_by_item.csv'), index=False, float_format='%.4f')

    tot_tests = df['tests_per_calendar_day'].sum()
    for tag, col in [('N=500', 'flags_per_day_N500'), ('N=5000', 'flags_per_day_N5000'),
                     ('previous (window-based)', 'old_flags_per_day')]:
        t = df[col].sum()
        print(f"\n{tag:26s} total {t:6.1f} flags/calendar-day  ({100 * t / tot_tests:.3f}% of "
              f"{tot_tests:,.0f} tests/day)")

    # ---- Markdown table ----
    lines = ['| Item | Flag rate (%) | Tests/day | Flagged/day |', '| :-- | --: | --: | --: |']
    for _, r in df.iterrows():
        lines.append(f"| {r['item']} | {r['rate_N500_pct']:.2f} | "
                     f"{r['tests_per_calendar_day']:,.0f} | {r['flags_per_day_N500']:.1f} |")
    tf = df['flags_per_day_N500'].sum()
    lines.append(f"| **Total (13)** | **{100 * tf / tot_tests:.2f}** | **{tot_tests:,.0f}** | "
                 f"**{tf:.1f}** |")
    table = '\n'.join(lines)
    with open(os.path.join(OUT, 'TableS_review_burden.md'), 'w', encoding='utf-8') as f:
        f.write(table + '\n')
    print('\n' + table)

    # ---- figure: per-item monthly distribution, old vs new ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.5, 1]})
    order = list(df['item'])
    data = [mon[mon['item'] == it]['rate_pct'].values for it in order]
    bp = a1.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True)
    for p in bp['boxes']:
        p.set_facecolor('#A6C8E0'); p.set_edgecolor(INK)
    for med in bp['medians']:
        med.set_color(VERM); med.set_linewidth(2)
    a1.scatter(range(1, len(order) + 1), [OLD_RATE[i] for i in order], marker='v', s=55,
               color=INK, zorder=5, label='previous window-based estimate')
    a1.axhline(0.5, color=GRAY, ls=':', lw=1.2, label='0.5% nominal budget')
    a1.set_ylabel('monthly flag rate (%)'); a1.set_xlabel('item')
    a1.set_title('Monthly flag rate over 2021-2025 under one protocol (rolling recent-500)',
                 fontsize=10, loc='left')
    a1.legend(fontsize=8); a1.grid(alpha=0.3, axis='y')

    x = np.arange(len(order)); w = 0.38
    a2.barh(x - w / 2, [df.set_index('item').loc[i, 'old_flags_per_day'] for i in order], w,
            color='#A6C8E0', edgecolor=INK, linewidth=0.5, label='previous')
    a2.barh(x + w / 2, [df.set_index('item').loc[i, 'flags_per_day_N500'] for i in order], w,
            color=VERM, edgecolor=INK, linewidth=0.5, label='unified protocol')
    a2.set_yticks(x); a2.set_yticklabels(order); a2.invert_yaxis()
    a2.set_xlabel('flagged curves per calendar day')
    a2.set_title('Review burden by item', fontsize=10, loc='left')
    a2.legend(fontsize=8); a2.grid(alpha=0.3, axis='x')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_review_burden_unified.png'), dpi=600, facecolor='white')
    plt.close(fig)
    print(f'\nDONE -> {OUT}')


if __name__ == '__main__':
    main()
