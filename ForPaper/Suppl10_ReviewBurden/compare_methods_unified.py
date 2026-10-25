# Do all detectors really share the same review budget? Measured under ONE protocol.
#
# The supplementary claims the eleven detectors operate at nearly the same review budget because the
# threshold is self-calibrated to the same percentile for every one of them. That claim currently
# rests on the method-comparison protocol (one 11-13 day window per item, threshold taken from a
# holdout block), while the headline burden in Table S3 is measured under the deployment protocol
# (rolling recent-500, self-calibrated, whole record). Stating both in the same subsection invites
# the obvious question of which applies.
#
# The claim is not guaranteed to survive the change of protocol. The excess over the 0.5% nominal
# budget comes from estimation noise in a 500-sample fit, and the methods differ enormously in how
# many parameters they fit: empirical Mahalanobis estimates 34 means + 595 covariance entries = 629
# parameters from 500 curves, whereas PCA at 95% variance keeps a handful of components. If the
# excess scales with parameter count, Mahalanobis would pay the most and the spread would widen.
#
# So: the six classical detectors, same rolling protocol, same self-calibrated P99.5, same anchors,
# whole record. The five deep-learning detectors are excluded here -- at ~34 s per fit they would
# need ~8 h, and this run decides whether that is necessary.
#
# Estimator definitions are copied verbatim from run_paper_method_comparison.py.

import gc
import os
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
OUT = os.path.join(BASE, 'Results', '20260720_MethodBudget_UnifiedProtocol')
COUNTS = os.path.join(BASE, 'Data', 'item_counts_ForPaper21.csv')

P0 = 5
RAM = [f'RAM{i}' for i in range(P0, 39)]
RAS = [f'RAS{i}' for i in range(P0, 39)]
N_GRID = [500, 5000]                # 500 = the paper's operating point; 5000 = the larger-window arm
BUDGET = 99.5
MAX_ANCHORS = 57                    # one per month of the record, spread evenly, non-overlapping
SEED = 0

PANEL = [('TP', 'TP'), ('ALB', 'ALB'), ('TB', 'TB'), ('DB', 'DB'), ('Ca', 'Ca'), ('IP', 'IP'),
         ('CRE', 'CRE'), ('UA', 'UA'), ('Fe', 'Fe'), ('Cu', 'Cu'), ('Zn', 'Zn'), ('GLU', 'GLU'),
         ('CRP', 'CRP')]

METHODS = ['maha', 'maha_robust', 'pca', 'lof', 'iforest', 'ocsvm']
LABEL = {'maha': 'Mahalanobis (empirical)', 'maha_robust': 'Mahalanobis (robust)', 'pca': 'PCA',
         'lof': 'LOF', 'iforest': 'Isolation Forest', 'ocsvm': 'One-Class SVM'}
# mean recall over the 13-item panel at each training size, from the method comparison (§3.2)
RECALL = {500:  {'maha': 0.995, 'maha_robust': 0.957, 'pca': 0.760, 'lof': 0.665,
                 'iforest': 0.147, 'ocsvm': 0.156},
          5000: {'maha': 0.995, 'maha_robust': 0.957, 'pca': 0.757, 'lof': 0.939,
                 'iforest': 0.153, 'ocsvm': 0.155}}

BLUE, ORANGE, VERM, INK, GRAY = '#0072B2', '#E69F00', '#D55E00', '#222222', '#7F7F7F'


def norm_fit(a):
    return float(a.min()), float(a.max())


def apply_norm(a, lo, hi):
    return (a - lo) / (hi - lo + 1e-8)      # affine, no clipping


def mcd_like(D, keep=0.95, iters=5):
    loc = D.mean(0); cov = np.cov(D.T); n = int(len(D) * keep)
    for _ in range(iters):
        d = np.sum((D - loc) @ np.linalg.pinv(cov) * (D - loc), axis=1)
        core = D[np.argsort(d)[:n]]; loc, cov = core.mean(0), np.cov(core.T)
    return loc, cov


def maha_scores(loc, cov, X):
    P = np.linalg.pinv(cov)
    return np.sum((X - loc) @ P * (X - loc), axis=1)


def scorer(method, Xtr):
    """Fit on Xtr, return a scoring function. Verbatim from the method comparison."""
    if method == 'maha':
        e = EmpiricalCovariance().fit(Xtr)
        loc, cov = e.location_, e.covariance_
        return lambda X: maha_scores(loc, cov, X), None
    if method == 'maha_robust':
        loc, cov = mcd_like(Xtr, 0.95)
        return lambda X: maha_scores(loc, cov, X), None
    if method == 'pca':
        m = PCA(n_components=0.95).fit(Xtr)
        return (lambda X: np.mean((X - m.inverse_transform(m.transform(X))) ** 2, axis=1),
                int(m.n_components_))
    if method == 'lof':
        m = LocalOutlierFactor(novelty=True).fit(Xtr)
        return lambda X: -m.decision_function(X), None
    if method == 'iforest':
        m = IsolationForest(n_estimators=100, random_state=SEED, n_jobs=-1).fit(Xtr)
        return lambda X: -m.decision_function(X), None
    if method == 'ocsvm':
        m = OneClassSVM(gamma='scale').fit(Xtr)
        return lambda X: -m.decision_function(X), None
    raise ValueError(method)


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


def anchors_for(n, n_train):
    """Evenly spaced, non-overlapping training/scoring block starts across the whole record."""
    usable = n - 2 * n_train
    if usable <= 0:
        return []
    step = max(n_train, usable // MAX_ANCHORS)
    return list(range(n_train, n - n_train, step))[:MAX_ANCHORS]


def volumes():
    c = pd.read_csv(COUNTS)
    c['key'] = c['item(paper)'].str.upper()
    return {r['key']: r['N_total'] / (pd.to_datetime(r['date_max']) -
                                      pd.to_datetime(r['date_min'])).days
            for _, r in c.iterrows()}


def main():
    os.makedirs(OUT, exist_ok=True)
    vol = volumes()
    rows, ncomp = [], []
    t_all = time.time()

    for item, a_name in PANEL:
        dates, calc = load(a_name)
        v = vol[item.upper()]
        for n_train in N_GRID:
            anc = anchors_for(len(calc), n_train)
            if not anc:
                print(f'  {item:4s} N={n_train}: too few curves - skipped', flush=True)
                continue
            t0 = time.time()
            flagged = {m: 0 for m in METHODS}
            scored = 0
            for s in anc:
                train = calc[s - n_train:s]
                block = calc[s:s + n_train]
                lo, hi = norm_fit(train)
                W, S = apply_norm(train, lo, hi), apply_norm(block, lo, hi)
                for m in METHODS:
                    f, nc = scorer(m, W)
                    thr = np.percentile(f(W), BUDGET)      # self-calibrated, no holdout
                    flagged[m] += int((f(S) >= thr).sum())
                    if m == 'pca' and nc is not None and n_train == N_GRID[0]:
                        ncomp.append(nc)
                scored += len(block)
            for m in METHODS:
                rate = flagged[m] / scored
                rows.append(dict(item=item, n_train=n_train, method=m, n_anchors=len(anc),
                                 n_scored=scored, flag_rate_pct=100 * rate,
                                 tests_per_calendar_day=v, flags_per_day=rate * v,
                                 window_days=n_train / v))
            print(f'  {item:4s} N={n_train:<5d} anchors={len(anc):3d} scored={scored:>8,} '
                  f'window={n_train / v:6.1f}d ({time.time() - t0:5.1f}s)  ' +
                  '  '.join(f'{m}={100 * flagged[m] / scored:.2f}%' for m in METHODS), flush=True)
        del dates, calc
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'method_budget_unified.csv'), index=False, float_format='%.4f')

    agg = df.groupby(['n_train', 'method']).agg(flags_per_day=('flags_per_day', 'sum'),
                                                n_scored=('n_scored', 'sum')).reset_index()
    tot_tests = df.groupby('item')['tests_per_calendar_day'].first().sum()
    agg['rate_pct'] = 100 * agg['flags_per_day'] / tot_tests
    agg['recall'] = [RECALL[n][m] for n, m in zip(agg['n_train'], agg['method'])]
    agg.to_csv(os.path.join(OUT, 'method_budget_summary.csv'), index=False, float_format='%.4f')

    for N in N_GRID:
        a = agg[agg['n_train'] == N].sort_values('flags_per_day', ascending=False)
        print(f"\n--- N={N} ---\n{'method':26s} {'flags/day':>10s} {'rate':>8s} {'recall':>8s}")
        for _, r in a.iterrows():
            print(f"{LABEL[r['method']]:26s} {r['flags_per_day']:10.1f} {r['rate_pct']:7.3f}% "
                  f"{r['recall']:8.3f}")
        lo_, hi_ = a['flags_per_day'].min(), a['flags_per_day'].max()
        print(f'  spread {lo_:.1f}-{hi_:.1f} flags/day ({hi_ / lo_:.2f}x)')
    print(f'\nPCA components retained at N=500: median {int(np.median(ncomp))}')
    print(f'total elapsed {time.time() - t_all:.0f}s')

    # ---- figure: how the larger training window moves each detector ----
    p = {(r['n_train'], r['method']): r for _, r in agg.iterrows()}
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for m in METHODS:
        a, b = p[(N_GRID[0], m)], p[(N_GRID[1], m)]
        ax.annotate('', xy=(b['flags_per_day'], b['recall']),
                    xytext=(a['flags_per_day'], a['recall']),
                    arrowprops=dict(arrowstyle='-|>', color=GRAY, lw=1.3,
                                    shrinkA=7, shrinkB=7, alpha=0.9))
        ax.scatter(a['flags_per_day'], a['recall'], s=95, color=VERM,
                   edgecolor=INK, linewidth=0.6, zorder=5)
        ax.scatter(b['flags_per_day'], b['recall'], s=95, color=BLUE, marker='s',
                   edgecolor=INK, linewidth=0.6, zorder=5)
        ax.annotate(LABEL[m], (a['flags_per_day'], a['recall']),
                    textcoords='offset points', xytext=(9, 6), fontsize=8.5, color=INK)
    ax.scatter([], [], s=95, color=VERM, edgecolor=INK, label=f'N = {N_GRID[0]} (operating point)')
    ax.scatter([], [], s=95, color=BLUE, marker='s', edgecolor=INK, label=f'N = {N_GRID[1]}')
    ax.set_xlabel('review burden (flagged curves per calendar day, 13-item panel)')
    ax.set_ylabel('mean recall at the operating point')
    ax.set_ylim(0, 1.08)
    ax.set_title('A larger training window cuts the review burden of every detector\n'
                 'rolling recent-N, self-calibrated P99.5, whole record; arrow = N 500 to 5,000',
                 fontsize=10.5, loc='left')
    ax.legend(fontsize=9, loc='center right'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_method_budget_vs_recall.png'), dpi=600, facecolor='white')
    plt.close(fig)
    print(f'DONE -> {OUT}')


if __name__ == '__main__':
    main()
