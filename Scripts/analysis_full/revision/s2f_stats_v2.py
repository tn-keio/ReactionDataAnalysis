# Stage 2f: assay-blocked rank tests on the v2 benchmark (manuscript section 3.6 / S6), N = 500.
#   * Friedman omnibus over the method set, blocks = the 21 assays
#   * Nemenyi all-pairs post-hoc on mean ranks (scikit-posthocs)
#   * assay-paired one-sided Wilcoxon signed-rank (maha > other) with Holm-Bonferroni correction
# Metrics: recall_main (k>=2), subtle k=1 (mean over A/B/C/E), recall_B_drift.
# The sample-level McNemar test of the submitted paper needs the per-sample list.csv, which the v2 run
# did not write (do_list=False); see the note printed at the end.
# Output: Results/20260917_Rev_v2Benchmark/stats_v2_{friedman,nemenyi,wilcoxon}.csv
# py3.14 (scipy + scikit-posthocs, no TF).

import os
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
V2 = os.path.join(BASE, 'Results', '20260917_Rev_v2Benchmark')
POS = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']
BASE_METHODS = ['maha', 'maha_robust', 'pca', 'ae', 'cnn1d', 'ae2d_and', 'ae2d_ens', 'twod_only',
                'lof', 'iforest', 'ocsvm']
N = 500


def load():
    recs = {}
    for g in ('main', 'ca', 'm2'):
        for l in open(os.path.join(V2, g, 'Reports', 'comparison.jsonl'), encoding='utf-8'):
            if l.strip():
                d = json.loads(l)
                recs[(d['item'], d['method'], d['N'])] = d
    zsel = pd.read_csv(os.path.join(V2, 'dae_z_selection.csv'))
    for _, r in zsel[zsel.N == N].iterrows():
        for tag, col in (('dae_elbow', 'z_elbow'), ('dae_oracle', 'z_oracle')):
            src = recs.get((r['item'], f"dae{int(r[col])}", N))
            if src is not None:
                recs[(r['item'], tag, N)] = src
    return recs


def value(rec, field):
    b = rec['budgets']['99.5']
    if field == 'k1':
        return float(np.nanmean([b[f'recall_{t}_k1']['mean'] for t in POS if f'recall_{t}_k1' in b]))
    return b[field]['mean'] if field in b else np.nan


def matrix(recs, methods, field):
    items = sorted({k[0] for k in recs})
    rows = []
    for it in items:
        if all((it, m, N) in recs for m in methods):
            rows.append([value(recs[(it, m, N)], field) for m in methods])
    return np.array(rows), items


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # Pre-specified family: the 11 submitted detectors + ONE representative compact AE (the convolutional,
    # validation-configured cae8).  Adding every compact-AE variant inflates the Nemenyi family size and
    # costs power, so the 15-method version is reported only as a sensitivity analysis (--family all).
    ap.add_argument('--family', choices=['prespecified', 'all'], default='prespecified')
    a = ap.parse_args()
    recs = load()
    have = {k[1] for k in recs if k[2] == N}
    extra = ('cae8',) if a.family == 'prespecified' else ('cae2', 'cae8', 'dae_elbow', 'dae_oracle')
    methods = BASE_METHODS + [m for m in extra if m in have]
    suffix = '' if a.family == 'prespecified' else '_allvariants'
    out_f, out_w, out_n = [], [], []
    for field, label in (('recall_main', 'recall_main(k>=2)'), ('k1', 'subtle k=1'), ('recall_B_drift', 'B_drift')):
        X, items = matrix(recs, methods, field)
        chi, p = friedmanchisquare(*X.T)
        mr = rankdata(-X, axis=1).mean(0)          # rank 1 = best; report mean rank on the 1..M scale
        order = np.argsort(mr)
        out_f.append(dict(metric=label, n_items=len(items), n_methods=len(methods), chi2=chi, p=p,
                          best_method=methods[order[0]], best_mean_rank=mr[order[0]]))
        print(f'\n=== {label}: Friedman chi2={chi:.1f}, p={p:.3g}, {len(items)} assays x {len(methods)} methods ===')
        print('  mean rank (1 = best): ' + ', '.join(f'{methods[i]} {mr[i]:.2f}' for i in order))
        for i, m in enumerate(methods):
            out_n.append(dict(metric=label, method=m, mean_rank=mr[i], mean_value=X[:, i].mean()))
        # Nemenyi
        try:
            import scikit_posthocs as sp
            nem = sp.posthoc_nemenyi_friedman(X)
            nem.columns = nem.index = methods
            nem.to_csv(os.path.join(V2, f'stats_v2_nemenyi_{field}{suffix}.csv'))
            print('  Nemenyi vs maha: ' + ', '.join(f'{m} p={nem.loc["maha", m]:.3g}' for m in methods if m != 'maha'))
        except Exception as e:
            print('  Nemenyi skipped:', e)
        # paired Wilcoxon, maha vs each other method, one-sided, Holm-corrected
        mv = X[:, methods.index('maha')]
        ps, rows = [], []
        for m in methods:
            if m == 'maha':
                continue
            ov = X[:, methods.index(m)]
            d = mv - ov
            if np.allclose(d, 0):
                stat, p1 = np.nan, 1.0
            else:
                res = wilcoxon(mv, ov, alternative='greater', zero_method='wilcox')
                stat, p1 = res.statistic, res.pvalue
            ps.append(p1)
            rows.append(dict(metric=label, comparison=f'maha>{m}', median_delta=float(np.median(d)),
                             n_better=int((d > 0).sum()), n_worse=int((d < 0).sum()), W=stat, p_raw=p1))
        holm = multipletests(ps, method='holm')[1]
        for r, ph in zip(rows, holm):
            r['p_holm'] = ph
            out_w.append(r)
            print(f"  {r['comparison']:22} Δmedian={r['median_delta']:+.3f} ({r['n_better']}>/{r['n_worse']}<) "
                  f"p_raw={r['p_raw']:.3g} p_holm={ph:.3g}")
    pd.DataFrame(out_f).to_csv(os.path.join(V2, f'stats_v2_friedman{suffix}.csv'), index=False)
    pd.DataFrame(out_n).to_csv(os.path.join(V2, f'stats_v2_meanranks{suffix}.csv'), index=False)
    pd.DataFrame(out_w).to_csv(os.path.join(V2, f'stats_v2_wilcoxon{suffix}.csv'), index=False)
    print(f'\nmethod family: {a.family} ({len(methods)} methods). Sample-level McNemar / bootstrap: see s2g_sample_level_tests.py.')


if __name__ == '__main__':
    main()
