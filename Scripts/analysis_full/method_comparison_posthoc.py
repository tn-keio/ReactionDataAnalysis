# Non-parametric omnibus + post-hoc multiple comparison for the 5-38 / 21-item method comparison.
# Design = randomized block (21 items measured by all 13 methods, repeated measures) -> analysed as a
# repeated-measures (matched) design.
#   omnibus  : Friedman test (per metric)
#   post-hoc : Nemenyi test (all-pairwise, based on the Friedman ranks; design-matched for blocked data)
#   worst-item uncertainty: item-block bootstrap of the mean and the minimum over items (B = 2000)
# (An independent-groups test such as Steel-Dwass is NOT appropriate here because it ignores the item
#  pairing; the item-paired Wilcoxon signed-rank + Holm in method_comparison_stat_tests.py complements
#  Nemenyi with higher per-pair power.)
# Focus: Mahalanobis(E) vs each of the 12 other methods, for recall_main / k=1(subtle) / B_drift.
# Inputs: comparison.jsonl of the main run (run_method_comparison_p5_38.py) and of the clean Ca / ALB / IP
#   re-runs, plus comparison_dae_elbow.jsonl written into each folder by select_compact_ae.py.
# py3.14 + scipy + scikit-posthocs.  No TF.
import os, json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

BASE = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
MAIN = os.path.join(BASE, 'Results', '20260717_MethodComparison_p5_38')
CACLEAN = os.path.join(BASE, 'Results', '20260718_MethodComparison_Ca_clean')
M2CLEAN = os.path.join(BASE, 'Results', '20260718_MethodComparison_M2_clean')
PANEL = ['TP', 'ALB', 'TB', 'DB', 'Ca', 'IP', 'CRE', 'UA', 'CRP', 'Fe', 'Cu', 'Zn', 'GLU',
         'TG', 'LDL-C', 'HDL-C', 'TC', 'IgG', 'KL-6', 'IgA', 'IgM']                   # 21-item panel
CLEAN_SRC = {'Ca': CACLEAN, 'ALB': M2CLEAN, 'IP': M2CLEAN}
METHODS = ['maha', 'maha_robust', 'pca', 'ae', 'cnn1d', 'ae2d_and', 'ae2d_ens', 'twod_only', 'lof', 'iforest', 'ocsvm',
           'dae_elbow', 'cae8']
LABEL = {'maha': 'Mahalanobis(E)', 'maha_robust': 'Mahalanobis(R)', 'pca': 'PCA', 'ae': '1D-AE',
         'cnn1d': '1D-CNN→iForest', 'ae2d_and': 'AE∧2D-CNN', 'ae2d_ens': 'AE⊕2D-CNN',
         'twod_only': '2D-CNN→iForest', 'lof': 'LOF', 'iforest': 'Isolation Forest', 'ocsvm': 'One-Class SVM',
         'dae_elbow': 'Compact AE (dense)', 'cae8': 'Compact AE (conv.)'}
POS = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']
N = 500


def _jl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()] if os.path.exists(path) else []


def load_key():
    """(item, method, N) -> record.  Ca / ALB / IP come from the clean re-runs, all other items from the
    main run; 'dae_elbow' is the dense compact AE with the z chosen by select_compact_ae.py."""
    key = {}
    for d in (MAIN, CACLEAN, M2CLEAN):
        for f in ('comparison.jsonl', 'comparison_dae_elbow.jsonl'):
            for r in _jl(os.path.join(d, 'Reports', f)):
                if CLEAN_SRC.get(r['item'], MAIN) == d:
                    key[(r['item'], r['method'], r['N'])] = r
    miss = [(it, m) for it in PANEL for m in METHODS if (it, m, N) not in key]
    assert not miss, f'missing results (run select_compact_ae.py for dae_elbow): {miss[:5]} ...'
    return key


def matrix(key, field):
    """rows = items (21, sorted), cols = methods (13).  value = metric."""
    items = sorted(PANEL)
    M = np.zeros((len(items), len(METHODS)))
    for i, it in enumerate(items):
        for j, m in enumerate(METHODS):
            b = key[(it, m, N)]['budgets']['99.5']
            if field == 'k1':
                M[i, j] = np.nanmean([b[f'recall_{t}_k1']['mean'] for t in POS if f'recall_{t}_k1' in b])
            else:
                M[i, j] = b[field]['mean']
    return pd.DataFrame(M, index=items, columns=METHODS)


def analyze(key, field, name):
    df = matrix(key, field)
    print(f'\n================ {name} ================')
    # omnibus
    chi2, p = friedmanchisquare(*[df[m].values for m in METHODS])
    print(f'Friedman omnibus: chi2={chi2:.2f}, p={p:.3g}')
    # mean ranks (higher value -> higher rank; rank 13 = best)
    ranks = df.rank(axis=1, method='average')
    mr = ranks.mean(0).sort_values(ascending=False)
    print('mean ranks (best first):', ', '.join(f'{LABEL[m]}={v:.2f}' for m, v in mr.items()))
    # post-hoc Nemenyi (design-matched, based on Friedman ranks) — p-value matrix
    nem = sp.posthoc_nemenyi_friedman(df.values)
    nem.index = nem.columns = METHODS
    nem.to_csv(os.path.join(MAIN, f'posthoc_nemenyi_{field}.csv'))
    print(f'{"maha vs":22} {"Δmean":>7} {"Δrank":>6} {"Nemenyi p":>11}')
    rows = []
    for m in METHODS:
        if m == 'maha':
            continue
        dmean = df['maha'].mean() - df[m].mean()
        pn = nem.loc['maha', m]
        sig_n = '***' if pn < 0.001 else '**' if pn < 0.01 else '*' if pn < 0.05 else 'ns'
        print(f'{LABEL[m]:22} {dmean:+7.3f} {mr["maha"] - mr[m]:+6.2f} {pn:11.4g} {sig_n:>3}')
        rows.append(dict(metric=name, comparison=f'maha_vs_{m}', delta_mean=round(dmean, 3),
                         mean_rank=round(mr[m], 2), maha_mean_rank=round(mr['maha'], 2),
                         friedman_chi2=chi2, friedman_p=p, nemenyi_p=pn))
    return rows


def worstitem_ci(key, B=2000, seed=20260726):
    """Item-block bootstrap of mean and worst-item recall (k >= 2): resample the 21 items with replacement."""
    df = matrix(key, 'recall_main')
    rng = np.random.default_rng(seed)
    rows = []
    for m in METHODS:
        v = df[m].values
        idx = rng.integers(0, len(v), size=(B, len(v)))
        bmin, bmean = v[idx].min(axis=1), v[idx].mean(axis=1)
        rows.append(dict(method=m, mean=v.mean(), mean_lo=np.percentile(bmean, 2.5), mean_hi=np.percentile(bmean, 97.5),
                         worst=v.min(), worst_item=df.index[int(v.argmin())],
                         worst_lo=np.percentile(bmin, 2.5), worst_hi=np.percentile(bmin, 97.5)))
    out = pd.DataFrame(rows).sort_values('mean', ascending=False)
    print(f'\n================ item-block bootstrap (k>=2, B={B}) ================')
    print(out.round(3).to_string(index=False))
    return out


def main():
    key = load_key()
    allrows = []
    for field, name in [('recall_main', 'recall_main (k>=2)'), ('k1', 'recall k=1 (subtle)'), ('recall_B_drift', 'recall B_drift')]:
        allrows += analyze(key, field, name)
    out = os.path.join(MAIN, 'posthoc_friedman_nemenyi.csv')
    pd.DataFrame(allrows).to_csv(out, index=False)
    worstitem_ci(key).to_csv(os.path.join(MAIN, 'worstitem_ci.csv'), index=False)
    print('\nDesign note: data are a randomized BLOCK design (items = blocks); Nemenyi (Friedman-based) is the')
    print('design-matched post-hoc. Independent-group tests (e.g. Steel-Dwass) are not appropriate here.')
    print('saved', out, 'and worstitem_ci.csv')


if __name__ == '__main__':
    main()
