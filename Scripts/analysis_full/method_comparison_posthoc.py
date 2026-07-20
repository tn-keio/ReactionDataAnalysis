# Non-parametric omnibus + post-hoc multiple comparison for the 5-38 / 14-item method comparison.
# Design = randomized block (14 items measured by all 11 methods, repeated measures) -> analysed as a
# repeated-measures (matched) design.
#   omnibus  : Friedman test (per metric)
#   post-hoc : Nemenyi test (all-pairwise, based on the Friedman ranks; design-matched for blocked data)
# (An independent-groups test such as Steel-Dwass is NOT appropriate here because it ignores the item
#  pairing; the item-paired Wilcoxon signed-rank + Holm in method_comparison_stat_tests.py complements
#  Nemenyi with higher per-pair power.)
# Focus: Mahalanobis(E) vs each of the 10 other methods, for recall_main / k=1(subtle) / B_drift.
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
PANEL = ['TP', 'ALB', 'TB', 'DB', 'Ca', 'IP', 'CRE', 'UA', 'Fe', 'Cu', 'Zn', 'GLU', 'GA', 'CRP']
METHODS = ['maha', 'maha_robust', 'pca', 'ae', 'cnn1d', 'ae2d_and', 'ae2d_ens', 'twod_only', 'lof', 'iforest', 'ocsvm']
LABEL = {'maha': 'Mahalanobis(E)', 'maha_robust': 'Mahalanobis(R)', 'pca': 'PCA', 'ae': '1D-AE',
         'cnn1d': '1D-CNN→iForest', 'ae2d_and': 'AE∧2D-CNN', 'ae2d_ens': 'AE⊕2D-CNN',
         'twod_only': '2D-CNN→iForest', 'lof': 'LOF', 'iforest': 'Isolation Forest', 'ocsvm': 'One-Class SVM'}
POS = {'A_spike', 'B_drift', 'C_hump', 'E_baseline'}


def load_key():
    recs = [r for r in (json.loads(l) for l in open(os.path.join(MAIN, 'Reports', 'comparison.jsonl'), encoding='utf-8'))
            if r['item'] not in ('Ca', 'ALB', 'IP')]
    for f in [os.path.join(CACLEAN, 'Reports', 'comparison.jsonl'), os.path.join(M2CLEAN, 'Reports', 'comparison.jsonl')]:
        recs += [r for r in (json.loads(l) for l in open(f, encoding='utf-8')) if r['item'] in ('Ca', 'ALB', 'IP')]
    return {(r['item'], r['method'], r['N']): r for r in recs}


def matrix(key, field):
    """rows = items (14), cols = methods (11).  value = metric."""
    M = np.zeros((len(PANEL), len(METHODS)))
    for i, it in enumerate(PANEL):
        for j, m in enumerate(METHODS):
            b = key[(it, m, 500)]['budgets']['99.5']
            if field == 'k1':
                M[i, j] = np.nanmean([b.get(f'recall_{t}_k1', {'mean': np.nan})['mean'] for t in POS])
            else:
                M[i, j] = b[field]['mean']
    return pd.DataFrame(M, index=PANEL, columns=METHODS)


def analyze(key, field, name):
    df = matrix(key, field)
    print(f'\n================ {name} ================')
    # omnibus
    chi2, p = friedmanchisquare(*[df[m].values for m in METHODS])
    print(f'Friedman omnibus: chi2={chi2:.2f}, p={p:.3g}')
    # mean ranks (higher value -> higher rank; rank 11 = best)
    ranks = df.rank(axis=1, method='average')
    mr = ranks.mean(0).sort_values(ascending=False)
    print('mean ranks (best first):', ', '.join(f'{LABEL[m]}={v:.1f}' for m, v in mr.items()))
    # post-hoc Nemenyi (design-matched, based on Friedman ranks) — p-value matrix
    nem = sp.posthoc_nemenyi_friedman(df.values)
    nem.index = nem.columns = METHODS
    print(f'{"maha vs":22} {"Δmean":>7} {"Nemenyi p":>11}')
    rows = []
    for m in METHODS:
        if m == 'maha':
            continue
        dmean = df['maha'].mean() - df[m].mean()
        pn = nem.loc['maha', m]
        sig_n = '***' if pn < 0.001 else '**' if pn < 0.01 else '*' if pn < 0.05 else 'ns'
        print(f'{LABEL[m]:22} {dmean:+7.3f} {pn:11.4g} {sig_n:>3}')
        rows.append(dict(metric=name, comparison=f'maha_vs_{m}', delta_mean=round(dmean, 3),
                         friedman_p=p, nemenyi_p=pn, maha_mean_rank=round(mr['maha'], 2)))
    return rows


def main():
    key = load_key()
    allrows = []
    for field, name in [('recall_main', 'recall_main (k>=2)'), ('k1', 'recall k=1 (subtle)'), ('recall_B_drift', 'recall B_drift')]:
        allrows += analyze(key, field, name)
    out = os.path.join(MAIN, 'posthoc_friedman_nemenyi.csv')
    pd.DataFrame(allrows).to_csv(out, index=False)
    print('\nDesign note: data are a randomized BLOCK design (items = blocks); Nemenyi (Friedman-based) is the')
    print('design-matched post-hoc. Independent-group tests (e.g. Steel-Dwass) are not appropriate here.')
    print('saved', out)


if __name__ == '__main__':
    main()
