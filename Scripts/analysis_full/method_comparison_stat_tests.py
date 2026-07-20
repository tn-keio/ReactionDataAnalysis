# Statistical significance of Mahalanobis(E) superiority in the 5-38 / 14-item method comparison.
#  (1) item-blocked Wilcoxon signed-rank (paired by item, one-sided maha>other) on recall_main / k=1 / B_drift
#  (2) per-sample McNemar on the synthetic anomaly battery (maha vs each method), pooled over items
# Uses the CONFIRMED merged data (clean Ca/ALB/IP re-runs).  py3.14 + scipy.  No TF.
import os, json, csv
import numpy as np
from scipy.stats import wilcoxon, binomtest

BASE = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
MAIN = os.path.join(BASE, 'Results', '20260717_MethodComparison_p5_38')
CACLEAN = os.path.join(BASE, 'Results', '20260718_MethodComparison_Ca_clean')
M2CLEAN = os.path.join(BASE, 'Results', '20260718_MethodComparison_M2_clean')
PANEL = ['TP', 'ALB', 'TB', 'DB', 'Ca', 'IP', 'CRE', 'UA', 'Fe', 'Cu', 'Zn', 'GLU', 'GA', 'CRP']  # method-comparison 14-item panel
METHODS = ['maha', 'maha_robust', 'pca', 'ae', 'cnn1d', 'ae2d_and', 'ae2d_ens', 'twod_only', 'lof', 'iforest', 'ocsvm']
OTHERS = [m for m in METHODS if m != 'maha']
LABEL = {'maha_robust': 'Mahalanobis(Robust)', 'pca': 'PCA', 'ae': '1D-AE', 'cnn1d': '1D-CNN→iForest',
         'ae2d_and': 'AE∧2D-CNN', 'ae2d_ens': 'AE⊕2D-CNN', 'twod_only': '2D-CNN→iForest', 'lof': 'LOF',
         'iforest': 'Isolation Forest', 'ocsvm': 'One-Class SVM'}
POS = {'A_spike', 'B_drift', 'C_hump', 'E_baseline'}
CLEAN_SRC = {'Ca': CACLEAN, 'ALB': M2CLEAN, 'IP': M2CLEAN}


# ---------- (1) Wilcoxon on per-item metrics ----------
def load_metrics():
    recs = [r for r in (json.loads(l) for l in open(os.path.join(MAIN, 'Reports', 'comparison.jsonl'), encoding='utf-8'))
            if r['item'] != 'Ca' and r['item'] not in ('ALB', 'IP')]
    for f in [os.path.join(CACLEAN, 'Reports', 'comparison.jsonl'), os.path.join(M2CLEAN, 'Reports', 'comparison.jsonl')]:
        recs += [r for r in (json.loads(l) for l in open(f, encoding='utf-8')) if r['item'] in ('Ca', 'ALB', 'IP')]
    key = {(r['item'], r['method'], r['N']): r for r in recs}
    return key


def metric_vec(key, method, field):
    out = []
    for it in PANEL:
        b = key[(it, method, 500)]['budgets']['99.5']
        if field == 'k1':
            v = np.nanmean([b.get(f'recall_{t}_k1', {'mean': np.nan})['mean'] for t in POS])
        else:
            v = b[field]['mean']
        out.append(v)
    return np.array(out)


def wilcoxon_block(key):
    print('=== (1) Item-blocked Wilcoxon signed-rank (paired by 14 items, one-sided maha > other) ===')
    rows = []
    for field, name in [('recall_main', 'recall_main(k>=2)'), ('k1', 'recall k=1(subtle)'), ('recall_B_drift', 'recall_B_drift')]:
        mv = metric_vec(key, 'maha', field)
        print(f'\n-- {name}: maha median={np.median(mv):.3f} --')
        for o in OTHERS:
            ov = metric_vec(key, o, field)
            diff = mv - ov
            if np.allclose(diff, 0):
                p, stat = 1.0, np.nan
            else:
                try:
                    res = wilcoxon(mv, ov, alternative='greater', zero_method='wilcox')
                    stat, p = res.statistic, res.pvalue
                except ValueError:
                    stat, p = np.nan, 1.0
            n_better = int((diff > 0).sum()); n_worse = int((diff < 0).sum())
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f'  maha vs {LABEL[o]:22} median Δ={np.median(diff):+.3f}  ({n_better}>/{n_worse}<)  W={stat}  p={p:.4g} {sig}')
            rows.append(dict(test='wilcoxon', metric=name, comparison=f'maha>{o}', median_maha=round(np.median(mv), 3),
                             median_other=round(np.median(ov), 3), median_delta=round(np.median(diff), 3),
                             n_better=n_better, n_worse=n_worse, W=stat, p=p))
    return rows


# ---------- (2) McNemar on per-sample synthetic detection ----------
def load_list(item):
    src = CLEAN_SRC.get(item, MAIN)
    p = os.path.join(src, item, 'N500', 'list.csv')
    rows = []
    for r in csv.DictReader(open(p, encoding='utf-8')):
        if r['source'] == 'synthetic' and r['ground_truth'] in POS:
            try:
                if float(r['severity_k']) >= 2:
                    rows.append({m: int(r[m]) for m in METHODS})
            except ValueError:
                pass
    return rows


def mcnemar(key_unused):
    print('\n\n=== (2) McNemar on synthetic anomaly detection (per-sample, pooled 14 items, seed0 P99.5) ===')
    print('    task: detect intended anomalies (POS types, k>=2); correct = flagged(1)')
    allrows = []
    for it in PANEL:
        allrows += load_list(it)
    print(f'    n synthetic main-anomaly samples pooled = {len(allrows)}')
    maha = np.array([r['maha'] for r in allrows])
    rows = []
    for o in OTHERS:
        oth = np.array([r[o] for r in allrows])
        b = int(((maha == 1) & (oth == 0)).sum())   # maha correct, other wrong
        c = int(((maha == 0) & (oth == 1)).sum())   # maha wrong, other correct
        # exact McNemar: test discordant pairs against 0.5 (one-sided: maha better if c<b)
        p = binomtest(c, b + c, 0.5, alternative='less').pvalue if (b + c) > 0 else 1.0
        p2 = binomtest(min(b, c), b + c, 0.5, alternative='two-sided').pvalue if (b + c) > 0 else 1.0
        sig = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else 'ns'
        direc = 'maha better' if b > c else 'other better' if c > b else 'tie'
        print(f'  maha vs {LABEL[o]:22} b(maha✓/oth✗)={b:5} c(maha✗/oth✓)={c:5}  {direc:12} p={p2:.3g} {sig}')
        rows.append(dict(test='mcnemar', metric='detect_main', comparison=f'maha vs {o}',
                         b_maha_correct=b, c_other_correct=c, direction=direc, p_two_sided=p2, p_maha_better=p))
    return rows


def main():
    key = load_metrics()
    r1 = wilcoxon_block(key)
    r2 = mcnemar(key)
    out = os.path.join(MAIN, 'statistical_tests.csv')
    allcols = sorted(set().union(*[r.keys() for r in r1 + r2]))
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=allcols); w.writeheader(); w.writerows(r1 + r2)
    print('\nNote: 10 methods x 3 metrics; apply Holm-Bonferroni for family-wise error if needed.')
    print('saved', out)


if __name__ == '__main__':
    main()
