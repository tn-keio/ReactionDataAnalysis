# Statistical significance of Mahalanobis(E) superiority in the 5-38 / 21-item method comparison (13 methods).
#  (1) item-blocked Wilcoxon signed-rank (paired by item, one-sided maha>other, Holm-corrected over the 12
#      comparisons within each metric) on recall_main / k=1 / B_drift
#  (2) per-sample McNemar discordance on the synthetic anomaly battery (maha vs each method), pooled over
#      items, with a base-curve cluster bootstrap of b/c and b - c (the 16 samples derived from one base
#      curve are correlated, so the base curve is the resampling unit)
# Uses the merged data (clean Ca/ALB/IP re-runs) and the dae_elbow selection of select_compact_ae.py.
# py3.14 + scipy + statsmodels.  No TF.
import os, csv
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest
from statsmodels.stats.multitest import multipletests
from method_comparison_posthoc import MAIN, PANEL, CLEAN_SRC, METHODS, LABEL, POS, N, load_key, matrix

OTHERS = [m for m in METHODS if m != 'maha']


# ---------- (1) Wilcoxon on per-item metrics ----------
def wilcoxon_block(key):
    print('=== (1) Item-blocked Wilcoxon signed-rank (paired by 21 items, one-sided maha > other, Holm) ===')
    rows = []
    for field, name in [('recall_main', 'recall_main(k>=2)'), ('k1', 'recall k=1(subtle)'), ('recall_B_drift', 'recall_B_drift')]:
        df = matrix(key, field)
        mv = df['maha'].values
        print(f'\n-- {name}: maha median={np.median(mv):.3f} --')
        block = []
        for o in OTHERS:
            ov = df[o].values
            diff = mv - ov
            if np.allclose(diff, 0):
                p, stat = 1.0, np.nan
            else:
                res = wilcoxon(mv, ov, alternative='greater', zero_method='wilcox')
                stat, p = res.statistic, res.pvalue
            block.append(dict(test='wilcoxon', metric=name, comparison=f'maha>{o}', median_maha=round(np.median(mv), 3),
                              median_other=round(np.median(ov), 3), median_delta=round(np.median(diff), 3),
                              n_better=int((diff > 0).sum()), n_worse=int((diff < 0).sum()), W=stat, p=p))
        for r, ph in zip(block, multipletests([r['p'] for r in block], method='holm')[1]):
            r['p_holm'] = ph
            sig = '***' if ph < 0.001 else '**' if ph < 0.01 else '*' if ph < 0.05 else 'ns'
            print(f"  maha vs {LABEL[r['comparison'][5:]]:22} median Δ={r['median_delta']:+.3f}  "
                  f"({r['n_better']}>/{r['n_worse']}<)  W={r['W']}  p={r['p']:.4g}  p_holm={ph:.4g} {sig}")
        rows += block
    return rows


# ---------- (2) McNemar on per-sample synthetic detection ----------
def load_list(item):
    """Scored synthetic anomalies (POS types, k >= 2) of one item: base curve id + 0/1 flag per method
    (seed 0, holdout P99.5).  dae_elbow = the dae<z> column with z from compact_ae_selection.csv."""
    src = CLEAN_SRC.get(item, MAIN)
    sel = pd.read_csv(os.path.join(src, 'Reports', 'compact_ae_selection.csv'))
    z = int(sel.loc[(sel['item'] == item) & (sel['N'] == N), 'z_elbow'].iloc[0])
    cols = {m: (f'dae{z}' if m == 'dae_elbow' else m) for m in METHODS}
    rows = []
    for r in csv.DictReader(open(os.path.join(src, item, f'N{N}', 'list.csv'), encoding='utf-8')):
        if r['source'] == 'synthetic' and r['ground_truth'] in POS and float(r['severity_k']) >= 2:
            rows.append({'item': item, 'base_id': int(float(r['id'])), **{m: int(r[c]) for m, c in cols.items()}})
    return rows


def mcnemar(B=2000, seed=20260726):
    print('\n\n=== (2) McNemar on synthetic anomaly detection (per-sample, pooled 21 items, seed0 P99.5) ===')
    print('    task: detect intended anomalies (POS types, k>=2); correct = flagged(1)')
    sub = pd.DataFrame([r for it in PANEL for r in load_list(it)])
    print(f'    n synthetic main-anomaly samples pooled = {len(sub)}, base curves = '
          f'{sub.groupby(["item", "base_id"]).ngroups}')
    # clusters = base curves, resampled with replacement within each item
    offs = [[g.index.values[ix] for ix in g.groupby('base_id').indices.values()] for _, g in sub.groupby('item')]
    A = sub['maha'].values.astype(bool)
    F = {o: sub[o].values.astype(bool) for o in OTHERS}
    ratios = {o: [] for o in OTHERS}; diffs = {o: [] for o in OTHERS}
    rng = np.random.default_rng(seed)
    for _ in range(B):
        take = np.concatenate([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]) for g in offs])
        a = A[take]
        for o in OTHERS:
            f = F[o][take]
            b = int((a & ~f).sum()); c = int((~a & f).sum())
            ratios[o].append(b / c if c else np.inf); diffs[o].append(b - c)
    rows = []
    for o in OTHERS:
        b = int((A & ~F[o]).sum())   # maha correct, other wrong
        c = int((~A & F[o]).sum())   # maha wrong, other correct
        rr = np.array(ratios[o]); fin = rr[np.isfinite(rr)]; dd = np.array(diffs[o], float)
        # exact McNemar (independence assumed; reported for reference only -- the cluster bootstrap is the inference)
        p = binomtest(b, b + c, 0.5, alternative='greater').pvalue if (b + c) > 0 else 1.0
        r = dict(test='mcnemar', metric='detect_main', comparison=f'maha vs {o}', b_maha_only=b, c_other_only=c,
                 ratio=b / c if c else np.inf,
                 ratio_lo=np.percentile(fin, 2.5) if fin.size else np.inf,
                 ratio_hi=np.percentile(fin, 97.5) if fin.size else np.inf,
                 diff_lo=np.percentile(dd, 2.5), diff_hi=np.percentile(dd, 97.5),
                 frac_favouring_maha=(dd > 0).mean(), p_exact_independent=p)
        print(f"  maha vs {LABEL[o]:22} b={b:6} c={c:5}  b/c={r['ratio']:8.1f} ({r['ratio_lo']:.1f}-{r['ratio_hi']:.1f})"
              f"  b-c 95% {r['diff_lo']:.0f}-{r['diff_hi']:.0f}  favouring maha {r['frac_favouring_maha']:.3f}")
        rows.append(r)
    return rows


def main():
    key = load_key()
    r1 = wilcoxon_block(key)
    r2 = mcnemar()
    out = os.path.join(MAIN, 'statistical_tests.csv')
    allcols = sorted(set().union(*[r.keys() for r in r1 + r2]))
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=allcols); w.writeheader(); w.writerows(r1 + r2)
    print('\nNote: base-curve cluster bootstrap, B = 2000, clusters resampled within each item.')
    print('saved', out)


if __name__ == '__main__':
    main()
