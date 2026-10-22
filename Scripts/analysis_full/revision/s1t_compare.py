# Stage 1t (compare): file order (unclipped re-runs) vs time order (s1t_run_timeorder), side by side.
#   --what classical : Table 1 panel metrics per method and N, per-item differences, rank agreement
#   --what dl        : v2 benchmark, N = 500, CRE and TP
#   --what others    : Cu case, Table S3 workload, S4 pooling/transfer, S14 lead times, s1m arm A, rolling evaluation
# Output: Results/20260925_Rev_TimeOrder/compare_<what>.csv and printed summaries.
import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(BASE, 'Results')
T = os.path.join(RES, '20260925_Rev_TimeOrder')
KEYS = {'recall_main': 'recall_main', 'specificity_clean': 'specificity_clean', 'B_drift': 'recall_B_drift',
        'spike_k1': 'recall_A_spike_k1'}


def read_jsonl(root):
    rows = []
    for f in glob.glob(os.path.join(root, '*', 'Reports', 'comparison.jsonl')):
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            b = r['budgets']['99.5']
            rec = dict(item=r['item'], method=r['method'], N=r['N'])
            for k, v in KEYS.items():
                if v in b:
                    rec[k] = b[v]['mean']
            k1 = [b[f'recall_{t}_k1']['mean'] for t in ('A_spike', 'B_drift', 'C_hump', 'E_baseline') if f'recall_{t}_k1' in b]
            rec['k1_mean4'] = float(np.mean(k1)) if k1 else np.nan
            rows.append(rec)
    return pd.DataFrame(rows).drop_duplicates(['item', 'method', 'N'], keep='last')


def classical():
    a = read_jsonl(os.path.join(RES, '20260917_Rev_M1_Classical_noclip'))
    b = read_jsonl(os.path.join(T, 'classical'))
    m = a.merge(b, on=['item', 'method', 'N'], suffixes=('_file', '_time'))
    m = m[m.method.isin(['maha', 'maha_robust', 'pca', 'lof'])]
    m.to_csv(os.path.join(T, 'compare_classical.csv'), index=False)
    print(f'matched item x method x N: {len(m)}')
    cols = ['recall_main', 'specificity_clean', 'B_drift', 'spike_k1', 'k1_mean4']
    for N, g in m.groupby('N'):
        print(f'\n=== N = {N}: panel mean (worst item for recall_main)  file -> time')
        for meth, h in g.groupby('method'):
            s = ', '.join(f"{c} {h[c + '_file'].mean():.3f}->{h[c + '_time'].mean():.3f}" for c in cols)
            print(f"  {meth:12s} worst {h.recall_main_file.min():.3f}->{h.recall_main_time.min():.3f} | {s}")
        d = (g.recall_main_time - g.recall_main_file).abs()
        print(f'  max |per-item change| recall_main {d.max():.3f} ({g.loc[d.idxmax(), "item"]} {g.loc[d.idxmax(), "method"]}); '
              f'specificity {(g.specificity_clean_time - g.specificity_clean_file).abs().max():.3f}')
        # rank agreement per item: order of the four methods by recall_main
        agree = []
        for it, h in g.groupby('item'):
            if len(h) == 4:
                agree.append(list(h.sort_values('recall_main_file', ascending=False).method) ==
                             list(h.sort_values('recall_main_time', ascending=False).method))
        rho = spearmanr(g.recall_main_file, g.recall_main_time).statistic
        print(f'  per-item method order identical in {sum(agree)}/{len(agree)} items (ties can reorder); Spearman {rho:.4f}')
        top = g[g.method == 'maha'].set_index('item')
        bf = g.groupby('item').recall_main_file.max().reindex(top.index)
        bt = g.groupby('item').recall_main_time.max().reindex(top.index)
        print(f"  empirical Mahalanobis best-or-tied on recall_main: file {int((top.recall_main_file >= bf - 1e-9).sum())}/{len(top)}, "
              f"time {int((top.recall_main_time >= bt - 1e-9).sum())}/{len(top)}")
        ch = top[(top.recall_main_time - top.recall_main_file).abs() > 1e-9]
        print('  maha items that changed (recall_main file->time; spec): ' +
              ', '.join(f"{i} {r.recall_main_file:.3f}->{r.recall_main_time:.3f} ({r.specificity_clean_file:.3f}->{r.specificity_clean_time:.3f})" for i, r in ch.iterrows()))


def dl():
    base = os.path.join(RES, '20260917_Rev_v2Benchmark')
    a = read_jsonl(base)
    b = read_jsonl(os.path.join(RES, '20260917_Rev_v2Benchmark_timeorder'))
    m = a.merge(b, on=['item', 'method', 'N'], suffixes=('_file', '_time'))
    m = m[(m.N == 500) & m.item.isin(['CRE', 'TP'])]
    m.to_csv(os.path.join(T, 'compare_dl.csv'), index=False)
    print(m[['item', 'method', 'recall_main_file', 'recall_main_time', 'k1_mean4_file', 'k1_mean4_time',
             'specificity_clean_file', 'specificity_clean_time']].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', choices=['classical', 'dl'], required=True)
    a = ap.parse_args()
    {'classical': classical, 'dl': dl}[a.what]()


if __name__ == '__main__':
    main()
