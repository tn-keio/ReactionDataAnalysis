# Stage 2c: summarize the v2 benchmark (pilot or full) against the submitted results.
#   * per item x method x N: recall_main, specificity, FP-matched recall, B_drift, 1-sigma, sub-1-sigma,
#     time-warp and saturation recall, submitted values alongside
#   * compact dense AE: z chosen by (a) unsupervised elbow on seed-mean clean validation MSE
#     (smallest z with mean MSE <= min + 0.1*(max - min)) and (b) oracle = max seed-mean recall on the
#     independent selection battery; reports the evaluation metrics of the chosen z ('dae_elbow', 'dae_oracle')
# Usage: py s2c_summarize_v2.py [--pilot]      (py3.14, no TensorFlow)

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'ForPaper'))
from _common import panel as P      # noqa: E402

RES = os.path.join(BASE, 'Results')
POS = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']
KLOW = ['0.02', '0.05', '0.1', '0.2', '0.5', '1']
DAE_Z = [1, 2, 4, 8, 16]


def jl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def metrics(rec):
    b = rec['budgets']['99.5']
    g = lambda k: b[k]['mean'] if k in b else np.nan
    out = dict(recall_main=g('recall_main'), spec=g('specificity_clean'), fp05=rec['recall_at_fpr']['0.5']['mean'],
               B_drift=g('recall_B_drift'), spike_k1=g('recall_A_spike_k1'))
    for k in KLOW:
        out[f'k{k}'] = np.nanmean([g(f'recall_{t}_k{k}') for t in POS])
    for r in ('0.95', '0.9', '0.8', '0.7', '0.5'):
        out[f'W{r}'] = g(f'recall_W_r{r}')
    for q in ('0.98', '0.95', '0.9', '0.8'):
        out[f'S{q}'] = g(f'recall_S_q{q}')
    for L in range(1, 6):
        out[f'D{L}'] = g(f'recall_D_lag{L}')
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pilot', action='store_true'); a = ap.parse_args()
    root = os.path.join(RES, '20260917_Rev_v2Benchmark' + ('_pilot' if a.pilot else ''))
    recs, sel = {}, []
    for g in ('main', 'ca', 'm2'):
        for r in jl(os.path.join(root, g, 'Reports', 'comparison.jsonl')):
            recs[(r['item'], r['method'], r['N'])] = r
        sel += jl(os.path.join(root, g, 'Reports', 'dae_selection.jsonl'))
    orig = P.load_metrics(21, require_all=False)
    rows = []
    for (it, m, N), r in recs.items():
        row = dict(item=it, method=m, N=N, **metrics(r))
        o = orig.get((it, m, N))
        if o is not None:
            om = metrics(o)
            for k in ('recall_main', 'spec', 'fp05', 'B_drift', 'spike_k1', 'k1'):
                row[f'{k}_sub'] = om[k]
        rows.append(row)
    df = pd.DataFrame(rows)
    # ---- compact AE selection ----
    if sel:
        s = pd.DataFrame(sel).drop_duplicates(['item', 'method', 'N', 'seed'], keep='last')
        picks = []
        for (it, N), g in s.groupby(['item', 'N']):
            mm = g.groupby('z')[['best_val_mse', 'sel_recall_main']].mean()
            lo, hi = mm.best_val_mse.min(), mm.best_val_mse.max()
            z_elbow = int(mm.index[mm.best_val_mse <= lo + 0.1 * (hi - lo)].min())
            z_oracle = int(mm.sel_recall_main.idxmax())
            picks.append(dict(item=it, N=N, z_elbow=z_elbow, z_oracle=z_oracle,
                              **{f'valmse_z{z}': mm.best_val_mse.get(z, np.nan) for z in DAE_Z},
                              **{f'selrec_z{z}': mm.sel_recall_main.get(z, np.nan) for z in DAE_Z}))
            for tag, z in (('dae_elbow', z_elbow), ('dae_oracle', z_oracle)):
                src = df[(df['item'] == it) & (df['method'] == f'dae{z}') & (df['N'] == N)]
                if len(src):
                    rr = src.iloc[0].to_dict(); rr['method'] = tag; rr['z'] = z
                    df = pd.concat([df, pd.DataFrame([rr])], ignore_index=True)
        pk = pd.DataFrame(picks)
        pk.to_csv(os.path.join(root, 'dae_z_selection.csv'), index=False)
        print('== compact-AE z selection ==\n', pk[['item', 'N', 'z_elbow', 'z_oracle']].to_string(index=False))
    df.to_csv(os.path.join(root, 'per_item_v2.csv'), index=False)
    # ---- panel summary ----
    agg = {'recall_main': ['mean', 'min'], 'spec': 'mean', 'fp05': 'mean', 'B_drift': 'mean', 'spike_k1': 'mean',
           'k1': 'mean', 'k0.5': 'mean', 'k0.2': 'mean', 'k0.1': 'mean', 'k0.05': 'mean', 'W0.8': 'mean',
           'W0.5': 'mean', 'S0.95': 'mean', 'S0.9': 'mean', 'D1': 'mean'}
    for c in ('recall_main_sub', 'spec_sub', 'k1_sub'):
        if c in df:
            agg[c] = ['mean', 'min'] if c == 'recall_main_sub' else 'mean'
    p = df.groupby(['N', 'method']).agg(agg)
    p.columns = ['_'.join(c) if isinstance(c, tuple) else c for c in p.columns]
    p['n_items'] = df.groupby(['N', 'method']).size()
    p = p.reset_index().sort_values(['N', 'recall_main_mean'], ascending=[True, False])
    p.to_csv(os.path.join(root, 'panel_v2.csv'), index=False)
    with pd.option_context('display.width', 300, 'display.max_columns', 40):
        print(p[p.N == 500].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
