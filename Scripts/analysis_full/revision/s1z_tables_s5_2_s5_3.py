# Stage 1z: Tables S5-2 and S5-3 rebuilt from unclipped runs.
#   S5-2  type D recall by lag, N = 500, 21 items: mean over items of per_item_v2.csv (v2 benchmark),
#         with the compact-AE selection corrected by s1z4_compact_selection_fix.py (run that first).
#   S5-3  before / after the module-2 re-base, N = 500:
#         after  = v2 benchmark, 21 items (per_item_v2.csv)
#         before = the same 18 non-re-based items + Ca/ALB/IP from s1y_prerebase_v2.py
#         compact-AE 'dae_elbow' / 'dae_oracle' for the pre-re-base items use the s2c_summarize_v2.py rule.
#   Cross-check: classical rows of s1y must equal s1x_prerebase_classical.py --clip off.
# Output: Results/20261013_Rev_M1_TablesS5_2_3/{table_s5_2.csv, table_s5_3.csv, prerebase_per_item.csv}
# py3.14 (no TensorFlow).

import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2c_summarize_v2 as S     # noqa: E402  (metrics(), jl(), DAE_Z; main() is not run)

RES = S.RES
V2 = os.path.join(RES, '20260917_Rev_v2Benchmark')
PRE = os.path.join(RES, '20261013_Rev_M1_PreRebase_v2', 'prerebase', 'Reports')
S1X = os.path.join(RES, '20261013_Rev_M1_PreRebase_noclip', 'prerebase', 'Reports', 'comparison.jsonl')
OUT = os.path.join(RES, '20261013_Rev_M1_TablesS5_2_3')
REBASED = ['Ca', 'ALB', 'IP']
ROWS = ['maha', 'maha_robust', 'pca', 'dae_oracle', 'dae_elbow', 'cae8', 'lof', 'ae2d_ens', 'ae', 'twod_only',
        'ae2d_and', 'ocsvm', 'iforest', 'cnn1d']
N0 = 500


def add_compact_picks(df, sel):
    """Same elbow / oracle rule as s2c_summarize_v2.main(), restricted to the dense AEs (dae1..dae16);
    s2c itself pools the cae / lr3e4 rows into z = 2 and 8 (see s1z4_compact_selection_fix.py)."""
    s = pd.DataFrame(sel)
    s = s[s['method'].isin([f'dae{z}' for z in S.DAE_Z])].drop_duplicates(['item', 'method', 'N', 'seed'], keep='last')
    extra = []
    for (it, N), g in s.groupby(['item', 'N']):
        mm = g.groupby('z')[['best_val_mse', 'sel_recall_main']].mean()
        lo, hi = mm.best_val_mse.min(), mm.best_val_mse.max()
        picks = {'dae_elbow': int(mm.index[mm.best_val_mse <= lo + 0.1 * (hi - lo)].min()),
                 'dae_oracle': int(mm.sel_recall_main.idxmax())}
        for tag, z in picks.items():
            src = df[(df['item'] == it) & (df['method'] == f'dae{z}') & (df['N'] == N)]
            if len(src):
                rr = src.iloc[0].to_dict(); rr['method'] = tag; rr['z'] = z
                extra.append(rr)
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def panel(df):
    d = df[df.N == N0]
    g = d.groupby('method')
    p = pd.DataFrame({'recall_mean': g.recall_main.mean(), 'worst_item': g.recall_main.min(),
                      'worst_item_name': g.apply(lambda x: x.loc[x.recall_main.idxmin(), 'item']),
                      'specificity': g.spec.mean(), 'n_items': g.size()})
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    # corrected compact-AE selection (dae_elbow / dae_oracle rows); all other rows equal per_item_v2.csv
    v2 = pd.read_csv(os.path.join(RES, '20261013_Rev_M3_CompactSelFix', 'per_item_v2_fixed.csv'))
    # ---- S5-2
    d = v2[(v2.N == N0) & v2.method.isin(ROWS)]
    t2 = d.groupby('method')[[f'D{L}' for L in range(1, 6)]].mean()
    t2['n_items'] = d.groupby('method').size()
    t2 = t2.loc[[m for m in ROWS if m in t2.index]]
    t2.to_csv(os.path.join(OUT, 'table_s5_2.csv'), float_format='%.4f')
    print('== Table S5-2 (N=500, mean over items) ==\n', t2.round(3).to_string())
    # ---- S5-3
    pre = pd.DataFrame([dict(item=r['item'], method=r['method'], N=r['N'], **S.metrics(r))
                        for r in S.jl(os.path.join(PRE, 'comparison.jsonl'))])
    pre = pre.drop_duplicates(['item', 'method', 'N'], keep='last')
    pre = add_compact_picks(pre, S.jl(os.path.join(PRE, 'dae_selection.jsonl')))
    pre.to_csv(os.path.join(OUT, 'prerebase_per_item.csv'), index=False, float_format='%.6f')
    miss = [(it, m) for it in REBASED for m in ROWS if not ((pre.item == it) & (pre.method == m) & (pre.N == N0)).any()]
    assert not miss, f'pre-re-base rows missing: {miss}'
    before = pd.concat([v2[~v2.item.isin(REBASED)], pre], ignore_index=True)
    a, b = panel(v2), panel(before)
    t3 = pd.DataFrame({'recall_before': b.recall_mean, 'worst_before': b.worst_item, 'worst_item_before': b.worst_item_name,
                       'spec_before': b.specificity, 'd_recall': a.recall_mean - b.recall_mean,
                       'd_worst': a.worst_item - b.worst_item, 'd_spec': a.specificity - b.specificity,
                       'recall_after': a.recall_mean, 'worst_after': a.worst_item, 'worst_item_after': a.worst_item_name,
                       'spec_after': a.specificity, 'n_before': b.n_items, 'n_after': a.n_items})
    t3 = t3.loc[ROWS]
    t3.to_csv(os.path.join(OUT, 'table_s5_3.csv'), float_format='%.4f')
    with pd.option_context('display.width', 250):
        print('== Table S5-3 (N=500) ==\n', t3.round(3).to_string())
    # rank stability
    for col in ('recall', 'worst'):
        rb = t3[f'{col}_before'].rank(ascending=False, method='min'); ra = t3[f'{col}_after'].rank(ascending=False, method='min')
        print(f'{col}: rank changed for', list(t3.index[rb != ra]))
    # ---- cross-check against s1x (classical, --clip off)
    if os.path.exists(S1X):
        x = {(r['item'], r['method'], r['N']): S.metrics(r) for r in S.jl(S1X)}
        dmax = 0.0
        for _, r in pre[pre.method.isin(['maha', 'maha_robust', 'pca', 'lof', 'iforest', 'ocsvm'])].iterrows():
            o = x.get((r['item'], r['method'], r['N']))
            if o is not None:
                dmax = max(dmax, abs(o['recall_main'] - r['recall_main']), abs(o['spec'] - r['spec']))
        print('cross-check vs s1x noclip: max |diff| (recall_main, spec) =', dmax)


if __name__ == '__main__':
    main()
