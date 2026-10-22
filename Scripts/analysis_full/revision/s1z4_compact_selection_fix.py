# Stage 1z4 (Reviewer #2, M3 bookkeeping): corrected compact-AE z selection for the v2 benchmark.
#
# Defect found 2026-10-13: s2c_summarize_v2.py groups Reports/dae_selection.jsonl by z without filtering
# the method family.  The later runs (run_v2_cae.ps1: cae2/cae8 and the lr3e4 variants) append to the same
# log, so their rows were pooled into the z = 2 and z = 8 means.  The dense-AE rows of the main run carry
# no 'kind'/'lr' field, which is why the mix is silent.  panel_v2.csv / per_item_v2.csv / dae_z_selection.csv
# (2026-09-18 07:33) therefore hold contaminated dae_elbow / dae_oracle rows; 00_Stage2_Results_v2Benchmark.md
# §1 (0.737 / 0.743) was written from the earlier, clean summary.
# This script applies the same elbow / oracle rule to the dense-AE rows only (methods dae1..dae16) and
# writes corrected copies; the original files are NOT modified (s2c is not changed: CodeFix register).
# Output: Results/20261013_Rev_M3_CompactSelFix/{dae_z_selection_fixed.csv, per_item_v2_fixed.csv,
#         panel_v2_fixed.csv, selection_old_vs_new.csv}
# py3.14 (no TensorFlow).

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2c_summarize_v2 as S     # noqa: E402

V2 = os.path.join(S.RES, '20260917_Rev_v2Benchmark')
OUT = os.path.join(S.RES, '20261013_Rev_M3_CompactSelFix')
DENSE = [f'dae{z}' for z in S.DAE_Z]


def picks(sel):
    """s2c_summarize_v2 rule, restricted to the dense AE family."""
    s = pd.DataFrame(sel)
    s = s[s['method'].isin(DENSE)].drop_duplicates(['item', 'method', 'N', 'seed'], keep='last')
    out = []
    for (it, N), g in s.groupby(['item', 'N']):
        mm = g.groupby('z')[['best_val_mse', 'sel_recall_main']].mean()
        lo, hi = mm.best_val_mse.min(), mm.best_val_mse.max()
        out.append(dict(item=it, N=N, z_elbow=int(mm.index[mm.best_val_mse <= lo + 0.1 * (hi - lo)].min()),
                        z_oracle=int(mm.sel_recall_main.idxmax()),
                        **{f'valmse_z{z}': mm.best_val_mse.get(z, np.nan) for z in S.DAE_Z},
                        **{f'selrec_z{z}': mm.sel_recall_main.get(z, np.nan) for z in S.DAE_Z}))
    return pd.DataFrame(out)


def apply_picks(per_item, pk):
    base = per_item[~per_item['method'].isin(['dae_elbow', 'dae_oracle'])]
    extra = []
    for r in pk.itertuples():
        for tag, z in (('dae_elbow', r.z_elbow), ('dae_oracle', r.z_oracle)):
            src = base[(base['item'] == r.item) & (base['method'] == f'dae{z}') & (base['N'] == r.N)]
            if len(src):
                rr = src.iloc[0].to_dict(); rr['method'] = tag; rr['z'] = z
                extra.append(rr)
    return pd.concat([base, pd.DataFrame(extra)], ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = []
    for g in ('main', 'ca', 'm2'):
        sel += S.jl(os.path.join(V2, g, 'Reports', 'dae_selection.jsonl'))
    pk = picks(sel)
    pk.to_csv(os.path.join(OUT, 'dae_z_selection_fixed.csv'), index=False)
    old = pd.read_csv(os.path.join(V2, 'dae_z_selection.csv'))
    cmp = old[['item', 'N', 'z_elbow', 'z_oracle']].merge(pk[['item', 'N', 'z_elbow', 'z_oracle']], on=['item', 'N'],
                                                          suffixes=('_old', '_new'))
    cmp['changed'] = (cmp.z_elbow_old != cmp.z_elbow_new) | (cmp.z_oracle_old != cmp.z_oracle_new)
    cmp.to_csv(os.path.join(OUT, 'selection_old_vs_new.csv'), index=False)
    print(f'picks changed: {int(cmp.changed.sum())} of {len(cmp)} (item x N)')
    for N in sorted(pk.N.unique()):
        p = pk[pk.N == N]
        print(f'N={N}: elbow == oracle in {int((p.z_elbow == p.z_oracle).sum())}/{len(p)} items; '
              f'elbow z=16 in {int((p.z_elbow == 16).sum())}')
    per = pd.read_csv(os.path.join(V2, 'per_item_v2.csv'))
    fixed = apply_picks(per, pk)
    fixed.to_csv(os.path.join(OUT, 'per_item_v2_fixed.csv'), index=False)
    cols = ['recall_main', 'spec', 'fp05', 'B_drift', 'spike_k1', 'k1', 'D1']
    rows = []
    for tag, df in (('old', per), ('fixed', fixed)):
        d = df[df.method.isin(['dae_elbow', 'dae_oracle'])]
        g = d.groupby(['N', 'method'])
        p = g[cols].mean()
        p['worst'] = g.recall_main.min()
        p['worst_item'] = g.apply(lambda x: x.loc[x.recall_main.idxmin(), 'item'])
        p['version'] = tag
        rows.append(p.reset_index())
    pan = pd.concat(rows, ignore_index=True)
    pan.to_csv(os.path.join(OUT, 'panel_v2_fixed.csv'), index=False, float_format='%.4f')
    with pd.option_context('display.width', 200):
        print(pan.sort_values(['N', 'method', 'version']).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
