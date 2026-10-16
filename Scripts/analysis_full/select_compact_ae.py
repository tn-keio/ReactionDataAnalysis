# Bottleneck selection for the dense compact autoencoder ('dae_elbow').
#
# For each item and training size N, z is chosen from {1, 2, 4, 8, 16} with the clean validation MSE
# only (the last 20% of the N training curves, the early-stopping split), averaged over seeds:
#   z_elbow = smallest z whose mean validation MSE <= min + 0.1 * (max - min).
# Neither the holdout, the evaluation battery nor any anomaly label is used.  The metrics of the chosen
# dae<z> are then copied as method 'dae_elbow'.
# Inputs : <results>/Reports/compact_ae_selection.jsonl and comparison.jsonl, written by
#          run_paper_method_comparison.py (one or more result folders, e.g. the main run and the
#          clean re-runs of rerun_ca_clean.py / rerun_m2_clean.py).
# Outputs: <results>/Reports/compact_ae_selection.csv and comparison_dae_elbow.jsonl in each folder.
# Usage  : python select_compact_ae.py <results_dir> [<results_dir> ...]      (no TensorFlow)

import os
import sys
import json
import numpy as np
import pandas as pd

DAE_Z = [1, 2, 4, 8, 16]
DENSE = [f'dae{z}' for z in DAE_Z]


def jl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def picks(sel):
    s = pd.DataFrame(sel)
    s = s[s['method'].isin(DENSE)].drop_duplicates(['item', 'method', 'N', 'seed'], keep='last')
    out = []
    for (it, N), g in s.groupby(['item', 'N']):
        mm = g.groupby('z')['best_val_mse'].mean()
        lo, hi = mm.min(), mm.max()
        out.append(dict(item=it, N=int(N), z_elbow=int(mm.index[mm <= lo + 0.1 * (hi - lo)].min()),
                        **{f'valmse_z{z}': mm.get(z, np.nan) for z in DAE_Z}))
    return pd.DataFrame(out)


def main(dirs):
    for d in dirs:
        rep = os.path.join(d, 'Reports')
        sel = jl(os.path.join(rep, 'compact_ae_selection.jsonl'))
        if not sel:
            print(f'{d}: no compact-AE log, skipped'); continue
        pk = picks(sel)
        pk.to_csv(os.path.join(rep, 'compact_ae_selection.csv'), index=False)
        recs = {(r['item'], r['method'], r['N']): r for r in jl(os.path.join(rep, 'comparison.jsonl'))}
        n = 0
        with open(os.path.join(rep, 'comparison_dae_elbow.jsonl'), 'w', encoding='utf-8') as fh:
            for r in pk.itertuples():
                src = recs.get((r.item, f'dae{r.z_elbow}', r.N))
                if src is None:
                    continue
                fh.write(json.dumps(dict(src, method='dae_elbow', z=r.z_elbow)) + '\n'); n += 1
        print(f'{d}: {len(pk)} item x N selections, {n} dae_elbow records')
        print(pk[['item', 'N', 'z_elbow']].to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: python select_compact_ae.py <results_dir> [...]')
    main(sys.argv[1:])
