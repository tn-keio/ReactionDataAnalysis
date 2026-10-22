# Stage 1z12: Table S5-1 (recall of intended anomalies at exactly matched clean false-positive
# rates, N = 500, 21 items) rebuilt from the v2 (unclipped, 13-method) benchmark, and the S5.4 statement
# "The ordering is identical to Table 1" checked against Table 1's mean recall at the P99.5 operating point.
# Reads recall_at_fpr from the v2 JSONL; dae_elbow / dae_oracle resolved with the corrected selection
# (dae_z_selection_fixed.csv), as in s1z5 / s2d2.
# Output: Results/20261020_Rev_TableS5_1_v2/table_s5_1_v2.csv
# py3.14 (no TensorFlow).
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2d2_m2_k50_yardsticks as Y     # noqa: E402  (jsonl_recs, METHODS13, SEL)

OUT = os.path.join(Y.D.RES, '20261020_Rev_TableS5_1_v2')
N = 500


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = Y.jsonl_recs()
    zsel = pd.read_csv(Y.SEL)
    for _, r in zsel[zsel.N == N].iterrows():
        for tag, col in (('dae_elbow', 'z_elbow'), ('dae_oracle', 'z_oracle')):
            src = recs.get((r['item'], f"dae{int(r[col])}", N))
            if src is not None:
                recs[(r['item'], tag, N)] = src
    items = sorted({k[0] for k in recs})
    rows = []
    for m in Y.METHODS13:
        v = {f: [] for f in ('0.5', '1.0', '5.0')}
        p995 = []
        for it in items:
            r = recs.get((it, m, N))
            if r is None:
                continue
            for f in v:
                v[f].append(r['recall_at_fpr'][f]['mean'])
            p995.append(r['budgets']['99.5']['recall_main']['mean'])
        rows.append(dict(method=m, n_items=len(p995),
                         fp05_mean=np.mean(v['0.5']), fp05_worst=np.min(v['0.5']),
                         fp1_mean=np.mean(v['1.0']), fp5_mean=np.mean(v['5.0']),
                         p995_mean_recall_main=np.mean(p995)))
    t = pd.DataFrame(rows)
    t13 = t.method != 'dae_oracle'           # ranks among the 13 Table 1 methods; the oracle row is S5 only
    for col, src in (('rank_fp05', 'fp05_mean'), ('rank_table1', 'p995_mean_recall_main')):
        t.loc[t13, col] = t.loc[t13, src].rank(ascending=False, method='min')
    t.to_csv(os.path.join(OUT, 'table_s5_1_v2.csv'), index=False)
    print(t.round(3).to_string(index=False))
    print('\nordering identical to Table 1 (13 methods, FP 0.5% mean vs P99.5 mean):',
          bool((t.loc[t13, 'rank_fp05'] == t.loc[t13, 'rank_table1']).all()))


if __name__ == '__main__':
    main()
