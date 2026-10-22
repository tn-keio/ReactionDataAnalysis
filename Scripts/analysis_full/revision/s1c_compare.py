# Stage 1c: compare re-run JSONLs against the submitted 21-item results.
#   repro  : Results/20260917_Rev_M1_Classical_clip   vs submitted -> max |diff| per (item, method, N)
#   delta  : Results/20260917_Rev_M1_Classical_noclip vs submitted -> Table-1-style panel summary
# py3.14 is fine (no TensorFlow).  Writes into the noclip/clip result folders.

import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'ForPaper'))
from _common import panel as P      # noqa: E402

RES = os.path.join(BASE, 'Results')
POS = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']


def load_run(tag):
    store = {}
    for g in ('main', 'ca', 'm2'):
        p = os.path.join(RES, tag, g, 'Reports', 'comparison.jsonl')
        if os.path.exists(p):
            for line in open(p, encoding='utf-8'):
                if line.strip():
                    r = json.loads(line); store[(r['item'], r['method'], r['N'])] = r
    return store


def flat(rec, prefix=''):
    out = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            out.update(flat(v, f'{prefix}{k}.'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and 'time' not in k and 'infer' not in k:
            out[prefix + k] = float(v)
    return out


def key_metrics(rec):
    b = rec['budgets']['99.5']
    g = lambda k: b[k]['mean'] if k in b else np.nan
    return dict(recall_main=g('recall_main'), spec=g('specificity_clean'),
                B_drift=g('recall_B_drift'), spike_k1=g('recall_A_spike_k1'),
                k1_mean=np.nanmean([g(f'recall_{t}_k1') for t in POS]),
                fp05=rec['recall_at_fpr']['0.5']['mean'])


def main():
    orig = P.load_metrics(21)
    items = P.proc_names(21)
    # ---- reproduction check ----
    rep = load_run('20260917_Rev_M1_Classical_clip')
    if rep:
        rows = []
        for k, r in rep.items():
            o = orig.get(k)
            if o is None:
                rows.append(dict(item=k[0], method=k[1], N=k[2], max_abs_diff=np.nan, note='no original')); continue
            fr, fo = flat(r), flat(o)
            common = set(fr) & set(fo)
            d = max(abs(fr[c] - fo[c]) for c in common) if common else np.nan
            rows.append(dict(item=k[0], method=k[1], N=k[2], max_abs_diff=d, n_fields=len(common)))
        df = pd.DataFrame(rows).sort_values(['item', 'method', 'N'])
        df.to_csv(os.path.join(RES, '20260917_Rev_M1_Classical_clip', 'repro_check.csv'), index=False)
        print('REPRO: max |diff| over all fields =', df['max_abs_diff'].max(), ' n=', len(df))
    # ---- unclipped vs submitted ----
    new = load_run('20260917_Rev_M1_Classical_noclip')
    if not new:
        return
    rows = []
    for (it, m, N), r in new.items():
        if it not in items or (it, m, N) not in orig:
            continue
        kn, ko = key_metrics(r), key_metrics(orig[(it, m, N)])
        rows.append(dict(item=it, method=m, N=N, **{f'{k}_new': v for k, v in kn.items()},
                         **{f'{k}_sub': v for k, v in ko.items()}))
    df = pd.DataFrame(rows)
    out = os.path.join(RES, '20260917_Rev_M1_Classical_noclip')
    df.to_csv(os.path.join(out, 'per_item_new_vs_submitted.csv'), index=False)
    summ = []
    for (m, N), d in df.groupby(['method', 'N']):
        n = len(d)
        row = dict(method=m, N=N, n_items=n)
        for s in ('sub', 'new'):
            row[f'mean_{s}'] = d[f'recall_main_{s}'].mean()
            row[f'worst_{s}'] = d[f'recall_main_{s}'].min()
            row[f'worst_item_{s}'] = d.loc[d[f'recall_main_{s}'].idxmin(), 'item']
            row[f'spec_{s}'] = d[f'spec_{s}'].mean()
            row[f'Bdrift_{s}'] = d[f'B_drift_{s}'].mean()
            row[f'spikek1_{s}'] = d[f'spike_k1_{s}'].mean()
            row[f'k1mean_{s}'] = d[f'k1_mean_{s}'].mean()
            row[f'fp05_{s}'] = d[f'fp05_{s}'].mean()
        summ.append(row)
    s = pd.DataFrame(summ).sort_values(['N', 'mean_new'], ascending=[True, False])
    s.to_csv(os.path.join(out, 'panel_summary_new_vs_submitted.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(s.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
