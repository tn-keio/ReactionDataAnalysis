# Stage 1v: the 2021-08 optical event of S10b.2 / S14.2
# (module-2 flag rates 65-86%, module-1 <1.5%, Spearman rho(main wavelength, flag rate) ~ 0.11)
# re-computed with and without np.clip.
#
# The scratch script that produced ForPaper/Suppl09b_OperatingPoint_Ca/event_wavelength.csv is not in
# the working tree.  This is a reconstruction of the protocol recorded in REPORT_Ca_operating_point.md
# §7.7: train on the pre-event window 2021-05-01..2021-07-15, self-calibrated P99.5 threshold, flag rate
# on the event window 2021-08-01..2021-08-22, 21 items, empirical Mahalanobis on points 5-38 with
# per-point min-max (event_robustness_analysis.maha_eval formulation).  --clip on must reproduce
# event_wavelength.csv (n_ev and flag_pct) before the --clip off values are used.
# Variants are exposed (--norm, --train) only to locate the submitted formulation.
# Output: Results/20261012_Rev_M1_OpticalEvent/event_wavelength_<tag>.csv
# py3.14 (no TensorFlow).

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
PARAM = os.path.join(BASE, 'Data', 'parameter_mapped.csv')
SUBMITTED = os.path.join(BASE, 'ForPaper', 'Suppl09b_OperatingPoint_Ca', 'event_wavelength.csv')
OUT = os.path.join(BASE, 'Results', '20261012_Rev_M1_OpticalEvent')

PTS = list(range(5, 39))
RAM = [f'RAM{i}' for i in PTS]
RAS = [f'RAS{i}' for i in PTS]
TR0, TR1 = '2021-05-01', '2021-07-15'
EV0, EV1 = '2021-08-01', '2021-08-22'
BUDGET = 99.5
# event_wavelength.csv item label -> processed A_Name partition
ITEMS = {'IP': 'IP', 'GLU': 'GLU', 'IGM': 'IgM', 'C4': 'C4', 'TB': 'TB', 'DB': 'DB', 'TP': 'TP',
         'CRE': 'CRE', 'GA': 'GA', 'LDL-C': 'LDL-C', 'CRP': 'CRP', 'ALB': 'ALB', 'UA': 'UA', 'TC': 'TC',
         'HDL-C': 'HDL-C', 'TG': 'TG', 'C3': 'C3', 'IGG': 'IgG', 'IGA': 'IgA', 'CA': 'Ca', 'FE': 'Fe'}


def load(a_name):
    d = ds.dataset(PROC, partitioning='hive')
    t = d.to_table(columns=['S_Date'] + RAM + RAS, filter=pc.field('A_Name') == a_name).to_pandas()
    t['S_Date'] = pd.to_datetime(t['S_Date'])
    t = t.sort_values('S_Date', kind='mergesort').reset_index(drop=True)
    calc = t[RAM].values.astype('float64') - t[RAS].values.astype('float64')
    return t['S_Date'].values, calc


def window(dates, calc, d0, d1):
    m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1) + np.timedelta64(1, 'D') - np.timedelta64(1, 'ns'))
    return calc[m]


def flag_rate(train, ev, clip, norm):
    if norm == 'point':
        lo, hi = train.min(0), train.max(0)
    else:
        lo, hi = float(train.min()), float(train.max())

    def nz(X):
        x = (X - lo) / (hi - lo + 1e-8)
        return np.clip(x, 0.0, 1.0) if clip else x
    ntr = nz(train); mu = ntr.mean(0); P = np.linalg.pinv(np.cov(ntr.T))
    dtr = np.einsum('ij,jk,ik->i', ntr - mu, P, ntr - mu)
    thr = np.percentile(dtr, BUDGET)
    nev = nz(ev); d = np.einsum('ij,jk,ik->i', nev - mu, P, nev - mu)
    outside = float((((ev - lo) / (hi - lo + 1e-8) < 0) | ((ev - lo) / (hi - lo + 1e-8) > 1)).any(1).mean())
    return float((d >= thr).mean() * 100), outside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--norm', choices=['point', 'scalar'], default='point')
    ap.add_argument('--train', default='all', help="'all' = whole pre-event window, or an int = last N curves")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sub = pd.read_csv(SUBMITTED)
    prm = pd.read_csv(PARAM)
    mod = {r.ItemName.upper(): int(r.module) for r in prm.itertuples()}
    rows = []
    for lab, a_name in ITEMS.items():
        dates, calc = load(a_name)
        tr = window(dates, calc, TR0, TR1)
        if args.train != 'all':
            tr = tr[-int(args.train):]
        ev = window(dates, calc, EV0, EV1)
        f_on, out_frac = flag_rate(tr, ev, True, args.norm)
        f_off, _ = flag_rate(tr, ev, False, args.norm)
        s = sub[sub['item'] == lab].iloc[0]
        rows.append(dict(item=lab, main=int(s['main']), sub=int(s['sub']), module=mod.get(lab, np.nan),
                         n_train=len(tr), n_ev=len(ev), n_ev_submitted=int(s['n_ev']),
                         flag_pct_submitted=float(s['flag_pct']), flag_pct_clip=f_on, flag_pct_noclip=f_off,
                         ev_curves_outside_range=out_frac))
        print(f'{lab:6} n_tr={len(tr):5} n_ev={len(ev):5}/{int(s["n_ev"]):5} '
              f'sub={s["flag_pct"]:6.2f} clip={f_on:6.2f} noclip={f_off:6.2f} outside={out_frac:.3f}', flush=True)
    df = pd.DataFrame(rows)
    tag = f'{args.norm}_train{args.train}'
    df.to_csv(os.path.join(OUT, f'event_wavelength_{tag}.csv'), index=False)
    for col in ['flag_pct_submitted', 'flag_pct_clip', 'flag_pct_noclip']:
        rho, p = spearmanr(df['main'], df[col])
        m2 = df[df.module == 2][col]; m1 = df[df.module == 1][col]
        print(f'{col:20} rho={rho:.3f} p={p:.3f}  mod2 {m2.min():.1f}-{m2.max():.1f} (n={len(m2)})  '
              f'mod1 {m1.min():.2f}-{m1.max():.2f} (n={len(m1)})  separated={m2.min() > m1.max()}')
    print('max |clip - submitted| =', float((df.flag_pct_clip - df.flag_pct_submitted).abs().max()))


if __name__ == '__main__':
    main()
