# Stage 1d: rolling recent-500 models evaluated on SUBSEQUENT,
# NON-OVERLAPPING local curves, with anomalies injected into those same curves.
#
# Design (per 21-panel item, analyzers pooled as in the submitted Section 3.10):
#   monthly anchor t (first curve dated >= month start)
#   train  W = the 500 curves immediately before t            (rolling recent-500)
#   test   S = the 500 curves immediately after t              (disjoint from W; its calendar span is logged)
#   threshold = self-calibrated P99.5 of the training scores   (deployment operating point)
#   routine flag rate = fraction of the REAL curves in S flagged (no screening -> what a lab would review)
#   recall = on a battery generated from S itself (contemporaneous_synth.make_battery), types A/B/C/E
#            at k in K_GRID (incl. sub-1-sigma levels) and D (lag 1-5); clean synthetic flag rate too.
#   base-curve screening: both screen_frac=None (unscreened) and 0.9 (inlier-screened) are reported.
#   normalization: scalar min-max from W, clip ON (submitted behaviour) and OFF (revised manuscript).
# Methods: E (empirical Mahalanobis, proposed), R (robust, iterative trimming), P (PCA 95%).
#
# Reuses (no modification): run_temporal_robustness_screened.py (mcd_like, maha, norm_fit),
# contemporaneous_synth.py (window width patched from 28 to 34 points at run time).
# Output: Results/20260917_Rev_R2_RollingContemporaneous/{series.csv, summary_by_item.csv, run_config.json}
# py3.14 (no TensorFlow).  Resumable per item (series_<item>.csv).

import os
import sys
import json
import gc
import argparse
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, '..'))
BASE = os.path.abspath(os.path.join(EXP, '..', '..'))
for p in (EXP, os.path.join(BASE, 'ForPaper')):
    if p not in sys.path:
        sys.path.insert(0, p)
import contemporaneous_synth as CS              # noqa: E402
import run_temporal_robustness_screened as TS   # noqa: E402
from _common import panel as P                  # noqa: E402

P0 = 5
PTS = list(range(P0, 39))
RAM = [f'RAM{i}' for i in PTS]
RAS = [f'RAS{i}' for i in PTS]
CS.T = len(PTS)                  # 34: module globals are read at call time
CS.IDX = np.arange(CS.T)

N = 500
M_TEST = 500
BUDGET = 99.5
K_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 8]
SCREENS = {'unscreened': None, 'screened90': 0.9}
CLIPS = {'clip': True, 'noclip': False}
METHODS = ['E', 'R', 'P']
POS = CS.POS_TYPES
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
OUT = os.path.join(BASE, 'Results', '20260917_Rev_R2_RollingContemporaneous')


def norm(a, lo, hi, clip):
    z = (a - lo) / (hi - lo + 1e-8)
    return np.clip(z, 0.0, 1.0) if clip else z


def fit(Wn, method):
    if method == 'E':
        e = EmpiricalCovariance().fit(Wn)
        loc, cov = e.location_, e.covariance_
        return lambda X: TS.maha(loc, cov, X)
    if method == 'R':
        loc, cov = TS.mcd_like(Wn)
        return lambda X: TS.maha(loc, cov, X)
    m = PCA(n_components=0.95).fit(Wn)
    return lambda X: np.mean((X - m.inverse_transform(m.transform(X))) ** 2, axis=1)


def run_item(proc):
    d = ds.dataset(PROC, partitioning='hive')
    tb = d.to_table(columns=['S_Date'] + RAM + RAS, filter=pc.field('A_Name') == proc).to_pandas()
    tb = tb.sort_values('S_Date', kind='mergesort').reset_index(drop=True)
    dates = pd.to_datetime(tb['S_Date']).values.astype('datetime64[ns]')
    ram = tb[RAM].values.astype('float64'); ras = tb[RAS].values.astype('float64')
    del tb; gc.collect()
    calc = ram - ras; n = len(calc)
    months = pd.period_range(pd.Timestamp(dates.min()).to_period('M') + 1,
                             pd.Timestamp(dates.max()).to_period('M'), freq='M')
    rows = []
    for mo in months:
        cut = int(np.searchsorted(dates, np.datetime64(mo.start_time, 'ns'), side='left'))
        if cut < N or cut + M_TEST > n:
            continue
        if pd.Timestamp(dates[cut]).to_period('M') != mo:   # month without curves -> would duplicate the next block
            continue
        W = calc[cut - N:cut]; S = calc[cut:cut + M_TEST]
        lo, hi = TS.norm_fit(W)
        span_train = (dates[cut - 1] - dates[cut - N]) / np.timedelta64(1, 'D')
        span_test = (dates[cut + M_TEST - 1] - dates[cut]) / np.timedelta64(1, 'D')
        batts = {sk: CS.make_battery(ram[cut:cut + M_TEST], ras[cut:cut + M_TEST], seed=0, k_grid=K_GRID,
                                     include_d=True, include_benign=False, screen_frac=sf)
                 for sk, sf in SCREENS.items()}
        S_raw = norm(S, lo, hi, False)
        clip_frac_real = float(((S_raw < 0) | (S_raw > 1)).any(1).mean())   # test curves np.clip would alter
        for ck, clip in CLIPS.items():
            Wn = norm(W, lo, hi, clip); Sn = norm(S, lo, hi, clip)
            for m in METHODS:
                f = fit(Wn, m)
                thr = float(np.percentile(f(Wn), BUDGET))
                row = dict(item=proc, month=str(mo), cut=cut, clip=ck, method=m,
                           train_span_d=round(float(span_train), 2), test_span_d=round(float(span_test), 2),
                           test_curves_outside01=clip_frac_real,
                           routine_flag_rate=float((f(Sn) >= thr).mean()))
                for sk, b in batts.items():
                    at, sv = b['atype'], b['severity']
                    sc = f(norm(b['calc'], lo, hi, clip)) >= thr
                    pos = np.isin(at, POS)
                    row[f'{sk}_clean_flag'] = float(sc[at == 'clean'].mean())
                    row[f'{sk}_recall_main'] = float(sc[pos & (sv >= 2)].mean())
                    row[f'{sk}_recall_k1'] = float(sc[pos & (sv == 1)].mean())
                    for k in K_GRID:
                        if k < 1:
                            row[f'{sk}_recall_k{k}'] = float(sc[pos & np.isclose(sv, k)].mean())
                    for t in POS:
                        row[f'{sk}_{t}_main'] = float(sc[(at == t) & (sv >= 2)].mean())
                        row[f'{sk}_{t}_k1'] = float(sc[(at == t) & (sv == 1)].mean())
                    for L in CS.D_LAG_DEFAULT:
                        row[f'{sk}_D_lag{L}'] = float(sc[(at == 'D_kinetic') & (sv == L)].mean())
                rows.append(row)
    del calc, ram, ras; gc.collect()
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for (it, ck, m), g in df.groupby(['item', 'clip', 'method']):
        r = dict(item=it, clip=ck, method=m, n_anchors=len(g),
                 test_span_d_median=g.test_span_d.median(), test_span_d_max=g.test_span_d.max(),
                 flag_rate_median=g.routine_flag_rate.median(), flag_rate_p90=g.routine_flag_rate.quantile(0.9),
                 flag_rate_max=g.routine_flag_rate.max(),
                 months_flag_gt2pct=int((g.routine_flag_rate > 0.02).sum()),
                 months_flag_gt5pct=int((g.routine_flag_rate > 0.05).sum()))
        for sk in SCREENS:
            for key in (['recall_main', 'recall_k1', 'clean_flag'] + [f'recall_k{k}' for k in K_GRID if k < 1]
                        + [f'D_lag{L}' for L in CS.D_LAG_DEFAULT]):
                c = f'{sk}_{key}'
                r[f'{c}_mean'] = g[c].mean(); r[f'{c}_min'] = g[c].min()
            r[f'{sk}_recall_main_worst_month'] = g.loc[g[f'{sk}_recall_main'].idxmin(), 'month']
        out.append(r)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=P.proc_names(21))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(N=N, M_test=M_TEST, budget=BUDGET, calibration='self (train P99.5)', window=f'{P0}-38',
                   k_grid=K_GRID, screens=list(SCREENS), clips=list(CLIPS), methods=METHODS,
                   analyzers='pooled', test='next 500 curves after the monthly anchor (disjoint from train)',
                   battery='contemporaneous_synth.make_battery(seed=0, T=34) from the test block',
                   date='2026-09-17'), open(os.path.join(OUT, 'run_config.json'), 'w'), indent=2)
    for i, it in enumerate(a.items, 1):
        p = os.path.join(OUT, f'series_{it}.csv')
        if os.path.exists(p):
            print(f'[{i}/{len(a.items)}] {it}: done, skip', flush=True); continue
        print(f'[{i}/{len(a.items)}] {it} ...', flush=True)
        df = run_item(it)
        df.to_csv(p, index=False)
        # NB: bracket access -- DataFrame.clip is a method, so df.clip would not select the column
        e = df[(df['method'] == 'E') & (df['clip'] == 'noclip')]
        print(f'   anchors={len(e)} E/noclip flag med={e["routine_flag_rate"].median():.4f} '
              f'recall_main min(unscr)={e["unscreened_recall_main"].min():.3f} '
              f'min(scr)={e["screened90_recall_main"].min():.3f}', flush=True)
    allp = [os.path.join(OUT, f'series_{it}.csv') for it in P.proc_names(21)
            if os.path.exists(os.path.join(OUT, f'series_{it}.csv'))]
    series = pd.concat([pd.read_csv(p) for p in allp], ignore_index=True)
    series.to_csv(os.path.join(OUT, 'series.csv'), index=False)
    summarize(series).to_csv(os.path.join(OUT, 'summary_by_item.csv'), index=False)
    print('DONE ->', OUT, flush=True)


if __name__ == '__main__':
    main()
