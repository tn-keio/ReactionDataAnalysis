# Stage 1m: is the fixed-reference excess of s1k a cross-analyzer transfer artefact?
#
# s1k fits the fixed reference on the earliest 500 curves of each assay with the analyzers POOLED and
# scores every later curve pooled.  The earliest 500 curves come from analyzer #2 only (#1/#3 start in
# mid-2022), so from mid-2022 on the s1k reference also scores foreign-analyzer curves, which S4 shows
# can inflate flag rates (ALB, HDL-C, Ca, LDL-C).  s1k's "62 unexplained months" and the "up to 19x
# stale-vs-recent" ratio therefore cannot be read as pure temporal drift until the analyzer is controlled.
#
# Arms (all: N_TRAIN = 500, empirical covariance, self-calibrated P99.5, scalar min-max WITHOUT clip):
#   B  pooled s1k model (earliest 500, analyzers pooled), later curves scored, reported PER ANALYZER.
#      Summed over analyzers it must reproduce s1k's monthly.csv exactly (--check-s1k).
#   A  per-analyzer fixed reference: each analyzer's own earliest 500 curves, scores its own later curves.
#   S  synchronized per-analyzer reference: each analyzer's first 500 curves on/after T, where T is the
#      date all three analyzers have started for that assay (equal anchor age).
#   D  contemporaneous transfer: #2's S-model scores #1/#3 on exactly the curves their own S-model scores,
#      so D - S isolates the analyzer difference at equal model age.
#   C  per-analyzer rolling comparator: for each month, fit on the analyzer's last 500 curves before the
#      month and score that month's curves of the same analyzer (recent vs stale at the same N).
# Output: Results/20260924_Rev_FixedReference_PerAnalyzer/{monthly_<item>.csv, anchors_<item>.csv,
#         monthly.csv, summary.csv, recency_ratio.csv}
# py3.14 (no TensorFlow).  One process per item: --items <ITEM>; then --summarize.

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
FORPAPER = os.path.join(BASE, 'ForPaper')
for p in (os.path.join(FORPAPER, 'Suppl10_ReviewBurden'), FORPAPER, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import measure_review_burden_unified as B      # noqa: E402
from _common import panel as P                 # noqa: E402
import s1k_fixed_reference_panel as K          # noqa: E402  (KNOWN, MIN_MONTH_N, apply_norm_noclip)

OUT = os.path.join(BASE, 'Results', '20260924_Rev_FixedReference_PerAnalyzer')
S1K = os.path.join(BASE, 'Results', '20260918_Rev_FixedReference_Panel21')
N_TRAIN = 500
ANALYZERS = ['#1', '#2', '#3']
OPTICAL = (pd.Timestamp('2021-07-25'), pd.Timestamp('2021-09-05'))   # S9 exclusion window (S10b)


def load_lean(a_name):
    """(dates, calc, analyzer) in the same order as s1k.load_lean (quicksort on dates)."""
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    d = ds.dataset(B.PROC, partitioning='hive')
    tb = d.to_table(columns=['S_Date', 'SystemName'] + B.RAM + B.RAS, filter=pc.field('A_Name') == a_name)
    raw = pd.to_datetime(tb.column('S_Date').to_numpy(zero_copy_only=False)).values
    order = np.argsort(raw, kind='quicksort')
    dates = pd.to_datetime(raw[order])
    sysn = np.asarray(tb.column('SystemName').to_pylist(), dtype=object)[order]
    anl = np.array([str(s).replace('LABOSPECT 008 ALPHA ', '') for s in sysn], dtype=object)
    calc = np.empty((tb.num_rows, len(B.RAM)), dtype='float64')
    for j, (cm, cs) in enumerate(zip(B.RAM, B.RAS)):
        calc[:, j] = tb.column(cm).to_numpy(zero_copy_only=False)[order].astype('float64')
        calc[:, j] -= tb.column(cs).to_numpy(zero_copy_only=False)[order].astype('float64')
    del tb, order
    return dates, calc, anl


def fit(train):
    lo, hi = B.norm_fit(train)
    W = K.apply_norm_noclip(train, lo, hi)
    e = B.EmpiricalCovariance().fit(W)
    thr = np.percentile(B.maha(e.location_, e.covariance_, W), B.BUDGET)
    return lo, hi, e.location_, e.covariance_, thr


def score(model, X):
    lo, hi, loc, cov, thr = model
    return B.maha(loc, cov, K.apply_norm_noclip(X, lo, hi)) >= thr


def monthly(item, arm, analyzer, dates, flags):
    if len(flags) == 0:
        return None
    df = pd.DataFrame({'month': [str(m) for m in np.asarray(pd.DatetimeIndex(dates).to_period('M'))],
                       'flag': flags})
    g = df.groupby('month').agg(n=('flag', 'size'), k=('flag', 'sum')).reset_index()
    g['rate'] = g['k'] / g['n']
    g.insert(0, 'analyzer', analyzer)
    g.insert(0, 'arm', arm)
    g.insert(0, 'item', item)
    return g


def run_item(item):
    dates, calc, anl = load_lean(item)
    rows, anchors = [], []

    def anchor(arm, a, idx):
        d0, d1 = dates[idx[0]], dates[idx[-1]]
        anchors.append(dict(item=item, arm=arm, analyzer=a, train_start=d0.date(), train_end=d1.date(),
                            overlaps_optical=bool(d0 <= OPTICAL[1] and d1 >= OPTICAL[0])))

    # B: pooled s1k model, scored per analyzer
    mB = fit(calc[:N_TRAIN])
    anchor('B', 'pooled', np.arange(N_TRAIN))
    fB = score(mB, calc[N_TRAIN:])
    for a in ANALYZERS:
        m = anl[N_TRAIN:] == a
        rows.append(monthly(item, 'B', a, dates[N_TRAIN:][m], fB[m]))

    # A: own earliest 500 per analyzer
    starts = {}
    for a in ANALYZERS:
        idx = np.where(anl == a)[0]
        if len(idx) < 2 * N_TRAIN:
            continue
        starts[a] = dates[idx[0]]
        anchor('A', a, idx[:N_TRAIN])
        mA = fit(calc[idx[:N_TRAIN]])
        rest = idx[N_TRAIN:]
        rows.append(monthly(item, 'A', a, dates[rest], score(mA, calc[rest])))

    # S and D: synchronized anchor T = the date by which every analyzer has started
    S_models, S_rest = {}, {}
    if len(starts) == len(ANALYZERS):
        T = max(starts.values())
        for a in ANALYZERS:
            idx = np.where((anl == a) & np.asarray(dates >= T))[0]
            if len(idx) < 2 * N_TRAIN:
                continue
            anchor('S', a, idx[:N_TRAIN])
            S_models[a] = fit(calc[idx[:N_TRAIN]])
            S_rest[a] = idx[N_TRAIN:]
            rows.append(monthly(item, 'S', a, dates[S_rest[a]], score(S_models[a], calc[S_rest[a]])))
        if '#2' in S_models:
            for a in ('#1', '#3'):
                if a in S_rest:
                    r = S_rest[a]
                    rows.append(monthly(item, 'D', a, dates[r], score(S_models['#2'], calc[r])))

    # C: per-analyzer rolling (last 500 before each month, score that month)
    per = np.asarray(pd.DatetimeIndex(dates).to_period('M'))
    for a in ANALYZERS:
        idx = np.where(anl == a)[0]
        if len(idx) < 2 * N_TRAIN:
            continue
        pa = per[idx]
        out_d, out_f = [], []
        for mth in pd.unique(pa):
            in_m = idx[pa == mth]
            prior = idx[dates[idx] < mth.start_time]
            if len(prior) < N_TRAIN:
                continue
            mC = fit(calc[prior[-N_TRAIN:]])
            out_d.append(dates[in_m]); out_f.append(score(mC, calc[in_m]))
        if out_d:
            rows.append(monthly(item, 'C', a, np.concatenate([np.asarray(x) for x in out_d]),
                                np.concatenate(out_f)))

    df = pd.concat([r for r in rows if r is not None], ignore_index=True)
    df['known_event'] = [m in K.KNOWN.get(item, []) or m in K.KNOWN['ALL'] for m in df['month']]
    df.to_csv(os.path.join(OUT, f'monthly_{item}.csv'), index=False)
    pd.DataFrame(anchors).to_csv(os.path.join(OUT, f'anchors_{item}.csv'), index=False)
    q = df[(df.n >= K.MIN_MONTH_N)]
    msg = []
    for arm in ('B', 'A', 'S', 'D', 'C'):
        x = q[q.arm == arm]
        if len(x):
            msg.append(f"{arm}: >=20% {int((x.rate >= .2).sum())} ({int(((x.rate >= .2) & ~x.known_event).sum())} unexpl)")
    print(f'  {item}: ' + '; '.join(msg), flush=True)


def wilson_lower(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / den


def check_s1k():
    """Arm B summed over analyzers must equal s1k's monthly rates (same model, same curves)."""
    ok_all = True
    for f in sorted(glob.glob(os.path.join(OUT, 'monthly_*.csv'))):
        item = os.path.basename(f)[len('monthly_'):-4]
        ref = os.path.join(S1K, f'monthly_{item}.csv')
        if item in ('', 'all') or not os.path.exists(ref):
            continue
        b = pd.read_csv(f)
        b = b[b.arm == 'B'].groupby('month').agg(n=('n', 'sum'), k=('k', 'sum')).reset_index()
        b['rate'] = b.k / b.n
        r = pd.read_csv(ref)
        m = r.merge(b, on='month', suffixes=('_s1k', '_B'))
        ok = len(m) == len(r) and (m.n_s1k == m.n_B).all() and np.allclose(m.rate_s1k, m.rate_B, atol=1e-12)
        ok_all &= bool(ok)
        print(f'  {item}: s1k months {len(r)}, matched {len(m)}, identical={ok}')
    print('ALL IDENTICAL' if ok_all else 'MISMATCH FOUND')


def summarize():
    files = [f for f in sorted(glob.glob(os.path.join(OUT, 'monthly_*.csv')))
             if os.path.basename(f) != 'monthly_all.csv']
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(OUT, 'monthly_all.csv'), index=False)
    df = df[df.n >= K.MIN_MONTH_N].copy()
    df['wl'] = wilson_lower(df.k.values, df.n.values)
    rows = []
    for (it, arm, a), g in df.groupby(['item', 'arm', 'analyzer']):
        r = dict(item=it, arm=arm, analyzer=a, months=len(g), median_rate=g.rate.median())
        for lv in (0.20, 0.05):
            t = int(lv * 100)
            hi = g[g.rate >= lv]
            r[f'n_ge{t}'] = len(hi)
            r[f'n_ge{t}_unexpl'] = int((~hi.known_event).sum())
            # sensitivity: n >= 100 and Wilson 95% lower bound above the level
            hs = g[(g.n >= 100) & (g.wl >= lv)]
            r[f'n_ge{t}_strict'] = len(hs)
            r[f'n_ge{t}_strict_unexpl'] = int((~hs.known_event).sum())
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, 'summary.csv'), index=False)

    cols = ['months', 'n_ge20', 'n_ge20_unexpl', 'n_ge20_strict', 'n_ge20_strict_unexpl',
            'n_ge5', 'n_ge5_unexpl', 'n_ge5_strict', 'n_ge5_strict_unexpl']
    print('PANEL TOTAL by arm (analyzer-months):')
    print(s.groupby('arm')[cols].sum().to_string())
    print('\nunexplained >=20% months by item x arm (analyzer-months):')
    piv = s.pivot_table(index='item', columns='arm', values='n_ge20_unexpl', aggfunc='sum').fillna(0).astype(int)
    print(piv[piv.sum(axis=1) > 0].to_string())

    # Transfer at equal age: D vs S on identical curves
    ds_ = df[df.arm.isin(['S', 'D'])].pivot_table(index=['item', 'analyzer', 'month'], columns='arm',
                                                  values='rate').dropna()
    if len(ds_):
        t = ds_.groupby('item').agg(S_median=('S', 'median'), D_median=('D', 'median'))
        t['D_minus_S'] = t.D_median - t.S_median
        t.to_csv(os.path.join(OUT, 'transfer_equal_age.csv'))
        print('\ncontemporaneous transfer (#2 model on #1/#3 vs own model, same curves), median monthly rate:')
        print(t.sort_values('D_minus_S', ascending=False).round(4).to_string())

    # Recency ratio per analyzer: fixed (A) vs rolling (C), same analyzer, same months
    ac = df[df.arm.isin(['A', 'C'])].pivot_table(index=['item', 'analyzer', 'month'], columns='arm',
                                                 values='rate').dropna()
    rr = ac.groupby(['item', 'analyzer']).agg(A_median=('A', 'median'), C_median=('C', 'median'))
    rr['ratio'] = rr.A_median / rr.C_median.replace(0, np.nan)
    rr.to_csv(os.path.join(OUT, 'recency_ratio.csv'))
    per_item = rr.groupby('item').agg(A_median=('A_median', 'median'), C_median=('C_median', 'median'),
                                      ratio_min=('ratio', 'min'), ratio_median=('ratio', 'median'),
                                      ratio_max=('ratio', 'max'))
    print('\nrecency ratio (per-analyzer fixed A / per-analyzer rolling C), summarized over analyzers:')
    print(per_item.sort_values('ratio_median', ascending=False).round(3).to_string())

    an = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, 'anchors_*.csv')))])
    an.to_csv(os.path.join(OUT, 'anchors_all.csv'), index=False)
    print('\ntraining windows overlapping the 2021 optical event:')
    o = an[an.overlaps_optical]
    print(o.to_string(index=False) if len(o) else '  none')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=None)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--check-s1k', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.check_s1k:
        return check_s1k()
    if a.summarize:
        return summarize()
    for it in (a.items or P.proc_names(21)):
        if os.path.exists(os.path.join(OUT, f'monthly_{it}.csv')):
            print(f'  {it}: exists, skip', flush=True)
            continue
        run_item(it)


if __name__ == '__main__':
    main()
