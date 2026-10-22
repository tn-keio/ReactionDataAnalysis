# Stage 1k: how often does the FIXED-REFERENCE monitor raise a cluster that is not a known event?
#
# The revision claims a two-level design: the rolling model screens single curves (sporadic errors),
# and a cluster of flags under a FIXED reference indicates a systematic change (reagent lot, optics).
# The question is what that costs: if every month produced a cluster the laboratory could not act
# on it.  This script therefore measures, for all 21 panel assays over the whole record, how many
# months exceed the alert levels used in S14 (>=20% of curves flagged; >=5% as a looser level), and
# separates months that coincide with an independently documented event from those that do not.
#
# Protocol (identical to ForPaper/Rev11_FixedReference_IgEvent/probe_fixed_reference.py):
#   fit once on the earliest 500 curves of the assay, score every subsequent curve, monthly flag rate,
#   months with >= MIN_MONTH_N curves.  Normalization WITHOUT clipping.
# Memory: the wrapped loader materialises a whole assay several times over; a lean loader is used here
# and validated against it (--verify-loader), because the OS killed the earlier combined runs.
# Output: Results/20260918_Rev_FixedReference_Panel21/{monthly.csv, summary.csv}
# py3.14 (no TensorFlow).  One process per item is recommended: --items <ITEM>

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
FORPAPER = os.path.join(BASE, 'ForPaper')
for p in (os.path.join(FORPAPER, 'Suppl10_ReviewBurden'), FORPAPER):
    if p not in sys.path:
        sys.path.insert(0, p)
import measure_review_burden_unified as B      # noqa: E402  (norm_fit / maha / EmpiricalCovariance)
from _common import panel as P                 # noqa: E402

OUT = os.path.join(BASE, 'Results', '20260918_Rev_FixedReference_Panel21')
N_TRAIN = 500
MIN_MONTH_N = 30
LEVELS = (0.20, 0.05)

# Independently documented events (S10b, S14.4): months in which an elevated flag rate is EXPECTED.
KNOWN = {
    'ALL': ['2021-07', '2021-08', '2021-09'],          # module-2 optical event (S10b), all module-2 assays
    'IgM': [f'2023-{m:02d}' for m in range(6, 13)] + ['2024-01', '2024-02', '2024-03', '2024-04'],
    'IgG': ['2024-12', '2025-01', '2025-02', '2025-03'],
    'IgA': ['2024-03', '2024-04'],
}


def apply_norm_noclip(a, lo, hi):
    return (a - lo) / (hi - lo + 1e-8)


def load_lean(a_name):
    """(dates, calc) identical to B.load but built column by column (peak memory ~ one matrix)."""
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    d = ds.dataset(B.PROC, partitioning='hive')
    tb = d.to_table(columns=['S_Date'] + B.RAM + B.RAS, filter=pc.field('A_Name') == a_name)
    raw = pd.to_datetime(tb.column('S_Date').to_numpy(zero_copy_only=False)).values
    order = np.argsort(raw, kind='quicksort')        # pandas sort_values default; ties must match
    dates = pd.to_datetime(raw[order])
    calc = np.empty((tb.num_rows, len(B.RAM)), dtype='float64')
    for j, (cm, cs) in enumerate(zip(B.RAM, B.RAS)):
        calc[:, j] = tb.column(cm).to_numpy(zero_copy_only=False)[order].astype('float64')
        calc[:, j] -= tb.column(cs).to_numpy(zero_copy_only=False)[order].astype('float64')
    del tb, order
    return dates, calc


def fixed_reference(dates, calc, anchor_after=None):
    """Fit on the earliest N_TRAIN curves, or on the earliest N_TRAIN curves AFTER anchor_after."""
    if anchor_after is not None:
        keep = np.asarray(dates >= pd.Timestamp(anchor_after))
        dates, calc = dates[keep], calc[keep]
    train = calc[:N_TRAIN]
    lo, hi = B.norm_fit(train)
    W = apply_norm_noclip(train, lo, hi)
    e = B.EmpiricalCovariance().fit(W)
    thr = np.percentile(B.maha(e.location_, e.covariance_, W), B.BUDGET)
    rest = calc[N_TRAIN:]
    d = B.maha(e.location_, e.covariance_, apply_norm_noclip(rest, lo, hi))
    return dates[N_TRAIN:], d >= thr


def periodic_reference(dates, calc, months=12):
    """Re-anchor every `months`: within each window, fit on its first 500 curves and score the rest.

    A single anchor cannot serve assays whose morphology drifts for years (Ca, LDL-C); this is the
    operational alternative, and its cost is that an episode longer than the window is re-absorbed.
    """
    per = pd.PeriodIndex(dates, freq='M')
    start = per.min()
    out_d, out_f = [], []
    while start <= per.max():
        end = start + months
        m = (per >= start) & (per < end)
        idx = np.where(m)[0]
        if len(idx) > N_TRAIN + 100:
            tr, sc = calc[idx[:N_TRAIN]], calc[idx[N_TRAIN:]]
            lo, hi = B.norm_fit(tr)
            W = apply_norm_noclip(tr, lo, hi)
            e = B.EmpiricalCovariance().fit(W)
            thr = np.percentile(B.maha(e.location_, e.covariance_, W), B.BUDGET)
            d = B.maha(e.location_, e.covariance_, apply_norm_noclip(sc, lo, hi))
            out_d.append(dates[idx[N_TRAIN:]]); out_f.append(d >= thr)
        start = end
    if not out_d:
        return dates[:0], np.zeros(0, bool)
    return pd.DatetimeIndex(np.concatenate([np.asarray(x) for x in out_d])), np.concatenate(out_f)


def frozen_before(dates, calc, when):
    """Fit on the last N_TRAIN curves BEFORE `when` and score everything after it.

    This is the 'freeze and hold' response: when a fault is suspected, updating of the rolling window
    is paused and the training set is held at the most recent known-good state.  Used here only as a
    retrospective illustration of what freezing preserves - the date comes from the reagent-lot record,
    not from a prospective trigger.
    """
    t = pd.Timestamp(when)
    before = np.asarray(dates < t)
    idx = np.where(before)[0]
    if len(idx) < N_TRAIN:
        raise SystemExit(f'only {len(idx)} curves before {when}')
    tr = calc[idx[-N_TRAIN:]]
    lo, hi = B.norm_fit(tr)
    W = apply_norm_noclip(tr, lo, hi)
    e = B.EmpiricalCovariance().fit(W)
    thr = np.percentile(B.maha(e.location_, e.covariance_, W), B.BUDGET)
    after = np.where(~before)[0]
    d = B.maha(e.location_, e.covariance_, apply_norm_noclip(calc[after], lo, hi))
    return dates[after], d >= thr


def run_item(item, anchor_after=None, suffix='', periodic=None, freeze_before=None):
    dates, calc = load_lean(item)
    if len(calc) < 3 * N_TRAIN:
        print(f'  {item}: too few curves', flush=True)
        return None
    if freeze_before:
        d2, flag = frozen_before(dates, calc, freeze_before)
    elif periodic:
        d2, flag = periodic_reference(dates, calc, periodic)
    else:
        d2, flag = fixed_reference(dates, calc, anchor_after)
    df = pd.DataFrame({'month': [str(m) for m in np.asarray(d2.to_period('M'))], 'flag': flag})
    g = df.groupby('month').agg(n=('flag', 'size'), rate=('flag', 'mean')).reset_index()
    g = g[g['n'] >= MIN_MONTH_N]
    g.insert(0, 'item', item)
    g['known_event'] = [m in KNOWN.get(item, []) or m in KNOWN['ALL'] for m in g['month']]
    g.to_csv(os.path.join(OUT, f'monthly_{item}{suffix}.csv'), index=False)
    print(f"  {item}: {len(g)} months, >=20%: {int((g.rate >= .2).sum())} "
          f"({int((g[g.rate >= .2].known_event).sum())} on a documented event)", flush=True)
    return g


def summarize():
    files = sorted(glob.glob(os.path.join(OUT, 'monthly_*.csv')))
    if not files:
        raise SystemExit('nothing to summarize')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(OUT, 'monthly.csv'), index=False)
    rows = []
    for it, g in df.groupby('item'):
        g = g.sort_values('month')
        r = dict(item=it, months=len(g), median_rate=g.rate.median())
        for lv in LEVELS:
            hi = g[g.rate >= lv]
            r[f'n_ge{int(lv * 100)}'] = len(hi)
            r[f'n_ge{int(lv * 100)}_known'] = int(hi.known_event.sum())
            r[f'n_ge{int(lv * 100)}_unexplained'] = int((~hi.known_event).sum())
        # assay-specific alert level, as S14.5 already does for the saline blank: the baseline is the
        # assay's own first 12 event-free months, and the level is median + 5 x MAD (floored at +2 pt).
        base = g[~g.known_event].head(12)['rate']
        if len(base) >= 6:
            med = float(base.median())
            mad = float(np.median(np.abs(base - med)))
            lvl = max(med + 5 * mad, med + 0.02)
            hi = g[g.rate >= lvl]
            r.update(baseline_median=med, baseline_mad=mad, adaptive_level=lvl,
                     n_adaptive=len(hi), n_adaptive_known=int(hi.known_event.sum()),
                     n_adaptive_unexplained=int((~hi.known_event).sum()))
        # Moving-baseline (step) rule: compare each month with the median of the previous 6 months of
        # the same assay.  A fixed level and a fixed-baseline level both fail on assays whose reaction
        # morphology drifts over years (Ca, LDL-C, ALB), because a 2021 anchor becomes stale; a step
        # rule asks the operational question instead - did something change recently?
        v = g['rate'].to_numpy()
        alerts = []
        for i in range(6, len(v)):
            prev = v[i - 6:i]
            med = float(np.median(prev))
            mad = float(np.median(np.abs(prev - med)))
            lvl = max(med + 5 * mad, med + 0.02, 0.05)
            if v[i] >= lvl:
                alerts.append(i)
        kn = g['known_event'].to_numpy()
        r.update(n_step=len(alerts), n_step_known=int(sum(kn[i] for i in alerts)),
                 n_step_unexplained=int(sum(not kn[i] for i in alerts)))
        rows.append(r)
    s = pd.DataFrame(rows).sort_values('n_ge20', ascending=False)
    s.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    cols = [c for c in ['months', 'n_ge20', 'n_ge20_known', 'n_ge20_unexplained', 'n_ge5', 'n_ge5_known',
                        'n_ge5_unexplained', 'n_adaptive', 'n_adaptive_known', 'n_adaptive_unexplained',
                        'n_step', 'n_step_known', 'n_step_unexplained']
            if c in s.columns]
    tot = s[cols].sum()
    with pd.option_context('display.width', 240, 'display.max_columns', 30):
        print(s.round(3).to_string(index=False))
    print('\nPANEL TOTAL: {months} assay-months; >=20% in {n_ge20} ({n_ge20_known} on a documented event, '
          '{n_ge20_unexplained} unexplained); >=5% in {n_ge5} ({n_ge5_known} / {n_ge5_unexplained})'
          .format(**tot.to_dict()))
    if 'n_adaptive' in tot:
        print('assay-specific level (own baseline + 5 MAD): {n_adaptive} alerts '
              '({n_adaptive_known} on a documented event, {n_adaptive_unexplained} unexplained)'
              .format(**tot.to_dict()))
    if 'n_step' in tot:
        print('moving-baseline step rule (previous 6 months + 5 MAD, floor 5%): {n_step} alerts '
              '({n_step_known} on a documented event, {n_step_unexplained} unexplained)'
              .format(**tot.to_dict()))
        st = s[['item', 'n_step', 'n_step_known', 'n_step_unexplained']].sort_values('n_step', ascending=False)
        print(st.head(8).to_string(index=False))
    print('unexplained >=20% months:')
    u = df[(df.rate >= 0.20) & (~df.known_event)]
    print(u[['item', 'month', 'n', 'rate']].round(3).to_string(index=False) if len(u) else '  none')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=None)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--freeze-before', default=None,
                    help="fit on the last 500 curves before this date and score everything after "
                         "('freeze and hold' illustration)")
    ap.add_argument('--periodic', type=int, default=None,
                    help='re-anchor every N months instead of once (operational alternative)')
    ap.add_argument('--anchor-after', default=None,
                    help='re-anchor the fixed reference on the first 500 curves after this date '
                         '(e.g. 2021-09-06, i.e. after the documented module-2 optical event)')
    ap.add_argument('--verify-loader', metavar='ITEM', default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.verify_loader:
        d1, c1 = B.load(a.verify_loader)
        d2, c2 = load_lean(a.verify_loader)
        ok = len(d1) == len(d2) and bool((np.asarray(d1) == np.asarray(d2)).all()) and np.array_equal(c1, c2)
        print(f'loader check {a.verify_loader}: n={len(d1)} identical={ok}')
        return
    if a.summarize:
        return summarize()
    if a.freeze_before:
        suffix = '_frozen' + str(a.freeze_before).replace('-', '')
    elif a.periodic:
        suffix = f'_periodic{a.periodic}'
    else:
        suffix = '' if not a.anchor_after else '_reanchored'
    for it in (a.items or P.proc_names(21)):
        if os.path.exists(os.path.join(OUT, f'monthly_{it}{suffix}.csv')):
            print(f'  {it}: exists, skip', flush=True)
            continue
        run_item(it, a.anchor_after, suffix, a.periodic, a.freeze_before)


if __name__ == '__main__':
    main()
