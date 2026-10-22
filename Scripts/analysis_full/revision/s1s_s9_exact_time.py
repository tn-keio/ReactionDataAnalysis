# Stage 1s: S9 with exact reagent-lot onsets (per-test measurement time M_Date from the raw data).
#
# s1q aligned lot events to the calendar day of the raw 'effective start', and ordered the curves within a day
# by file order (the processed data carry no time of day), so the 500 'post' curves could include old-lot curves
# measured earlier that day.  Here every processed curve gets its measurement time from s1r (join by analyzer,
# sample ID, date and assay; the 0.2% of keys with repeated measurements are resolved by the reported value and
# then by order), the curves are ordered by time, and a lot event starts at the first measurement of the new lot
# on its effective-start day.  Maintenance (date only) and placebo dates start at 00:00 of the day.  Everything
# else is the S9 code path (EV.eval_at, blocks 500/500/500, self-calibrated P99.5, unclipped maha_eval, placebo
# rules, optical-event exclusion), as in s1q.
# Output: Results/20260924_Rev_S9_RawLot/exact_time/{event_robustness_results.csv, join_report.csv, summary.txt}
# py3.14.

import os
import sys
import csv
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, '..'))):
    if p not in sys.path:
        sys.path.insert(0, p)
import event_robustness_analysis as EV      # noqa: E402
import s1q_s9_rawlot as Q                   # noqa: E402  (sets EV.maha_eval to the unclipped version)
import s1p_lot_reanalysis as L              # noqa: E402
import s1n_cluster_classification as N      # noqa: E402

TIMES = os.path.join(EV.BASE, 'Results', '20260924_Rev_TimeFromRaw', 'times_rows.parquet')
OUT = os.path.join(Q.OUT, 'exact_time')
FIELDS = EV.FIELDS + ['lot', 'post_newlot_share', 't_exact']


def attach_time(item, df):
    """add M_Date and lot to the processed rows (df has sys, S_ID, d, Value)"""
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    t = ds.dataset(TIMES).to_table(filter=pc.field('A_Name') == item).to_pandas()
    t['day'] = t['S_Date'].dt.normalize()
    t['v'] = pd.to_numeric(t['Value'], errors='coerce').astype('float32').round(3)
    df = df.copy()
    df['day'] = df['d'].dt.normalize()
    df['v'] = df['Value'].astype('float32').round(3)
    df['_row'] = np.arange(len(df))
    k = ['sys', 'S_ID', 'day']
    t = t.rename(columns={'analyzer': 'sys'})
    t['S_ID'] = t['S_ID'].astype(str).str.strip()
    df['S_ID'] = df['S_ID'].astype(str).str.strip()
    # rank within (key, value) so that repeated measurements pair one-to-one, earliest with earliest
    t = t.sort_values(k + ['v', 'M_Date'])
    t['r'] = t.groupby(k + ['v'], dropna=False).cumcount()
    df = df.sort_values(k + ['v', '_row'])
    df['r'] = df.groupby(k + ['v'], dropna=False).cumcount()
    m = df.merge(t[k + ['v', 'r', 'M_Date', 'R1_Lot']], on=k + ['v', 'r'], how='left')
    nkey = df.groupby(k).size()
    rep = dict(item=item, n=len(df), matched=float(m['M_Date'].notna().mean()),
               rows_in_repeated_keys=float(nkey[nkey > 1].sum() / len(df)))
    return m.sort_values('_row').reset_index(drop=True), rep


def process_item(item, maint, ev_log, writer, reps):
    df = pd.read_parquet(EV.PROC, columns=['S_Date', 'SystemName', 'S_ID', 'Value'] + EV.RAM + EV.RAS,
                         filters=[('A_Name', '==', item)])
    df['d'] = pd.to_datetime(df['S_Date'])
    df['sys'] = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    df, rep = attach_time(item, df)
    reps.append(rep)
    raw = L.raw_lots(item)
    lotd = Q.raw_lot_dates(item, raw, 'raw_eff')
    cald = Q.cal_only_dates(item, ev_log, lotd)
    rng = np.random.RandomState(EV.SEED)
    rows = []
    for s in EV.ANALYZERS:
        sub = df[(df['sys'] == s) & df['M_Date'].notna()].sort_values('M_Date', kind='mergesort')
        if len(sub) < EV.N_TRAIN + EV.N_PRE + EV.M_POST:
            continue
        times = sub['M_Date'].values.astype('datetime64[ns]')
        calc = sub[EV.RAM].values.astype('float64') - sub[EV.RAS].values.astype('float64')
        curlot = sub['R1_Lot'].astype(str).str.strip().to_numpy()
        maint_all = sorted(maint[s]['cell_lamp'] + maint[s]['mechanism'])
        lot_dates = [t for t, _ in lotd[s]]

        def add(etype, t, other, lot='', t_exact=None):
            n0 = len(rows)
            ok = EV.eval_at(times, calc, t_exact if t_exact is not None else t, etype, t,
                            EV.near(other, t, EV.COOCCUR_DAYS), rows, item, s)
            if ok:
                i = rows[n0]['idx']
                rows[n0]['lot'] = lot
                rows[n0]['post_newlot_share'] = round(float(np.mean(curlot[i:i + EV.M_POST] == lot)), 4) if lot else ''
                rows[n0]['t_exact'] = str(t_exact) if t_exact is not None else ''
            return ok
        for etype, elist in [('cell_lamp', maint[s]['cell_lamp']), ('mechanism', maint[s]['mechanism'])]:
            for t in elist:
                add(etype, pd.Timestamp(t), lot_dates)
        for t, lot in lotd[s]:
            day0, day1 = pd.Timestamp(t), pd.Timestamp(t) + pd.Timedelta(days=1)
            on = (curlot == str(lot)) & (times >= np.datetime64(day0)) & (times < np.datetime64(day1))
            t_exact = pd.Timestamp(times[np.argmax(on)]) if on.any() else None
            add('reagent', day0, maint_all, str(lot), t_exact)
        for t in cald[s]:
            add('calibration', pd.Timestamp(t), maint_all)
        all_ev = [pd.Timestamp(x) for x in maint_all] + lot_dates + list(cald[s])
        lo_d, hi_d = pd.Timestamp(times[0]).normalize(), pd.Timestamp(times[-1]).normalize()
        span = (hi_d - lo_d).days
        got = tries = 0
        while got < EV.N_PLACEBO and tries < 400:
            tries += 1
            t = lo_d + pd.Timedelta(days=int(rng.randint(0, max(1, span))))
            if Q.in_excl(t) or EV.near(all_ev, t, EV.PLACEBO_GAP_DAYS):
                continue
            if add('placebo', t, []):
                got += 1
    for r in rows:
        for c in ('lot', 'post_newlot_share', 't_exact'):
            r.setdefault(c, '')
        writer.writerow(r)
    print(f'  {item}: {len(rows)} rows, time matched {rep["matched"]:.4f}', flush=True)


def summarize():
    r = pd.read_csv(os.path.join(OUT, 'event_robustness_results.csv'))
    d = pd.read_csv(os.path.join(Q.OUT, 'raw_eff', 'event_robustness_results.csv'))
    lines = []

    def pr(x=''):
        print(x); lines.append(str(x))
    pr('=== exact time (this run) vs day-level raw_eff (s1q)')
    for et in ['reagent', 'cell_lamp', 'mechanism', 'calibration', 'placebo']:
        a, b = r[r.event_type == et], d[d.event_type == et]
        if len(a):
            mx = a.loc[a.post_rate.idxmax()]
            pr(f'  {et:12s} exact n={len(a):4d} surge>=5% {100 * (a.post_rate >= .05).mean():.2f}% >=2% {100 * (a.post_rate >= .02).mean():.1f}% '
               f'max {100 * mx.post_rate:.1f}% ({mx["item"]} {mx.analyzer} {mx.event_date}) | day-level n={len(b)} '
               f'surge {100 * (b.post_rate >= .05).mean():.2f}%')
    re = r[r.event_type == 'reagent']
    sh = pd.to_numeric(re.post_newlot_share, errors='coerce')
    shd = pd.to_numeric(d[d.event_type == 'reagent'].post_newlot_share, errors='coerce')
    pr(f'  post-block share on the new lot: exact median {sh.median():.2f} (>=0.8 in {100 * (sh >= .8).mean():.0f}%); '
       f'day-level median {shd.median():.2f} (>=0.8 in {100 * (shd >= .8).mean():.0f}%)')
    hi = re[sh >= .8]
    pr(f'  exact, share >= 0.8: n={len(hi)} surge {100 * (hi.post_rate >= .05).mean():.2f}%')
    k = ['analyzer', 'item', 'event_date']
    m = re.merge(d[d.event_type == 'reagent'][k + ['post_rate']], on=k, suffixes=('_exact', '_day'))
    pr(f'  paired reagent events n={len(m)}: median post rate exact {100 * m.post_rate_exact.median():.2f}% vs day {100 * m.post_rate_day.median():.2f}%; '
       f'surge exact {int((m.post_rate_exact >= .05).sum())} vs day {int((m.post_rate_day >= .05).sum())}; '
       f'both {int(((m.post_rate_exact >= .05) & (m.post_rate_day >= .05)).sum())}')
    u = re.assign(s5=re.post_rate >= .05).groupby(['item', 'lot']).s5.agg(['any', 'mean'])
    pr(f'  per item-lot: {int(u["any"].sum())}/{len(u)} lots surge on >=1 analyzer; mean per-lot surge fraction {100 * u["mean"].mean():.2f}%')
    by = re.groupby('item').apply(lambda g: f'{int((g.post_rate >= .05).sum())}/{len(g)}', include_groups=False)
    pr('  reagent by item: ' + ', '.join(f'{i} {v}' for i, v in by.items()))
    s5 = re[re.post_rate >= .05].sort_values('post_rate', ascending=False).head(15)
    pr('  top surges: ' + '; '.join(f'{x.item} {x.analyzer} {x.event_date} lot {x.lot} {100 * x.post_rate:.1f}%' for x in s5.itertuples()))
    with open(os.path.join(OUT, 'summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def main():
    os.makedirs(OUT, exist_ok=True)
    maint = EV.load_maint()
    ev_log = N.load_events()
    reps = []
    with open(os.path.join(OUT, 'event_robustness_results.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for item in EV.PANEL:
            process_item(item, maint, ev_log, w, reps)
            fh.flush()
    pd.DataFrame(reps).to_csv(os.path.join(OUT, 'join_report.csv'), index=False)
    print(pd.DataFrame(reps).round(4).to_string(index=False))
    summarize()


if __name__ == '__main__':
    main()
