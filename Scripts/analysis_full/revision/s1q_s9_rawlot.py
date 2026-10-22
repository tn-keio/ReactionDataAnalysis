# Stage 1q: S9 (maintenance / reagent-lot robustness) re-run with reagent-lot dates from the raw instrument data.
#
# The submitted S9 (event_robustness_analysis.py; unclipped re-run in s1f) takes lot-change dates from the
# hand-entered reagent log, which has omissions and date offsets.  This script keeps the S9 design and code
# path (EV.eval_at, blocks 500/500/500, self-calibrated P99.5, placebo rules, 2021 optical-event exclusion,
# unclipped maha_eval from s1f) and changes only where the lot dates come from.
#   --source log      : hand-entered log, exactly as S9 -> must reproduce s1f (validation)
#   --source raw_eff  : raw lots (s1o), effective start (>= 10% of the day's curves) incl. re-use after >= 14 d
#   --source raw_start: raw lots, first use only
# Additions:
#   * 'calibration' events = hand-log calibration-only entries with no raw lot event within +-3 d;
#   * share of the 500 post-event curves that are on the new lot (sensitivity: events with share >= 0.8);
#   * paired placebo: the s1f placebo dates that remain > 30 d from every event under the raw lot dates.
#   (Starting the post block at the first new-lot measurement is not possible here, because
#    S_Date carries no time of day.)
# Output: Results/20260924_Rev_S9_RawLot/<source>/event_robustness_results.csv, summary.txt
# py3.14.

import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, '..'))):
    if p not in sys.path:
        sys.path.insert(0, p)
import event_robustness_analysis as EV      # noqa: E402
import s1f_event_interanalyzer_noclip as F  # noqa: E402  (unclipped maha_eval)
import s1p_lot_reanalysis as L              # noqa: E402  (raw_lots, key_map, lot_events)
import s1n_cluster_classification as N      # noqa: E402  (load_events: hand-log parser)

EV.maha_eval = F.ev_maha_eval_noclip
OUT = os.path.join(EV.BASE, 'Results', '20260924_Rev_S9_RawLot')
S1F = os.path.join(EV.BASE, 'Results', '20260917_Rev_M1_S9S4_noclip', 'event', 'event_robustness_results.csv')
FIELDS = EV.FIELDS + ['lot', 'post_newlot_share']


def in_excl(t):
    return EV.EXCL0 <= pd.Timestamp(t) <= EV.EXCL1


def raw_lot_dates(item, raw, source):
    """{analyzer: [(date, lot)]} from raw lots, excluding the optical-event window"""
    out = {}
    for s in EV.ANALYZERS:
        _, _, evdf = L.lot_events(raw, s)
        if evdf is None or evdf.empty:
            out[s] = []
            continue
        if source == 'raw_start':
            evdf = evdf[evdf.kind == 'start']
        out[s] = sorted((pd.Timestamp(r.day), r.lot) for r in evdf.itertuples() if not in_excl(r.day))
    return out


def cal_only_dates(item, ev_log, lotd):
    """hand-log calibration-only entries with no raw lot event within +-3 d"""
    abbr = EV.REAGENT_ABBR[item]
    out = {}
    for s in EV.ANALYZERS:
        e = ev_log[(ev_log.abbr == abbr) & (ev_log.analyzer == s) & (ev_log.kind == 'cal')]
        lots = [t for t, _ in lotd.get(s, [])]
        out[s] = sorted({pd.Timestamp(t).normalize() for t in e.date
                         if not in_excl(t) and not EV.near(lots, pd.Timestamp(t), 3)})
    return out


def process_item(item, source, maint, loglot, ev_log, writer):
    df = pd.read_parquet(EV.PROC, columns=['S_Date', 'SystemName', 'S_ID'] + EV.RAM + EV.RAS,
                         filters=[('A_Name', '==', item)])
    df['d'] = pd.to_datetime(df['S_Date'])
    df['sys'] = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    raw = L.raw_lots(item)
    km = L.key_map(raw)
    abbr = EV.REAGENT_ABBR[item]
    if source == 'log':
        lotd = {s: [(pd.Timestamp(t), '') for t in loglot.get((s, abbr), [])] for s in EV.ANALYZERS}
        cald = {s: [] for s in EV.ANALYZERS}
    else:
        lotd = raw_lot_dates(item, raw, source)
        cald = cal_only_dates(item, ev_log, lotd)
    rng = np.random.RandomState(EV.SEED)
    rows = []
    for s in EV.ANALYZERS:
        sub = df[df['sys'] == s].sort_values('d')          # identical ordering to EV.process_item
        if len(sub) < EV.N_TRAIN + EV.N_PRE + EV.M_POST:
            continue
        dates = sub['d'].values.astype('datetime64[ns]')
        calc = sub[EV.RAM].values.astype('float64') - sub[EV.RAS].values.astype('float64')
        keys = pd.MultiIndex.from_arrays([np.full(len(sub), s, dtype=object), sub['S_ID'].astype(str).str.strip().values,
                                          sub['d'].dt.normalize().values])
        curlot = km.reindex(keys).to_numpy()
        maint_all = sorted(maint[s]['cell_lamp'] + maint[s]['mechanism'])
        lot_dates = [t for t, _ in lotd[s]]

        def add(etype, t, other, lot=''):
            n0 = len(rows)
            ok = EV.eval_at(dates, calc, t, etype, t, EV.near(other, t, EV.COOCCUR_DAYS), rows, item, s)
            if ok:
                i = rows[n0]['idx']
                post = curlot[i:i + EV.M_POST]
                rows[n0]['lot'] = lot
                rows[n0]['post_newlot_share'] = round(float(np.mean(post == lot)), 4) if lot else ''
            return ok
        for etype, elist in [('cell_lamp', maint[s]['cell_lamp']), ('mechanism', maint[s]['mechanism'])]:
            for t in elist:
                add(etype, pd.Timestamp(t), lot_dates)
        for t, lot in lotd[s]:
            add('reagent', pd.Timestamp(t), maint_all, lot)
        for t in cald[s]:
            add('calibration', pd.Timestamp(t), maint_all)
        # placebo (same rules and seed as S9; exclusion uses this source's lot dates and calibration dates)
        all_ev = [pd.Timestamp(x) for x in maint_all] + lot_dates + list(cald[s])
        lo_d, hi_d = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
        span = (hi_d - lo_d).days
        got = tries = 0
        while got < EV.N_PLACEBO and tries < 400:
            tries += 1
            t = lo_d + pd.Timedelta(days=int(rng.randint(0, max(1, span))))
            if in_excl(t) or EV.near(all_ev, t, EV.PLACEBO_GAP_DAYS):
                continue
            if add('placebo', t, []):
                got += 1
    for r in rows:
        r.setdefault('lot', ''); r.setdefault('post_newlot_share', '')
        writer.writerow(r)
    print(f'  {item}: {len(rows)} rows', flush=True)
    return {s: [t for t, _ in lotd[s]] + list(cald[s]) + sorted(pd.Timestamp(x) for x in maint[s]['cell_lamp'] + maint[s]['mechanism'])
            for s in EV.ANALYZERS}


def summarize(source, all_events):
    r = pd.read_csv(os.path.join(OUT, source, 'event_robustness_results.csv'))
    lines = []

    def pr(x=''):
        print(x); lines.append(str(x))
    pr(f'=== source = {source}')
    for et, g in r.groupby('event_type'):
        mx = g.loc[g.post_rate.idxmax()]
        pr(f'  {et:12s} n={len(g):5d}  surge>=5% {100 * (g.post_rate >= .05).mean():.2f}%  >=2% {100 * (g.post_rate >= .02).mean():.1f}%  '
           f'max {100 * mx.post_rate:.1f}% ({mx["item"]} {mx.analyzer} {mx.event_date})')
    re = r[r.event_type == 'reagent']
    if source != 'log' and len(re):
        sh = pd.to_numeric(re.post_newlot_share, errors='coerce')
        pr(f'  reagent: post-block share on the new lot median {sh.median():.2f}; share >= 0.8 in {100 * (sh >= .8).mean():.1f}% of events')
        hi = re[sh >= .8]
        pr(f'  reagent, share >= 0.8 only: n={len(hi)}  surge>=5% {100 * (hi.post_rate >= .05).mean():.2f}%  >=2% {100 * (hi.post_rate >= .02).mean():.1f}%')
        s5 = re[re.post_rate >= .05].sort_values('post_rate', ascending=False)
        pr('  reagent surges (>=5%): ' + '; '.join(f'{x.item} {x.analyzer} {x.event_date} lot {x.lot} {100 * x.post_rate:.1f}%' for x in s5.itertuples()))
        by = re.groupby('item').apply(lambda g: pd.Series({'n': len(g), 'surge5': (g.post_rate >= .05).sum()}), include_groups=False)
        pr('  reagent by item: ' + ', '.join(f'{i} {int(v.surge5)}/{int(v.n)}' for i, v in by.iterrows()))
    if all_events is not None and os.path.exists(S1F):
        f = pd.read_csv(S1F)
        pl = f[f.event_type == 'placebo'].copy()
        keep = [not EV.near(all_events.get((x.item, x.analyzer), []), pd.Timestamp(x.event_date), EV.PLACEBO_GAP_DAYS)
                for x in pl.itertuples()]
        pk = pl[keep]
        pr(f'  paired placebo (s1f placebo dates still valid under {source}): n={len(pk)} of {len(pl)}; '
           f'surge>=5% {100 * (pk.post_rate >= .05).mean():.2f}%  >=2% {100 * (pk.post_rate >= .02).mean():.1f}%')
    with open(os.path.join(OUT, source, 'summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['log', 'raw_eff', 'raw_start'], required=True)
    a = ap.parse_args()
    d = os.path.join(OUT, a.source)
    os.makedirs(d, exist_ok=True)
    maint = EV.load_maint()
    loglot = EV.load_lot()
    ev_log = N.load_events()
    allev = {}
    with open(os.path.join(d, 'event_robustness_results.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for item in EV.PANEL:
            evs = process_item(item, a.source, maint, loglot, ev_log, w)
            for s, v in evs.items():
                allev[(item, s)] = v
            fh.flush()
    if a.source == 'log':
        new = pd.read_csv(os.path.join(d, 'event_robustness_results.csv'))
        old = pd.read_csv(S1F)
        k = ['item', 'analyzer', 'event_type', 'event_date', 'idx']
        o = old.sort_values(k).reset_index(drop=True)
        n = new.sort_values(k).reset_index(drop=True)
        same = (len(o) == len(n) and (o[k].astype(str).values == n[k].astype(str).values).all()
                and np.allclose(o.post_rate, n.post_rate) and np.allclose(o.pre_rate, n.pre_rate))
        print(f'VALIDATION vs s1f: rows s1f {len(o)}, new {len(n)}, identical={same}')
    summarize(a.source, allev if a.source != 'log' else None)


if __name__ == '__main__':
    main()
