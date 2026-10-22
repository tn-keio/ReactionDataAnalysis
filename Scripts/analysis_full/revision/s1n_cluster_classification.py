# Stage 1n: follow-up to s1m (fixed-reference monitoring, per analyzer).
#
#  (a) Re-classify fixed-reference cluster episodes (monthly flag rate >= 20%) as known fault /
#      reagent-lot change / calibration / unexplained, against the laboratory's reagent-event log.
#      Onset DAY is located on the daily flag-rate series (a monthly window of ~60 days
#      makes 'lot-associated' nearly automatic), and the association window is tight: onset-7 .. onset+3 d.
#      The ~60-day monthly window is kept as a sensitivity analysis.  Chance control: the same classifier
#      applied to random days of the same series (base rate; permutation test, 10,000 draws).
#  (b) Per-analyzer freeze analysis for IgG (and IgM as positive control): fit on the analyzer's last 500
#      curves before the freeze date, score its later curves.  The pooled version (s1k --freeze-before) is
#      recomputed as a reproduction check.
#  (c) Relative instead of absolute counting (flag rates are bounded counts):
#      R1 step     : month vs the previous 6 months pooled (>= 3 months available)
#      R2 anchored : month vs the first 6 months after the anchor
#      alert = one-sided Fisher exact p < 0.001 AND rate >= baseline + 0.02 AND rate >= 2 x baseline.
#
# Arms reuse s1m: A (own-analyzer earliest 500), B (pooled s1k model scored per analyzer), S (anchor after
# the date all three analyzers started).  N_TRAIN = 500, empirical covariance, P99.5, no clipping.
# Output: Results/20260924_Rev_FixedReference_Classify/
# py3.14.  One process per item: --items <ITEM>; then --summarize.  (b) runs with --freeze.

import os
import re
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import s1m_fixed_reference_per_analyzer as M   # noqa: E402  (load_lean, fit, score, ANALYZERS, N_TRAIN)
import s1k_fixed_reference_panel as K           # noqa: E402  (KNOWN)

BASE = M.BASE
OUT = os.path.join(BASE, 'Results', '20260924_Rev_FixedReference_Classify')
LOG = os.path.join(BASE, 'Data', '20210301-20251231_reagent-comments.csv')
MIN_N = 30
LEVEL = 0.20
TIGHT = (-7, 3)          # days relative to the onset day
N_PERM = 10_000
RNG = np.random.default_rng(20260924)

CAL_LOT = re.compile(r'cal\s*[:：]?\s*lot', re.I)                       # calibrator lot, e.g. "CAL:LOT R308"
LOT = re.compile(r'lot|(?<![A-Za-z0-9])[A-Z]\d{3}[A-Z]?(?![0-9])', re.I)
CAL = re.compile(r'cal', re.I)


# ----------------------------------------------------------------------------------------- event log
def load_events():
    d = pd.read_csv(LOG, encoding='utf-8-sig')
    d.columns = ['date', 'sys', 'case', 'cd', 'abbr', 'comment']
    d['date'] = pd.to_datetime(d['date'].astype(str).str.strip(), format='mixed', errors='coerce')
    d['analyzer'] = d['sys'].astype(str).str.replace('LABOSPECT 008 ALPHA ', '', regex=False).str.strip()
    d['abbr'] = d['abbr'].astype(str).str.strip().str.upper()
    c = d['comment'].astype(str)
    cal_lot = c.str.contains(CAL_LOT)
    # calibration-only: calibrator-lot entries, or 'cal' without any reagent-lot code; everything else in
    # this log is a reagent-lot entry (bare codes such as '154RLR' or '20L07' included)
    is_cal = cal_lot | (c.str.contains(CAL) & ~c.str.contains(LOT))
    d['kind'] = np.where(is_cal, 'cal', 'lot')
    return d.dropna(subset=['date'])


def events_for(ev, item, analyzer):
    e = ev[(ev.abbr == item.upper()) & (ev.analyzer == analyzer)]
    return (np.sort(e[e.kind == 'lot'].date.values.astype('datetime64[D]')),
            np.sort(e[e.kind == 'cal'].date.values.astype('datetime64[D]')))


def has_event(days, lo, hi):
    """any event with lo <= day <= hi (datetime64[D])"""
    i = np.searchsorted(days, lo, side='left')
    return i < len(days) and days[i] <= hi


# ----------------------------------------------------------------------------------------- per-curve flags
def per_curve_flags(item):
    """{(arm, analyzer): (dates datetime64[D], flags bool)} for arms A, B, S."""
    dates, calc, anl = M.load_lean(item)
    out = {}
    mB = M.fit(calc[:M.N_TRAIN])
    fB = M.score(mB, calc[M.N_TRAIN:])
    dB, aB = dates[M.N_TRAIN:], anl[M.N_TRAIN:]
    starts = {}
    for a in M.ANALYZERS:
        m = aB == a
        out[('B', a)] = (np.asarray(dB[m]).astype('datetime64[D]'), fB[m])
        idx = np.where(anl == a)[0]
        if len(idx) < 2 * M.N_TRAIN:
            continue
        starts[a] = dates[idx[0]]
        rest = idx[M.N_TRAIN:]
        out[('A', a)] = (np.asarray(dates[rest]).astype('datetime64[D]'), M.score(M.fit(calc[idx[:M.N_TRAIN]]), calc[rest]))
    if len(starts) == len(M.ANALYZERS):
        T = max(starts.values())
        for a in M.ANALYZERS:
            idx = np.where((anl == a) & np.asarray(dates >= T))[0]
            if len(idx) < 2 * M.N_TRAIN:
                continue
            rest = idx[M.N_TRAIN:]
            out[('S', a)] = (np.asarray(dates[rest]).astype('datetime64[D]'), M.score(M.fit(calc[idx[:M.N_TRAIN]]), calc[rest]))
    return out


def monthly_counts(d, f):
    df = pd.DataFrame({'month': d.astype('datetime64[M]'), 'f': f})
    g = df.groupby('month').agg(n=('f', 'size'), k=('f', 'sum')).reset_index()
    g['rate'] = g.k / g.n
    return g


def daily_counts(d, f):
    df = pd.DataFrame({'day': d, 'f': f})
    return df.groupby('day').agg(n=('f', 'size'), k=('f', 'sum'))


def onset_day(daily, month, base_rate, month_rate):
    """First day, from 31 d before the month start to the month end, on which the trailing 7-day flag rate
    reaches the midpoint between the pre-episode baseline and the first episode month's rate."""
    lvl = (base_rate + month_rate) / 2
    m0 = np.datetime64(month, 'D')
    m1 = (np.datetime64(month, 'M') + 1).astype('datetime64[D]') - 1
    lo = m0 - 31
    s = daily.loc[(daily.index >= lo) & (daily.index <= m1)]
    if s.empty:
        return m0
    full = pd.DataFrame(index=pd.date_range(pd.Timestamp(s.index.min()), pd.Timestamp(s.index.max()), freq='D'))
    full = full.join(s.set_index(pd.to_datetime(s.index))).fillna(0)
    k7 = full.k.rolling(7, min_periods=1).sum()
    n7 = full.n.rolling(7, min_periods=1).sum()
    ok = (n7 >= 20) & (k7 / n7.replace(0, np.nan) >= lvl)
    return np.datetime64(ok.idxmax().date(), 'D') if ok.any() else m0


def runs(flags_bool, gap=1):
    """runs of True (allowing `gap` False months inside), as lists of positions"""
    pos = np.where(flags_bool)[0]
    if len(pos) == 0:
        return []
    out, cur = [], [pos[0]]
    for p in pos[1:]:
        if p - cur[-1] <= gap + 1:
            cur.append(p)
        else:
            out.append(cur); cur = [p]
    out.append(cur)
    return out


def known_month(item, month_str):
    return month_str in K.KNOWN.get(item, []) or month_str in K.KNOWN['ALL']


def classify(item, analyzer, arm, rule, g, alert, daily, lots, cals, series_days):
    """one row per alert episode (run of alert months)"""
    rows = []
    for r in runs(alert.to_numpy()):
        months = g.month.iloc[r]
        mstr = [str(m)[:7] for m in months]
        first = g.iloc[r[0]]
        pre = g.iloc[max(0, r[0] - 6):r[0]]
        base = float(pre.k.sum() / pre.n.sum()) if len(pre) and pre.n.sum() else 0.0
        od = onset_day(daily, first.month, base, float(first.rate))
        ms = np.datetime64(first.month, 'D')
        me = (np.datetime64(first.month, 'M') + 1).astype('datetime64[D]') - 1
        tight = (od + TIGHT[0], od + TIGHT[1])
        wide = (ms - 31, me)
        is_known = any(known_month(item, m) for m in mstr)
        lot_t, cal_t = has_event(lots, *tight), has_event(cals, *tight)
        lot_w, cal_w = has_event(lots, *wide), has_event(cals, *wide)
        cls_t = 'known_fault' if is_known else 'lot' if lot_t else 'calibration' if cal_t else 'unexplained'
        cls_w = 'known_fault' if is_known else 'lot' if lot_w else 'calibration' if cal_w else 'unexplained'
        # chance: probability that a random scored day of this series has a lot event in the tight window
        rnd = series_days[RNG.integers(0, len(series_days), 2000)]
        p_lot = float(np.mean([has_event(lots, x + TIGHT[0], x + TIGHT[1]) for x in rnd]))
        rows.append(dict(item=item, analyzer=analyzer, arm=arm, rule=rule, onset_month=mstr[0],
                         last_month=mstr[-1], n_months=len(r), onset_day=str(od), base_rate=round(base, 4),
                         first_rate=round(float(first.rate), 4), max_rate=round(float(g.rate.iloc[r].max()), 4),
                         cls_tight=cls_t, cls_wide=cls_w, chance_lot_tight=round(p_lot, 4)))
    return rows


def relative_alerts(g, mode):
    """R1 step (previous 6 months) / R2 anchored (first 6 months)"""
    al = np.zeros(len(g), bool)
    for i in range(len(g)):
        if mode == 'R1':
            prev = g.iloc[max(0, i - 6):i]
            if len(prev) < 3:
                continue
        else:
            if i < 6:
                continue
            prev = g.iloc[:6]
        bk, bn = int(prev.k.sum()), int(prev.n.sum())
        k, n = int(g.k.iloc[i]), int(g.n.iloc[i])
        b = bk / bn if bn else 0.0
        r = k / n
        if r < b + 0.02 or r < 2 * b:
            continue
        p = fisher_exact([[k, n - k], [bk, bn - bk]], alternative='greater')[1]
        al[i] = p < 1e-3
    return pd.Series(al)


def run_item(item, ev):
    flags = per_curve_flags(item)
    rows, mon = [], []
    for (arm, a), (d, f) in flags.items():
        g = monthly_counts(d, f)
        g = g[g.n >= MIN_N].reset_index(drop=True)
        if g.empty:
            continue
        daily = daily_counts(d, f)
        lots, cals = events_for(ev, item, a)
        sdays = np.unique(d)
        gg = g.assign(item=item, arm=arm, analyzer=a, month=g.month.astype(str).str[:7])
        rules = {'abs20': pd.Series(g.rate >= LEVEL)}
        if arm in ('A', 'S'):
            rules['R1'] = relative_alerts(g, 'R1')
            rules['R2'] = relative_alerts(g, 'R2')
        for rule, al in rules.items():
            gg[rule] = al.values
            rows += classify(item, a, arm, rule, g, al.reset_index(drop=True), daily, lots, cals, sdays)
        mon.append(gg)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f'episodes_{item}.csv'), index=False)
    pd.concat(mon).to_csv(os.path.join(OUT, f'monthly_{item}.csv'), index=False)
    print(f'  {item}: {len(rows)} episodes', flush=True)


# ----------------------------------------------------------------------------------------- (b) freeze
FREEZE = {'IgG': (['2024-01-01', '2024-06-01', '2024-10-01', '2024-12-05'], ('2024-12', '2025-03')),
          'IgM': (['2023-05-11'], ('2023-06', '2024-04'))}


def freeze_summary(item, dates, flags, fdate, ep, label):
    m = monthly_counts(np.asarray(dates).astype('datetime64[D]'), flags)
    m = m[m.n >= MIN_N]
    ms = m.month.astype(str).str[:7]
    e0, e1 = ep
    pre = m[(ms < e0)]
    epi = m[(ms >= e0) & (ms <= e1)]
    pre_rate = float(pre.k.sum() / pre.n.sum()) if len(pre) else np.nan
    return dict(item=item, analyzer=label, freeze=fdate, pre_months=len(pre), pre_rate=pre_rate,
                ep_months=len(epi), ep_median=float(epi.rate.median()) if len(epi) else np.nan,
                ep_max=float(epi.rate.max()) if len(epi) else np.nan,
                ratio=float(epi.rate.median() / pre_rate) if len(epi) and pre_rate else np.nan,
                n_ge20=int((epi.rate >= .2).sum()), n_ge3x=int((epi.rate >= 3 * pre_rate).sum()) if pre_rate == pre_rate else 0)


def run_freeze():
    rows = []
    for item, (fdates, ep) in FREEZE.items():
        dates, calc, anl = M.load_lean(item)
        for fd in fdates:
            t = pd.Timestamp(fd)
            before = np.asarray(dates < t)
            # pooled (reproduces s1k --freeze-before)
            idx = np.where(before)[0]
            after = np.where(~before)[0]
            f = M.score(M.fit(calc[idx[-M.N_TRAIN:]]), calc[after])
            rows.append(freeze_summary(item, dates[after], f, fd, ep, 'pooled'))
            for a in M.ANALYZERS:
                ia = np.where(before & (anl == a))[0]
                ja = np.where(~before & (anl == a))[0]
                if len(ia) < M.N_TRAIN or len(ja) == 0:
                    continue
                f = M.score(M.fit(calc[ia[-M.N_TRAIN:]]), calc[ja])
                r = freeze_summary(item, dates[ja], f, fd, ep, a)
                r['train_start'] = str(dates[ia[-M.N_TRAIN]].date())
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'freeze_per_analyzer.csv'), index=False)
    with pd.option_context('display.width', 220):
        print(df.round(4).to_string(index=False))
    # reproduction check against s1k pooled freeze files
    for item, (fdates, ep) in FREEZE.items():
        for fd in fdates:
            p = os.path.join(M.S1K, f"monthly_{item}_frozen{fd.replace('-', '')}.csv")
            if os.path.exists(p):
                r = pd.read_csv(p)
                ms = r.month.astype(str)
                old = r[(ms >= ep[0]) & (ms <= ep[1])].rate.max()
                new = df[(df.item == item) & (df.freeze == fd) & (df.analyzer == 'pooled')].ep_max.iloc[0]
                print(f'  check {item} {fd}: s1k ep_max {old:.4f} vs pooled {new:.4f} -> {np.isclose(old, new)}')


# ----------------------------------------------------------------------------------------- summary
def summarize():
    ep = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, 'episodes_*.csv')))
                    if os.path.getsize(f) > 5], ignore_index=True)
    ep.to_csv(os.path.join(OUT, 'episodes_all.csv'), index=False)
    lines = []

    def pr(s=''):
        print(s); lines.append(s)

    for (arm, rule), g in ep.groupby(['arm', 'rule']):
        pr(f'\n=== arm {arm}, rule {rule}: {len(g)} episodes, {int(g.n_months.sum())} analyzer-months')
        for col in ('cls_tight', 'cls_wide'):
            c = g.groupby(col).agg(episodes=('item', 'size'), months=('n_months', 'sum'))
            pr(f'  {col}: ' + ', '.join(f'{k} {r.episodes} ep / {r.months} mo' for k, r in c.iterrows()))
        # chance control over non-known episodes (tight window)
        nk = g[g.cls_tight != 'known_fault']
        if len(nk):
            obs = int((nk.cls_tight == 'lot').sum())
            p = nk.chance_lot_tight.to_numpy()
            sims = (RNG.random((N_PERM, len(p))) < p).sum(axis=1)
            pr(f'  lot-associated (tight, non-known): observed {obs}/{len(nk)}; expected by chance '
               f'{p.sum():.1f} (mean base rate {p.mean():.3f}); permutation p = {(np.sum(sims >= obs) + 1) / (N_PERM + 1):.4g}')
        un = g[g.cls_tight == 'unexplained'].groupby('item').n_months.sum().sort_values(ascending=False)
        pr('  unexplained months by item (tight): ' + (', '.join(f'{k} {v}' for k, v in un.items()) or 'none'))

    pr('\n=== reagent episodes under relative rules (arm A and S): alert months inside the documented periods')
    mon = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, 'monthly_*.csv')))], ignore_index=True)
    mon.to_csv(os.path.join(OUT, 'monthly_all_rules.csv'), index=False)
    for item in ('IgM', 'IgG'):
        ks = K.KNOWN[item]
        for arm in ('A', 'S'):
            x = mon[(mon.item == item) & (mon.arm == arm) & mon.month.isin(ks)]
            s = x.groupby('analyzer')[['abs20', 'R1', 'R2']].sum().astype(int)
            pr(f'  {item} arm {arm} ({len(ks)} episode months): ' +
               '; '.join(f'{a}: abs20 {r.abs20}, R1 {r.R1}, R2 {r.R2}' for a, r in s.iterrows()))
    with open(os.path.join(OUT, 'summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=None)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--events', action='store_true', help='print the parsed event log summary')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.freeze:
        return run_freeze()
    if a.summarize:
        return summarize()
    ev = load_events()
    if a.events:
        print(ev.groupby('kind').size())
        print(ev[ev.abbr.isin(['CA', 'IGG', 'CRE', 'LDL-C'])].groupby(['abbr', 'kind']).size())
        print(ev[ev.kind == 'other'].comment.head(15).tolist())
        return
    from _common import panel as P   # noqa: E402
    for it in (a.items or P.proc_names(21)):
        if os.path.exists(os.path.join(OUT, f'episodes_{it}.csv')):
            print(f'  {it}: exists, skip', flush=True)
            continue
        run_item(it, ev)


if __name__ == '__main__':
    main()
