# Stage 1p: re-analysis of the fixed-reference clusters with per-measurement reagent lots from the raw data.
#
# Lots come from s1o (raw R1_Lot/R3_Lot).  Design:
#   * join by (analyzer, sample ID, date, assay); a key carrying more than one lot is 'AMB' (no row
#     duplication: the processed curves are looked up, never joined as a product);
#   * lot event = 'effective start': first day on which a lot (>= 50 curves over >= 2 days on that analyzer)
#     reaches >= 10% of the day's curves, and again after >= 14 days of absence; the majority switch
#     (> 50% of the day for >= 3 of the next 5 days with data) is a secondary definition;
#   * s1n episode classification repeated with these events (tight window onset-7..+3 d, chance by random
#     days of the same series), side by side with the hand-entered log;
#   * event study: change in the arm-A daily flag rate from days -7..-1 to +1..+7 around each lot start,
#     against the same change around random days of the same series (instead of eta^2);
#   * lot-level flag rates, and targeted checks (Ca #1 in 2023-10; IgG lots in anchors, freezes, episode).
# Output: Results/20260924_Rev_LotFromRaw/
# py3.14.  --items <ITEM> per process, then --summarize.

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import s1m_fixed_reference_per_analyzer as M   # noqa: E402
import s1n_cluster_classification as N          # noqa: E402

OUT = os.path.join(M.BASE, 'Results', '20260924_Rev_LotFromRaw')
LOTS = os.path.join(OUT, 'lots_rows.parquet')
MIN_LOT_CURVES, MIN_LOT_DAYS, SHARE, ABSENCE = 50, 2, 0.10, 14
RNG = np.random.default_rng(20260925)


# ------------------------------------------------------------------------------------------ data
def load_with_id(item):
    """processed curves in the same order as s1m.load_lean, plus the sample ID"""
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    d = ds.dataset(M.B.PROC, partitioning='hive')
    tb = d.to_table(columns=['S_Date', 'SystemName', 'S_ID'] + M.B.RAM + M.B.RAS, filter=pc.field('A_Name') == item)
    raw = pd.to_datetime(tb.column('S_Date').to_numpy(zero_copy_only=False)).values
    order = np.argsort(raw, kind='quicksort')
    dates = pd.to_datetime(raw[order])
    anl = np.array([str(s).replace('LABOSPECT 008 ALPHA ', '') for s in
                    np.asarray(tb.column('SystemName').to_pylist(), dtype=object)[order]], dtype=object)
    sid = np.asarray([str(s).strip() for s in tb.column('S_ID').to_pylist()], dtype=object)[order]
    calc = np.empty((tb.num_rows, len(M.B.RAM)), dtype='float64')
    for j, (cm, cs) in enumerate(zip(M.B.RAM, M.B.RAS)):
        calc[:, j] = tb.column(cm).to_numpy(zero_copy_only=False)[order].astype('float64')
        calc[:, j] -= tb.column(cs).to_numpy(zero_copy_only=False)[order].astype('float64')
    return dates, calc, anl, sid


def raw_lots(item):
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    t = ds.dataset(LOTS).to_table(filter=pc.field('A_Name') == item).to_pandas()
    t['day'] = t['S_Date'].dt.normalize()
    return t


def key_map(raw):
    g = raw.groupby(['analyzer', 'S_ID', 'day'])['R1_Lot'].agg(['nunique', 'first'])
    g['lot'] = np.where(g['nunique'] == 1, g['first'], 'AMB')
    return g['lot']


def lot_events(raw, analyzer):
    """effective starts (>=10% of a day) and majority switches, per analyzer"""
    r = raw[(raw.analyzer == analyzer) & (raw.R1_Lot != '')]
    daily = r.groupby(['day', 'R1_Lot']).size().unstack(fill_value=0).sort_index()
    if daily.empty:
        return np.array([], 'datetime64[D]'), np.array([], 'datetime64[D]'), pd.DataFrame()
    share = daily.div(daily.sum(axis=1), axis=0)
    tot, ndays = daily.sum(), (daily > 0).sum()
    valid = [c for c in daily.columns if tot[c] >= MIN_LOT_CURVES and ndays[c] >= MIN_LOT_DAYS]
    eff, rows = [], []
    for c in valid:
        on = share.index[share[c] >= SHARE]
        last = None
        for dday in on:
            if last is None or (dday - last).days > ABSENCE:
                eff.append(dday); rows.append(dict(analyzer=analyzer, lot=c, day=dday.date(), kind='start' if last is None else 'reuse'))
            last = dday
    maj = share[valid].idxmax(axis=1) if valid else pd.Series(dtype=object)
    majsw = []
    vals, idx = maj.to_numpy(), maj.index
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1] and np.sum(vals[i:i + 5] == vals[i]) >= 3:
            majsw.append(idx[i])
    to = lambda x: np.sort(np.array([np.datetime64(pd.Timestamp(v).date(), 'D') for v in x], dtype='datetime64[D]'))
    return to(eff), to(majsw), pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ per item
def run_item(item, ev_log):
    dates, calc, anl, sid = load_with_id(item)
    raw = raw_lots(item)
    km = key_map(raw)
    days = dates.normalize()
    keys = pd.MultiIndex.from_arrays([anl, sid, days])
    lot = km.reindex(keys).to_numpy()
    lot = np.where(pd.isna(lot), 'NA', lot).astype(object)
    match = pd.DataFrame({'item': [item], 'n': [len(lot)], 'matched': [float(np.mean(lot != 'NA'))],
                          'ambiguous': [float(np.mean(lot == 'AMB'))],
                          'r1_eq_r3': [float((raw.R1_Lot == raw.R3_Lot).mean())]})

    # arms A, S, B (same definitions as s1m / s1n) with per-curve lot
    arms = {}
    mB = M.fit(calc[:M.N_TRAIN])
    fB = M.score(mB, calc[M.N_TRAIN:])
    starts, anchor_lots = {}, []
    for a in M.ANALYZERS:
        mk = anl[M.N_TRAIN:] == a
        arms[('B', a)] = (np.arange(M.N_TRAIN, len(anl))[mk], fB[mk])
        idx = np.where(anl == a)[0]
        if len(idx) < 2 * M.N_TRAIN:
            continue
        starts[a] = dates[idx[0]]
        arms[('A', a)] = (idx[M.N_TRAIN:], M.score(M.fit(calc[idx[:M.N_TRAIN]]), calc[idx[M.N_TRAIN:]]))
        vc = pd.Series(lot[idx[:M.N_TRAIN]]).value_counts()
        anchor_lots.append(dict(item=item, arm='A', analyzer=a, lots=';'.join(f'{k}:{v}' for k, v in vc.items())))
    if len(starts) == len(M.ANALYZERS):
        T = max(starts.values())
        for a in M.ANALYZERS:
            idx = np.where((anl == a) & np.asarray(dates >= T))[0]
            if len(idx) < 2 * M.N_TRAIN:
                continue
            arms[('S', a)] = (idx[M.N_TRAIN:], M.score(M.fit(calc[idx[:M.N_TRAIN]]), calc[idx[M.N_TRAIN:]]))
            vc = pd.Series(lot[idx[:M.N_TRAIN]]).value_counts()
            anchor_lots.append(dict(item=item, arm='S', analyzer=a, lots=';'.join(f'{k}:{v}' for k, v in vc.items())))

    ep_rows, es_rows, lot_rows, ev_rows, conc = [], [], [], [], []
    for (arm, a), (ix, f) in arms.items():
        d = np.asarray(dates[ix]).astype('datetime64[D]')
        g = N.monthly_counts(d, f)
        g = g[g.n >= N.MIN_N].reset_index(drop=True)
        if g.empty:
            continue
        daily = N.daily_counts(d, f)
        eff, majsw, evdf = lot_events(raw, a)
        loglots, logcals = N.events_for(ev_log, item, a)
        sdays = np.unique(d)
        rules = {'abs20': pd.Series(g.rate >= N.LEVEL)}
        if arm == 'A':
            rules['R1'] = N.relative_alerts(g, 'R1')
        for rule, al in rules.items():
            al = al.reset_index(drop=True)
            for src, lots_ in (('raw_eff', eff), ('raw_major', majsw), ('log', loglots)):
                cals_ = logcals if src == 'log' else np.array([], 'datetime64[D]')
                for r in N.classify(item, a, arm, rule, g, al, daily, lots_, cals_, sdays):
                    r['source'] = src
                    ep_rows.append(r)
        if arm != 'A':
            continue
        # lot-level flag rates (arm A)
        lt = pd.DataFrame({'lot': lot[ix], 'f': f}).groupby('lot').agg(n=('f', 'size'), rate=('f', 'mean')).reset_index()
        lt = lt[(lt.n >= 200) & ~lt.lot.isin(['NA', 'AMB'])]
        first = pd.DataFrame({'lot': lot[ix], 'd': d}).groupby('lot').d.min()
        for r in lt.itertuples():
            lot_rows.append(dict(item=item, analyzer=a, lot=r.lot, n=r.n, rate=r.rate, first_day=str(first.get(r.lot))))
        # event study around effective starts (arm A daily flags)
        dd = daily.copy(); dd.index = pd.to_datetime(dd.index)

        def change(t):
            t = pd.Timestamp(t)
            pre = dd[(dd.index >= t - pd.Timedelta(days=7)) & (dd.index <= t - pd.Timedelta(days=1))]
            post = dd[(dd.index >= t + pd.Timedelta(days=1)) & (dd.index <= t + pd.Timedelta(days=7))]
            if pre.n.sum() < 20 or post.n.sum() < 20:
                return np.nan
            return post.k.sum() / post.n.sum() - pre.k.sum() / pre.n.sum()
        for t in eff:
            if t < d.min() + 7:
                continue
            es_rows.append(dict(item=item, analyzer=a, kind='lot', day=str(t), delta=change(t)))
        for t in sdays[RNG.integers(0, len(sdays), 20 * max(1, len(eff)))]:
            es_rows.append(dict(item=item, analyzer=a, kind='random', day=str(t), delta=change(t)))
        # concordance raw effective starts vs hand-entered log (+-3 days)
        for t in eff:
            conc.append(dict(item=item, analyzer=a, source='raw', day=str(t), matched=N.has_event(loglots, t - 3, t + 3)))
        for t in loglots:
            conc.append(dict(item=item, analyzer=a, source='log', day=str(t), matched=N.has_event(eff, t - 3, t + 3)))
        if len(evdf):
            ev_rows.append(evdf.assign(item=item))

    pd.DataFrame(ep_rows).to_csv(os.path.join(OUT, f'episodes_{item}.csv'), index=False)
    pd.DataFrame(es_rows).to_csv(os.path.join(OUT, f'eventstudy_{item}.csv'), index=False)
    pd.DataFrame(lot_rows).to_csv(os.path.join(OUT, f'lotrates_{item}.csv'), index=False)
    pd.DataFrame(conc).to_csv(os.path.join(OUT, f'concordance_{item}.csv'), index=False)
    pd.DataFrame(anchor_lots).to_csv(os.path.join(OUT, f'anchorlots_{item}.csv'), index=False)
    match.to_csv(os.path.join(OUT, f'match_{item}.csv'), index=False)
    if ev_rows:
        pd.concat(ev_rows).to_csv(os.path.join(OUT, f'lotevents_{item}.csv'), index=False)
    # daily lot composition (raw) for targeted checks
    comp = raw.groupby(['analyzer', 'day', 'R1_Lot']).size().rename('n').reset_index()
    comp.to_csv(os.path.join(OUT, f'dailylots_{item}.csv'), index=False)
    print(f'  {item}: matched {match.matched[0]:.4f}, ambiguous {match.ambiguous[0]:.4f}, '
          f'R1==R3 {match.r1_eq_r3[0]:.4f}, episodes {len(ep_rows)}', flush=True)


# ------------------------------------------------------------------------------------------ summary
def cat(pattern):
    fs = [f for f in sorted(glob.glob(os.path.join(OUT, pattern))) if os.path.getsize(f) > 5]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def summarize():
    lines = []

    def pr(s=''):
        print(s); lines.append(str(s))

    m = cat('match_*.csv')
    pr('JOIN: ' + m.round(4).to_string(index=False))
    ep = cat('episodes_*.csv'); ep.to_csv(os.path.join(OUT, 'episodes_all.csv'), index=False)
    for (arm, rule), g in ep.groupby(['arm', 'rule']):
        pr(f'\n=== arm {arm}, rule {rule}')
        for src, h in g.groupby('source'):
            c = h.groupby('cls_tight').agg(ep=('item', 'size'), mo=('n_months', 'sum'))
            nk = h[h.cls_tight != 'known_fault']
            obs = int((nk.cls_tight == 'lot').sum())
            p = nk.chance_lot_tight.to_numpy()
            sims = (RNG.random((10_000, len(p))) < p).sum(axis=1) if len(p) else np.zeros(1)
            pv = (np.sum(sims >= obs) + 1) / 10_001
            pr(f'  [{src}] ' + ', '.join(f'{k} {r.ep} ep/{r.mo} mo' for k, r in c.iterrows()) +
               f' | lot among non-known {obs}/{len(nk)}, chance {p.sum():.1f}, p={pv:.2g}')
            un = h[h.cls_tight == 'unexplained'].groupby(['item', 'analyzer']).n_months.sum().sort_values(ascending=False)
            pr('      unexplained: ' + (', '.join(f'{i} {a} {v}' for (i, a), v in un.items()) or 'none'))
    es = cat('eventstudy_*.csv').dropna(subset=['delta'])
    es.to_csv(os.path.join(OUT, 'eventstudy_all.csv'), index=False)
    L, R = es[es.kind == 'lot'].delta.abs(), es[es.kind == 'random'].delta.abs()
    pr(f'\nEVENT STUDY (arm A, |change| in daily flag rate, -7..-1 vs +1..+7 d): lot starts n={len(L)} median {L.median():.4f}, '
       f'P(>=0.05) {np.mean(L >= .05):.3f}; random days n={len(R)} median {R.median():.4f}, P(>=0.05) {np.mean(R >= .05):.3f}; '
       f'Mann-Whitney p={mannwhitneyu(L, R, alternative="greater").pvalue:.2g}')
    per = es.assign(big=es.delta.abs() >= .05).groupby(['item', 'kind']).big.mean().unstack()
    pr('per item P(|change|>=5 pt):\n' + per.round(3).to_string())
    cc = cat('concordance_*.csv')
    if len(cc):
        pr('\nCONCORDANCE (+-3 d): raw lot starts found in the log: '
           f"{cc[cc.source == 'raw'].matched.mean():.3f} of {int((cc.source == 'raw').sum())}; "
           f"log lot entries found in the raw data: {cc[cc.source == 'log'].matched.mean():.3f} of {int((cc.source == 'log').sum())}")
        pr(cc[cc.source == 'raw'].groupby('item').matched.mean().round(2).to_string())
    with open(os.path.join(OUT, 'summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=None)
    ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    ev_log = N.load_events()
    from _common import panel as P   # noqa: E402
    for it in (a.items or P.proc_names(21)):
        if os.path.exists(os.path.join(OUT, f'episodes_{it}.csv')):
            print(f'  {it}: exists, skip', flush=True)
            continue
        run_item(it, ev_log)


if __name__ == '__main__':
    main()
