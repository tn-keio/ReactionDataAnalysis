# Do reagent lot changes / equipment maintenance invalidate a recently-trained Mahalanobis model?
# Deployment-realistic, event-aligned before/after design (PLAN Suppl08_MaintenanceLotRobustness, §11 review):
#   For each event at date t on (analyzer, item):
#     train  = the 500 samples immediately BEFORE t         (rolling "train on recent 500")
#     thr    = self-calibrated P99.5 of the training distances
#     pre    = the 500 samples before the training block     (out-of-sample pre-event baseline)
#     post   = the 500 samples immediately AFTER t           (the "vulnerable period" until 500 new samples accrue)
#   Report post_flag_rate (operational alert rate) and its localization (baseline vs reaction points).
# A PLACEBO (random non-event dates, same "next-500" extraction) gives the per-(analyzer,item) null so an
# event effect is separated from natural calendar drift.  Maha only -> py3.14, no TF.  Resumable per item.
#
# *** CODE-REVIEW STAGE: created but NOT executed. ***
import os, csv, json
import numpy as np
import pandas as pd

BASE = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
EQ = os.path.join(BASE, 'Data', '20210301-20251231_equipment-comments.csv')
RG = os.path.join(BASE, 'Data', '20210301-20251231_reagent-comments.csv')
OUT = os.path.join(BASE, 'Results', '20260718_EventRobustness')
RESULTS = os.path.join(OUT, 'event_robustness_results.csv')

PTS = list(range(5, 39)); RAM = [f'RAM{i}' for i in PTS]; RAS = [f'RAS{i}' for i in PTS]
PANEL = ['TP', 'ALB', 'TB', 'DB', 'Ca', 'IP', 'CRE', 'UA', 'CRP', 'TC', 'HDL-C', 'LDL-C', 'TG']
MODULE = {'TP': 2, 'ALB': 2, 'TB': 1, 'DB': 1, 'Ca': 2, 'IP': 2, 'CRE': 1, 'UA': 1,
          'CRP': 2, 'TC': 2, 'HDL-C': 1, 'LDL-C': 1, 'TG': 2}
REAGENT_ABBR = {'TP': 'TP', 'ALB': 'ALB', 'TB': 'TB', 'DB': 'DB', 'Ca': 'CA', 'IP': 'IP', 'CRE': 'CRE',
                'UA': 'UA', 'CRP': 'CRP', 'TC': 'TC', 'HDL-C': 'HDL-C', 'LDL-C': 'LDL-C', 'TG': 'TG'}
ANALYZERS = ['#2', '#1', '#3']            # #2 has the longest history (2021-); #1/#3 from mid-2022

N_TRAIN = 500          # rolling training block (deployment: recent 500)
N_PRE = 500            # out-of-sample pre-event baseline block (immediately before training)
M_POST = 500           # post-event evaluation block (the vulnerable period)
BUDGET = 99.5          # self-calibrated operating point
EXCL0, EXCL1 = pd.Timestamp('2021-07-25'), pd.Timestamp('2021-09-05')   # 2021-08 event, excluded
COOCCUR_DAYS = 14      # maint<->lot co-occurrence flag
PLACEBO_GAP_DAYS = 30  # placebo dates must be >30d from ANY event (for that analyzer/item)
N_PLACEBO = 20         # placebo draws per (analyzer, item)
SEED = 42


# ---------------- maha ----------------
def maha_eval(train, pre, post):
    """Fit empirical Mahalanobis on `train` (per-point min-max from train), self-cal P99.5 threshold,
    return (pre_rate, post_rate, baseline_dev_sigma, reaction_dev_sigma, onset_pt)."""
    lo, hi = train.min(0), train.max(0)
    def nz(X): return np.clip((X - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    ntr = nz(train); mu = ntr.mean(0); P = np.linalg.pinv(np.cov(ntr.T))
    dtr = np.einsum('ij,jk,ik->i', ntr - mu, P, ntr - mu)
    thr = np.percentile(dtr, BUDGET)
    def rate(X):
        nx = nz(X); d = np.einsum('ij,jk,ik->i', nx - mu, P, nx - mu)
        return float((d >= thr).mean())
    pre_rate = rate(pre); post_rate = rate(post)
    # localization: standardized per-point deviation of post vs train (raw calc units)
    rmu, rsd = train.mean(0), train.std(0) + 1e-9
    onset = int(np.argmax(np.abs(np.diff(rmu))))
    dev = np.abs(post.mean(0) - rmu) / rsd
    bmask = np.arange(len(PTS)) <= onset
    return pre_rate, post_rate, float(dev[bmask].mean()), float(dev[~bmask].mean()), PTS[onset]


# ---------------- events ----------------
def load_maint():
    eq = pd.read_csv(EQ, encoding='utf-8-sig')
    eq['d'] = pd.to_datetime(eq['管理日'])
    eq['sys'] = eq['機器名称'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    eq = eq[~((eq['d'] >= EXCL0) & (eq['d'] <= EXCL1))]
    out = {}
    for s in ANALYZERS:
        g = eq[eq['sys'] == s]
        out[s] = {'cell_lamp': sorted(g[(g['セル'] == 1) | (g['ランプ'] == 1)]['d'].unique()),
                  'mechanism': sorted(g[g['機構メンテナンス'] == 1]['d'].unique())}
    return out


def load_lot():
    rg = pd.read_csv(RG, encoding='utf-8-sig')
    rg['d'] = pd.to_datetime(rg['管理日'])
    rg['sys'] = rg['機器名称'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    rg = rg[~((rg['d'] >= EXCL0) & (rg['d'] <= EXCL1))]
    islot = rg['コメント'].astype(str).str.contains('Lot|New', case=False, regex=True)
    rg = rg[islot]
    out = {}
    for (s, abbr), g in rg.groupby(['sys', 'abbr']):
        out[(s, abbr)] = sorted(g['d'].unique())
    return out


def near(dates_list, t, days):
    return any(abs((pd.Timestamp(x) - t).days) <= days for x in dates_list)


# ---------------- per (analyzer, item) ----------------
def eval_at(dates, calc, t, etype, edate, cooccur, rows, item, analyzer):
    n = len(dates)
    i = int(np.searchsorted(dates, np.datetime64(t)))     # first index with date >= t
    if i - (N_TRAIN + N_PRE) < 0 or i + M_POST > n:
        return False
    pre = calc[i - N_TRAIN - N_PRE:i - N_TRAIN]
    train = calc[i - N_TRAIN:i]
    post = calc[i:i + M_POST]
    pre_rate, post_rate, bdev, rdev, onset = maha_eval(train, pre, post)
    post_span = int((pd.Timestamp(dates[i + M_POST - 1]) - pd.Timestamp(dates[i])).days)
    rows.append({'analyzer': analyzer, 'item': item, 'module': MODULE[item], 'event_type': etype,
                 'event_date': pd.Timestamp(edate).date().isoformat() if edate is not None else '',
                 'cooccur': int(cooccur), 'n_total': n, 'idx': i,
                 'pre_rate': round(pre_rate, 4), 'post_rate': round(post_rate, 4),
                 'delta': round(post_rate - pre_rate, 4), 'post_span_days': post_span,
                 'baseline_dev_sigma': round(bdev, 3), 'reaction_dev_sigma': round(rdev, 3), 'onset_pt': onset})
    return True


def process_item(item, maint, lot, done, writer, fh):
    if item in done:
        print(f'  {item}: already done, skip', flush=True); return
    df = pd.read_parquet(PROC, columns=['S_Date', 'SystemName'] + RAM + RAS, filters=[('A_Name', '==', item)])
    df['d'] = pd.to_datetime(df['S_Date'])
    df['sys'] = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    abbr = REAGENT_ABBR[item]
    rng = np.random.RandomState(SEED)
    rows = []
    for s in ANALYZERS:
        sub = df[df['sys'] == s].sort_values('d')
        if len(sub) < N_TRAIN + N_PRE + M_POST:
            continue
        dates = sub['d'].values.astype('datetime64[ns]')
        calc = (sub[RAM].values.astype('float64') - sub[RAS].values.astype('float64'))
        maint_all = sorted(maint[s]['cell_lamp'] + maint[s]['mechanism'])
        lot_dates = lot.get((s, abbr), [])
        # real events
        for etype, elist in [('cell_lamp', maint[s]['cell_lamp']), ('mechanism', maint[s]['mechanism']),
                             ('reagent', lot_dates)]:
            other = lot_dates if etype != 'reagent' else maint_all
            for t in elist:
                t = pd.Timestamp(t)
                eval_at(dates, calc, t, etype, t, near(other, t, COOCCUR_DAYS), rows, item, s)
        # placebo: random dates >30d from any event, with data available
        all_ev = [pd.Timestamp(x) for x in maint_all] + [pd.Timestamp(x) for x in lot_dates]
        lo_d, hi_d = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
        span = (hi_d - lo_d).days
        got = 0; tries = 0
        while got < N_PLACEBO and tries < 400:
            tries += 1
            t = lo_d + pd.Timedelta(days=int(rng.randint(0, max(1, span))))
            if (EXCL0 <= t <= EXCL1) or near(all_ev, t, PLACEBO_GAP_DAYS):
                continue
            if eval_at(dates, calc, t, 'placebo', t, 0, rows, item, s):
                got += 1
    for r in rows:
        writer.writerow(r)
    fh.flush()
    print(f'  {item}: wrote {len(rows)} rows ({sorted(set(r["analyzer"] for r in rows))})', flush=True)


FIELDS = ['analyzer', 'item', 'module', 'event_type', 'event_date', 'cooccur', 'n_total', 'idx',
          'pre_rate', 'post_rate', 'delta', 'post_span_days', 'baseline_dev_sigma', 'reaction_dev_sigma', 'onset_pt']


def main():
    os.makedirs(OUT, exist_ok=True)
    maint = load_maint(); lot = load_lot()
    with open(os.path.join(OUT, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'panel': PANEL, 'analyzers': ANALYZERS, 'N_TRAIN': N_TRAIN, 'N_PRE': N_PRE, 'M_POST': M_POST,
                   'budget': BUDGET, 'excl': [EXCL0.date().isoformat(), EXCL1.date().isoformat()],
                   'cooccur_days': COOCCUR_DAYS, 'placebo_gap_days': PLACEBO_GAP_DAYS, 'n_placebo': N_PLACEBO,
                   'seed': SEED}, f, indent=2, ensure_ascii=False)
    done = set()
    if os.path.exists(RESULTS):
        for r in csv.DictReader(open(RESULTS, encoding='utf-8')):
            done.add(r['item'])
    new = not os.path.exists(RESULTS)
    fh = open(RESULTS, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        writer.writeheader()
    for item in PANEL:
        print(f'=== {item} (module {MODULE[item]}) ===', flush=True)
        process_item(item, maint, lot, done, writer, fh)
    fh.close()
    print('DONE ->', RESULTS, flush=True)


if __name__ == '__main__':
    main()
