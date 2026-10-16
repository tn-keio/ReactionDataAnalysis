# Can other analyzers' data be used for training?  (Inter-analyzer transferability, PLAN Suppl03_InterAnalyzerTransferability)
# Contemporaneous cross-analyzer transfer test (deployment-realistic, N=500), so analyzer differences are
# separated from temporal drift:
#   Per (item, quarter-window W where all of #1/#2/#3 run, i.e. 2022-07 onward):
#     each analyzer's W-samples (date-sorted) are split train_block=[0:500] / eval_block=[500:1000].
#     within  : fit maha on B.train_block -> flag rate on B.eval_block          (per-analyzer baseline)
#     cross   : fit maha on A.train_block -> flag rate on B.eval_block (A!=B)    (transfer)
#     pool    : fit maha on 167+167+166 of the three train_blocks -> flag rate on each B.eval_block
#   transfer penalty = flag_rate - within_rate(test analyzer).  ~0 => analyzers harmonized => pooling OK.
# Maha only -> py3.14, no TF.  Resumable per item.
#
# *** CODE-REVIEW STAGE: created but NOT executed. ***
import os, csv, json
import numpy as np
import pandas as pd

BASE = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
OUT = os.path.join(BASE, 'Results', '20260718_InterAnalyzer')
RESULTS = os.path.join(OUT, 'inter_analyzer_results.csv')

PTS = list(range(5, 39)); RAM = [f'RAM{i}' for i in PTS]; RAS = [f'RAS{i}' for i in PTS]
PANEL = ['TP', 'ALB', 'TB', 'DB', 'Ca', 'IP', 'CRE', 'UA', 'CRP', 'TC', 'HDL-C', 'LDL-C', 'TG']
MODULE = {'TP': 2, 'ALB': 2, 'TB': 1, 'DB': 1, 'Ca': 2, 'IP': 2, 'CRE': 1, 'UA': 1,
          'CRP': 2, 'TC': 2, 'HDL-C': 1, 'LDL-C': 1, 'TG': 2}
ANALYZERS = ['#1', '#2', '#3']
N_TRAIN = 500
N_EVAL = 500
BUDGET = 99.5
POOL_SHARES = {'#1': 167, '#2': 167, '#3': 166}     # sum 500
# quarter windows where all three analyzers run (#1 from 2022-05-24, #3 from 2022-06-02)
QUARTERS = pd.date_range('2022-07-01', '2025-10-01', freq='QS')   # quarter starts


def maha_fit(train):
    lo, hi = train.min(0), train.max(0)
    def nz(X): return (X - lo) / (hi - lo + 1e-8)      # affine, no clipping
    ntr = nz(train); mu = ntr.mean(0); P = np.linalg.pinv(np.cov(ntr.T))
    dtr = np.einsum('ij,jk,ik->i', ntr - mu, P, ntr - mu)
    thr = np.percentile(dtr, BUDGET)
    rmu, rsd = train.mean(0), train.std(0) + 1e-9
    onset = int(np.argmax(np.abs(np.diff(rmu))))
    return dict(nz=nz, mu=mu, P=P, thr=thr, rmu=rmu, rsd=rsd, onset=onset)


def flag_and_loc(m, ev):
    nx = m['nz'](ev); d = np.einsum('ij,jk,ik->i', nx - m['mu'], m['P'], nx - m['mu'])
    rate = float((d >= m['thr']).mean())
    dev = np.abs(ev.mean(0) - m['rmu']) / m['rsd']
    bmask = np.arange(len(PTS)) <= m['onset']
    return rate, float(dev[bmask].mean()), float(dev[~bmask].mean())


def process_item(item, done, writer, fh):
    if item in done:
        print(f'  {item}: already done, skip', flush=True); return
    df = pd.read_parquet(PROC, columns=['S_Date', 'SystemName'] + RAM + RAS, filters=[('A_Name', '==', item)])
    df['d'] = pd.to_datetime(df['S_Date'])
    df['sys'] = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    calc_all = df[RAM].values.astype('float64') - df[RAS].values.astype('float64')
    rows = []
    for q in QUARTERS:
        q0, q1 = q, q + pd.offsets.QuarterBegin(startingMonth=1)
        blocks = {}   # analyzer -> (train, eval)
        for a in ANALYZERS:
            m = ((df['sys'] == a) & (df['d'] >= q0) & (df['d'] < q1)).values
            idx = np.where(m)[0]
            if len(idx) < N_TRAIN + N_EVAL:
                continue
            order = idx[np.argsort(df['d'].values[idx])]
            c = calc_all[order]
            blocks[a] = (c[:N_TRAIN], c[N_TRAIN:N_TRAIN + N_EVAL])
        if len(blocks) < len(ANALYZERS):
            continue                       # need all 3 analyzers present in this quarter
        # within (fit each on its own train, eval its own eval) + models cache
        models = {a: maha_fit(blocks[a][0]) for a in ANALYZERS}
        within = {}
        for b in ANALYZERS:
            r, bd, rd = flag_and_loc(models[b], blocks[b][1]); within[b] = r
        qlabel = f'{q.year}Q{(q.month - 1) // 3 + 1}'
        # single-source transfer matrix
        for a in ANALYZERS:
            for b in ANALYZERS:
                r, bd, rd = flag_and_loc(models[a], blocks[b][1])
                rows.append({'item': item, 'module': MODULE[item], 'window': qlabel,
                             'train_src': a, 'test_analyzer': b,
                             'kind': 'within' if a == b else 'cross', 'flag_rate': round(r, 4),
                             'within_rate': round(within[b], 4),
                             'penalty': round(r - within[b], 4), 'n_train': N_TRAIN, 'n_test': N_EVAL,
                             'baseline_dev': round(bd, 3), 'reaction_dev': round(rd, 3)})
        # pooled model
        pool_train = np.concatenate([blocks[a][0][:POOL_SHARES[a]] for a in ANALYZERS])
        mp = maha_fit(pool_train)
        for b in ANALYZERS:
            r, bd, rd = flag_and_loc(mp, blocks[b][1])
            rows.append({'item': item, 'module': MODULE[item], 'window': qlabel,
                         'train_src': 'pool', 'test_analyzer': b, 'kind': 'pool', 'flag_rate': round(r, 4),
                         'within_rate': round(within[b], 4), 'penalty': round(r - within[b], 4),
                         'n_train': N_TRAIN, 'n_test': N_EVAL, 'baseline_dev': round(bd, 3),
                         'reaction_dev': round(rd, 3)})
    for r in rows:
        writer.writerow(r)
    fh.flush()
    print(f'  {item}: wrote {len(rows)} rows ({len(set(r["window"] for r in rows))} quarters)', flush=True)


FIELDS = ['item', 'module', 'window', 'train_src', 'test_analyzer', 'kind', 'flag_rate', 'within_rate',
          'penalty', 'n_train', 'n_test', 'baseline_dev', 'reaction_dev']


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'panel': PANEL, 'analyzers': ANALYZERS, 'N_TRAIN': N_TRAIN, 'N_EVAL': N_EVAL,
                   'budget': BUDGET, 'pool_shares': POOL_SHARES,
                   'quarters': [f'{q.year}Q{(q.month - 1) // 3 + 1}' for q in QUARTERS]}, f, indent=2)
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
        process_item(item, done, writer, fh)
    fh.close()
    print('DONE ->', RESULTS, flush=True)


if __name__ == '__main__':
    main()
