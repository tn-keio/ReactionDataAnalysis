# Stage 1z10:
# how often were CLEAN and real curves clipped in the deployment-style analyses, and is the Mahalanobis
# distance there really invariant to the per-point (diagonal) min-max map, given that the code inverts the
# covariance with np.linalg.pinv (exact invariance needs a full-rank covariance)?
#
# s1a counted clipping only for the main benchmark (scalar constants from the 10,000-curve pool).  The
# deployment-style analyses take the constants from the training WINDOW itself, so later real curves can
# leave [0, 1] much more often:
#   s11  S11.2 / §4.4 workload (Table S3): rolling blocks, scalar min-max from the preceding N curves
#        (measure_review_burden_unified.rolling_flags, N = 500 and 5,000), EmpiricalCovariance.
#   s9   S9 maintenance / lot robustness, final exact-time run (s1s): 500 train / 500 pre / 500 post,
#        per-point min-max from the 500 train curves.
#   s4   S4 inter-analyzer transfer (s1f path): quarter blocks, 500 train / 500 eval, per-point.
#   cu   S10 real Cu case (s1e path): 500 train, 416 contemporaneous test curves + the case, per-point.
#   s34  S3.4 rolling estimator selection: monthly anchors, 500 train / M_SCORE real curves, per-point
#        (real curves only; s1u already counted the synthetic battery in aggregate).
# For every fit the wrapper also recomputes the empirical-Mahalanobis decisions on the RAW calc values
# (no normalization) and counts decisions that differ from the normalized path, with the covariance rank
# and condition number.  Identical decisions = the invariance holds in practice for that analysis.
#
# Existing scripts are imported unchanged; the s9 / s4 / cu runs re-execute the
# unclipped code paths with outputs redirected here, and their result tables are compared with the stored
# revision outputs (repro check).  Nothing outside Results/20261016_Rev_M1_DeployClipFreq is written.
# Output: Results/20261016_Rev_M1_DeployClipFreq/<analysis>/clipfreq_<analysis>.csv (+ run outputs)
# py3.14 (no TensorFlow).  Usage: py -3.14 s1z10_deploy_clipfreq.py --which s11 s9 s4 cu s34

import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, '..'))):
    if p not in sys.path:
        sys.path.insert(0, p)
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(BASE, 'Results')
OUT = os.path.join(RES, '20261016_Rev_M1_DeployClipFreq')
BUDGET = 99.5

CTX = {}
ACC = {}          # (analysis, item, group, role) -> accumulators


def _acc(key):
    return ACC.setdefault(key, dict(n_fits=0, n_curves=0, n_curves_out=0, n_points=0, n_points_out=0,
                                    n_below0=0, n_above1=0, n_flag_diff=0, n_flag=0, max_reldiff=0.0,
                                    min_rank=10 ** 9, max_cond=0.0))


def check_fit(analysis, item, group, train, evals, lo, hi, bias=False):
    """train / evals: RAW calc arrays.  lo, hi: the constants the analysis uses (vector = per point,
    float = scalar).  Counts out-of-range curves per role and compares normalized vs raw decisions."""
    s = hi - lo + 1e-8
    nz = lambda X: (X - lo) / s
    ntr = nz(train)
    cov_n = np.cov(ntr.T, bias=bias); cov_r = np.cov(train.T, bias=bias)
    Pn, Pr = np.linalg.pinv(cov_n), np.linalg.pinv(cov_r)
    mu_n, mu_r = ntr.mean(0), train.mean(0)
    d = lambda X, mu, P: np.einsum('ij,jk,ik->i', X - mu, P, X - mu)
    thr_n = np.percentile(d(ntr, mu_n, Pn), BUDGET); thr_r = np.percentile(d(train, mu_r, Pr), BUDGET)
    rank = int(np.linalg.matrix_rank(cov_n)); cond = float(np.linalg.cond(cov_n))
    for role, X in [('train', train)] + list(evals.items()):
        if X is None or len(X) == 0:
            continue
        nx = nz(X)
        lo_m, hi_m = nx < 0.0, nx > 1.0
        a = _acc((analysis, item, group, role))
        a['n_curves'] += len(X); a['n_curves_out'] += int((lo_m | hi_m).any(1).sum())
        a['n_points'] += nx.size; a['n_points_out'] += int((lo_m | hi_m).sum())
        a['n_below0'] += int(lo_m.any(1).sum()); a['n_above1'] += int(hi_m.any(1).sum())
        dn, dr = d(nx, mu_n, Pn), d(X, mu_r, Pr)
        fn, fr = dn >= thr_n, dr >= thr_r
        a['n_flag'] += int(fn.sum()); a['n_flag_diff'] += int((fn != fr).sum())
        a['max_reldiff'] = max(a['max_reldiff'], float(np.max(np.abs(dn - dr) / np.maximum(np.abs(dr), 1e-12))))
        a['n_fits'] += 1; a['min_rank'] = min(a['min_rank'], rank); a['max_cond'] = max(a['max_cond'], cond)


def write(analysis):
    rows = [dict(analysis=k[0], item=k[1], group=k[2], role=k[3], **v) for k, v in ACC.items() if k[0] == analysis]
    df = pd.DataFrame(rows)
    df['frac_curves_out'] = df.n_curves_out / df.n_curves
    df['frac_points_out'] = df.n_points_out / df.n_points
    d = os.path.join(OUT, analysis); os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, f'clipfreq_{analysis}.csv'), index=False)
    tot = df.groupby(['group', 'role'])[['n_fits', 'n_curves', 'n_curves_out', 'n_points', 'n_points_out',
                                          'n_flag', 'n_flag_diff']].sum()
    tot['frac_curves_out'] = tot.n_curves_out / tot.n_curves
    tot['frac_points_out'] = tot.n_points_out / tot.n_points
    tot['min_rank'] = df.groupby(['group', 'role']).min_rank.min()
    tot['max_cond'] = df.groupby(['group', 'role']).max_cond.max()
    tot['max_reldiff'] = df.groupby(['group', 'role']).max_reldiff.max()
    tot.to_csv(os.path.join(d, f'clipfreq_{analysis}_total.csv'))
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(f'\n== {analysis} ==\n', tot.to_string(), flush=True)


def compare_csv(new, old, keys, cols, label):
    a, b = pd.read_csv(new), pd.read_csv(old)
    m = a.merge(b, on=keys, suffixes=('_new', '_old'), how='outer', indicator=True)
    nonmatch = int((m._merge != 'both').sum())
    diff = max(float(np.nanmax(np.abs(m[f'{c}_new'] - m[f'{c}_old']))) for c in cols) if len(m) else np.nan
    msg = f'repro {label}: rows new {len(a)} old {len(b)} unmatched {nonmatch} max|diff| {diff:.2e}'
    print(msg, flush=True)
    return msg


# ------------------------------------------------------------------ S11 workload (scalar, rolling blocks)
def run_s11(items=None):
    FORPAPER = os.path.join(BASE, 'ForPaper')
    for p in (os.path.join(FORPAPER, 'Suppl10_ReviewBurden'), FORPAPER):
        if p not in sys.path:
            sys.path.insert(0, p)
    import measure_review_burden_unified as B
    from _common import panel as P                 # Table S3 uses the 21-item panel (as s1w); B.PANEL has 13
    from sklearn.covariance import EmpiricalCovariance
    rates = []
    for item, _ in P.PANEL21:
        if items and item not in items:
            continue
        dates, calc = B.load(item)
        for N in (500, 5000):
            if len(calc) < 3 * N:
                continue
            flags = []
            for start in range(N, len(calc), N):
                stop = min(start + N, len(calc))
                train = calc[start - N:start]
                lo, hi = B.norm_fit(train)
                check_fit('s11', item, f'N{N}', train, {'scored': calc[start:stop]}, lo, hi, bias=True)
                W = (train - lo) / (hi - lo + 1e-8)          # unclipped = the revision path
                e = EmpiricalCovariance().fit(W)
                thr = np.percentile(B.maha(e.location_, e.covariance_, W), B.BUDGET)
                flags.append(B.maha(e.location_, e.covariance_, (calc[start:stop] - lo) / (hi - lo + 1e-8)) >= thr)
            rates.append(dict(item=item, N=N, rate_pct=100 * float(np.concatenate(flags).mean())))
        print(f'  s11 {item} done', flush=True)
        del calc
    write('s11')
    pd.DataFrame(rates).to_csv(os.path.join(OUT, 's11', 'rates_check.csv'), index=False)


# ------------------------------------------------------------------ S9 (final exact-time run, s1s)
def run_s9():
    import event_robustness_analysis as EV
    import s1s_s9_exact_time as T              # imports s1q, which sets EV.maha_eval to the unclipped copy
    inner = EV.maha_eval
    orig_eval_at = EV.eval_at

    def eval_at(dates, calc, t, etype, edate, cooccur, rows, item, analyzer):
        CTX.update(item=item, group=f'{analyzer}|{etype}')
        return orig_eval_at(dates, calc, t, etype, edate, cooccur, rows, item, analyzer)

    def maha_eval(train, pre, post):
        check_fit('s9', CTX['item'], CTX['group'], train, {'pre': pre, 'post': post}, train.min(0), train.max(0))
        return inner(train, pre, post)

    EV.eval_at = eval_at; EV.maha_eval = maha_eval
    stored = os.path.join(T.OUT, 'event_robustness_results.csv')
    T.OUT = os.path.join(OUT, 's9', 'run'); os.makedirs(T.OUT, exist_ok=True)
    T.main()
    write('s9')
    keys = ['analyzer', 'item', 'event_type', 'event_date', 'idx']
    compare_csv(os.path.join(T.OUT, 'event_robustness_results.csv'), stored, keys, ['pre_rate', 'post_rate'], 's9')


# ------------------------------------------------------------------ S4 inter-analyzer (s1f path)
def run_s4():
    import inter_analyzer_analysis as IA
    import s1f_event_interanalyzer_noclip as F
    stored = os.path.join(RES, '20260917_Rev_M1_S9S4_noclip', 'inter', os.path.basename(IA.RESULTS))
    st = {'fits': 0, 'calls': 0, 'own': {}}
    labels = list(IA.ANALYZERS) + ['pool']
    orig_process = IA.process_item
    orig_flag = IA.flag_and_loc

    def process_item(item, done, writer, fh):
        CTX['item'] = item
        return orig_process(item, done, writer, fh)

    def maha_fit(train):
        if st['fits'] % 4 == 0:
            st['calls'] = 0; st['own'] = {}
        m = F.ia_maha_fit_noclip(train)
        m['_train'], m['_lo'], m['_hi'] = train, train.min(0), train.max(0)
        m['_src'] = labels[st['fits'] % 4]; st['fits'] += 1
        return m

    def flag_and_loc(m, ev):
        k = st['calls']; st['calls'] += 1
        if k < len(IA.ANALYZERS):                     # the within-analyzer pass that fills `within`
            st['own'][id(ev)] = m['_src']
        else:
            kind = 'pool' if m['_src'] == 'pool' else ('within' if st['own'].get(id(ev)) == m['_src'] else 'cross')
            check_fit('s4', CTX['item'], kind, m['_train'], {'eval': ev}, m['_lo'], m['_hi'])
        return orig_flag(m, ev)

    IA.process_item = process_item; IA.maha_fit = maha_fit; IA.flag_and_loc = flag_and_loc
    IA.OUT = os.path.join(OUT, 's4', 'run'); os.makedirs(IA.OUT, exist_ok=True)
    IA.RESULTS = os.path.join(IA.OUT, os.path.basename(IA.RESULTS))
    assert not os.path.exists(IA.RESULTS), 'S4 run output exists; remove it to recount'
    IA.main()
    write('s4')
    compare_csv(IA.RESULTS, stored, ['item', 'window', 'train_src', 'test_analyzer'], ['flag_rate', 'penalty'], 's4')


# ------------------------------------------------------------------ Cu real case (s1e path)
def run_cu():
    import real_anomaly_cu_validation as C
    import s1e_cu_case_noclip as E
    st = {'n': 0}
    orig_dist = C.dist

    def maha_fit(train):
        m = E.maha_fit_noclip(train)
        m['_train'], m['_lo'], m['_hi'] = train, train.min(0), train.max(0)
        m['_tag'] = f'fit{st["n"]}'; st['n'] += 1
        return m

    def dist(m, X):
        role = 'case' if len(X) == 1 else 'test'
        check_fit('cu', 'Cu', f'{m["_tag"]}|p{X.shape[1]}', m['_train'], {role: X}, m['_lo'], m['_hi'])
        return orig_dist(m, X)

    C.maha_fit = maha_fit; C.dist = dist
    C.OUT = os.path.join(OUT, 'cu', 'run'); C.FIGDIR = os.path.join(C.OUT, 'Figures')
    os.makedirs(C.FIGDIR, exist_ok=True)
    C.main()
    write('cu')


# ------------------------------------------------------------------ S3.4 rolling estimator selection (real curves)
def run_s34(items=None):
    import run_estimator_selection as ES
    import pyarrow.dataset as pds
    import pyarrow.compute as pc
    for item in ES.forpaper_items():
        if items and item not in items:
            continue
        tb = pds.dataset(ES.PROC, partitioning='hive').to_table(
            columns=['S_Date'] + ES.ALLM + ES.ALLS, filter=pc.field('A_Name') == item).to_pandas()
        tb = tb.sort_values('S_Date').reset_index(drop=True)            # same ordering as ES.run_item
        d_ns = pd.to_datetime(tb['S_Date']).values.astype('datetime64[ns]')
        ram = tb[ES.ALLM].values.astype('float64'); ras = tb[ES.ALLS].values.astype('float64'); n = len(ram); del tb
        months = pd.period_range(pd.Timestamp(d_ns.min()).to_period('M') + 1,
                                 pd.Timestamp(d_ns.max()).to_period('M'), freq='M')
        anchors = []
        for mo in months:
            cut = int(np.searchsorted(d_ns, np.datetime64(mo.start_time, 'ns'), side='left'))
            if cut >= ES.N_TRAIN and cut + ES.M_SCORE <= n:
                anchors.append((str(mo), cut))
        if len(anchors) > ES.MAX_ANCHORS:
            anchors = [anchors[i] for i in np.linspace(0, len(anchors) - 1, ES.MAX_ANCHORS).astype(int)]
        for wname, (s, e) in ES.WINDOWS.items():
            j0, j1 = s - 1, e
            for mkey, cut in anchors:
                W = ram[cut - ES.N_TRAIN:cut, j0:j1] - ras[cut - ES.N_TRAIN:cut, j0:j1]
                realN = ram[cut:cut + ES.M_SCORE, j0:j1] - ras[cut:cut + ES.M_SCORE, j0:j1]
                lo, hi = ES.norm_fit(W)
                check_fit('s34', item, wname, W, {'real': realN}, lo, hi, bias=True)
        print(f'  s34 {item}: anchors={len(anchors)}', flush=True)
        del ram, ras
    write('s34')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', nargs='+', default=['s11', 's9', 's4', 'cu', 's34'])
    ap.add_argument('--items', nargs='+', default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for w in a.which:
        print(f'===== {w} =====', flush=True)
        {'s11': lambda: run_s11(a.items), 's9': run_s9, 's4': run_s4, 'cu': run_cu,
         's34': lambda: run_s34(a.items)}[w]()


if __name__ == '__main__':
    main()
