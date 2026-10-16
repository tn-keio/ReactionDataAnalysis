# MAIN-ESTIMATOR SELECTION -- Empirical vs Robust (vs E-union-R, Ledoit-Wolf) at points 5-38.
#
# Resolves the unresolved main-estimator question (Users-overview.md S0 / PaperOverview S3): Robust
# wins on specificity but its recall collapses on some items (DB 0.43, IP/TB/UA) because it is blind to
# B_drift; Empirical keeps recall but over-flags on a few items (Ca). Prior comparisons used points
# 11-38; the window is now fixed at 5-38, so we re-run the E/R trade-off there and test whether the OR
# combination (E union R) achieves recall~=E and specificity~=R across all items.
#
# Operationally-faithful, matching run_temporal_robustness_screened / run_reaction_points_forpaper:
#   Calc = RAM - RAS on the chosen window; per-window min-max normalization fit on the trailing N=500;
#   self-calibrated P99.5 threshold; monthly rolling anchors; contemporaneous inlier-screened synthetic
#   battery regenerated per scoring block. 21 ForPaper items. No TensorFlow.
#
# ***** PLAN STAGE: created but NOT yet executed. Validate on first run. *****
#
# Fair-comparison protocol: every estimator is calibrated to the SAME in-sample flag COUNT before
# comparison. E/R/LW use the P99.5 of their own training distances (at N=500 this flags exactly 3
# samples = 0.6% in-sample; budget granularity is 1/N = 0.2%, so an exact 0.5% is unreachable). OR is
# calibrated by or_match_count() to the tightest common percentile whose UNION count <= that same
# 3-sample budget, so OR is never given a looser budget than the singles (review point 1). The achieved
# in-sample flag rate of every method is recorded (series.insample_flag / calibration_diagnostics.csv).
# Metrics are read at that operating point (never AUC): per-anomaly-type recall (esp. B_drift), severity
# response (k=1..8), specificity/flag-rate on real normals, benign_offset control, E/R complementarity,
# plus calibration_diagnostics.csv (thr_E, thr_R, thr_ratio, B_drift-k2 distance vs threshold) to test
# the Robust-threshold-inflation hypothesis (review point 2).
#
# Estimator note (review point 3): LW (Ledoit-Wolf) shrinks the covariance toward a scaled identity to
# regularize small-N noise; it is NOT outlier-robust (no trimming) -- it is "N-regularized Empirical",
# outlier-fragile like E, and is NOT a substitute for R's specificity. It is included to test whether
# shrinkage (rather than trimming) stabilizes small N WITHOUT R's B_drift blindness.

import os, sys, json, gc, time
import numpy as np, pandas as pd
import pyarrow.dataset as ds, pyarrow.compute as pc
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
PARAM = os.path.join(BASE, 'Data', 'parameter_mapped.csv')
OUT = os.path.join(BASE, 'Results', '20260716_EstimatorSelection_E_vs_R')

WINDOWS = {'p5_38': (5, 38), 'p11_38': (11, 38)}     # primary p5_38; p11_38 = reference contrast
N_TRAIN = 500
M_SCORE = 1500
BUDGET = 0.5                                          # training FP budget (%) -> single-est P99.5
SEEDS = [0, 1, 2]
SCREEN_FRAC = 0.9
K_GRID = [1, 2, 3, 5, 8]
POS_TYPES = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']
METHODS = ['E', 'R', 'OR', 'LW']
MAX_ANCHORS = 30
ALLM = [f'RAM{i}' for i in range(1, 39)]
ALLS = [f'RAS{i}' for i in range(1, 39)]


# ---------------- window-generalized contemporaneous battery (T inferred from the block) ----------
SETTLE = 3
BENIGN_K = [2, 5, 8]


def _rcov(X, keep=0.9, iters=5):
    loc = X.mean(0); cov = np.cov(X.T); n = max(2, int(len(X) * keep))
    for _ in range(iters):
        d = np.sum((X - loc) @ np.linalg.pinv(cov) * (X - loc), axis=1)
        core = X[np.argsort(d)[:n]]; loc, cov = core.mean(0), np.cov(core.T)
    return loc, cov


def _screen(ram_b, ras_b, frac):
    calc = np.asarray(ram_b, float) - np.asarray(ras_b, float)
    loc, cov = _rcov(calc)
    d = np.sum((calc - loc) @ np.linalg.pinv(cov) * (calc - loc), axis=1)
    return np.argsort(d)[:max(1, int(len(calc) * frac))]


def make_battery(ram_block, ras_block, seed=0, m_base=200, screen_frac=0.9, include_benign=True):
    """Standard A/B/C/E (+benign) battery on an arbitrary-width window (T inferred). Mirrors
    contemporaneous_synth.make_battery but T is taken from the block width (works for 34- or 28-pt
    windows). Anomaly injected on RAM only -> survives in Calc; benign shifts RAM & RAS equally."""
    rng = np.random.RandomState(seed)
    ram_block = np.asarray(ram_block, float); ras_block = np.asarray(ras_block, float)
    if screen_frac and screen_frac < 1.0:
        keep = _screen(ram_block, ras_block, screen_frac)
        ram_block, ras_block = ram_block[keep], ras_block[keep]
    n, T = ram_block.shape
    IDX = np.arange(T)
    mu = ram_block.mean(0); sd = ram_block.std(0)
    onset = int(np.argmax(np.abs(np.diff(mu))))
    sigbar = float(np.median(sd)) + 1e-9
    r0 = onset
    # unit shapes
    denom = max(1, (T - 1) - onset)
    ramp = np.clip((IDX - onset) / denom, 0, 1)
    hump = np.exp(-((IDX - (onset + 2)) / 1.6) ** 2) - 0.5 * np.exp(-((IDX - (min(T - 1, onset + SETTLE + 1) + 3)) / 2.0) ** 2)
    hump = hump / (np.abs(hump).max() + 1e-9)
    base_shift = np.where(IDX < onset, 1.0, np.clip((onset + 2 - IDX) / 2.0, 0, 1))
    M = min(m_base, n)
    sel = rng.permutation(n)[:M]
    base_ram = ram_block[sel]; base_ras = ras_block[sel]
    spike_pt = rng.randint(max(r0, 1), T, size=M)
    rows, atype, sev = [], [], []

    def add(name, s, ram_pert):
        rows.append(ram_pert - base_ras); atype.extend([name] * M); sev.extend([float(s)] * M)

    add('clean', 0.0, base_ram)
    for k in K_GRID:
        A = k * sigbar
        pert = np.zeros((M, T))
        for b in range(M):
            pert[b] = A * np.exp(-((IDX - spike_pt[b]) / 0.9) ** 2)
        add('A_spike', k, base_ram + pert)
        add('B_drift', k, base_ram + A * ramp[None, :])
        add('C_hump', k, base_ram + A * hump[None, :])
        add('E_baseline', k, base_ram + A * base_shift[None, :])
    if include_benign:
        for k in BENIGN_K:
            add('benign_offset', k, base_ram + k * sigbar)                 # RAS also shifted -> cancels
            rows[-1] = (base_ram + k * sigbar) - (base_ras + k * sigbar)
    return dict(calc=np.concatenate(rows, 0), atype=np.array(atype), severity=np.array(sev, float), onset=onset)


# ---------------- estimators / distances ----------------
def norm_fit(a): return a.min(0), a.max(0)
def apply_norm(a, lo, hi): return (a - lo) / (hi - lo + 1e-8)      # affine, no clipping


def mcd_like(D, keep=0.95, iters=5):
    loc = D.mean(0); cov = np.cov(D.T); n = int(len(D) * keep)
    for _ in range(iters):
        d = np.sum((D - loc) @ np.linalg.pinv(cov) * (D - loc), axis=1)
        core = D[np.argsort(d)[:n]]; loc, cov = core.mean(0), np.cov(core.T)
    return loc, cov


def maha(loc, cov, X):
    P = np.linalg.pinv(cov)
    return np.sum((X - loc) @ P * (X - loc), axis=1)


def fit_est(Wn, m):
    if m == 'E':
        e = EmpiricalCovariance().fit(Wn); return e.location_, e.covariance_
    if m == 'LW':
        e = LedoitWolf().fit(Wn); return e.location_, e.covariance_
    return mcd_like(Wn)                                                     # 'R'


def or_match_count(dE_tr, dR_tr, n_target):
    """Calibrate the OR (E-union-R) operating point to the SAME in-sample flag COUNT as the single
    estimators, using a common percentile q for both channels.

    Rationale (fair counting at N=500): the single estimators self-calibrated at P99.5 flag a discrete
    number of training samples (3 for N=500 == 0.6%, not 0.5%; budget granularity is 1/N = 0.2%).
    Calibrating OR to a nominal 0.5% union would make it TIGHTER than the singles and bias the
    comparison. Instead we find the tightest common q whose union count is <= n_target (== the singles'
    count), so OR is never given a looser budget than the estimators it is compared against. Returns the
    two thresholds, q, and the ACHIEVED union count (recorded for transparency)."""
    def ucount(q):
        tE = np.percentile(dE_tr, q); tR = np.percentile(dR_tr, q)
        return int(((dE_tr >= tE) | (dR_tr >= tR)).sum()), float(tE), float(tR)
    lo, hi = 99.0, 100.0
    for _ in range(60):                       # find smallest q with union count <= n_target
        mid = 0.5 * (lo + hi)
        c, _, _ = ucount(mid)
        if c > n_target:
            lo = mid                          # too many flags -> tighten (raise q)
        else:
            hi = mid
    c, tE, tR = ucount(hi)
    return tE, tR, hi, c


def forpaper_items():
    import glob
    parts = {os.path.basename(p).replace('A_Name=', '') for p in glob.glob(os.path.join(PROC, 'A_Name=*'))}
    low = {p.lower(): p for p in parts}
    fp = pd.read_csv(PARAM); fp = fp[fp['ForPaper'] == 1]['ItemName'].tolist()
    return sorted(low[x.lower()] for x in fp)


def run_item(item):
    d = ds.dataset(PROC, partitioning='hive')
    tb = d.to_table(columns=['S_Date'] + ALLM + ALLS, filter=pc.field('A_Name') == item).to_pandas()
    tb = tb.sort_values('S_Date').reset_index(drop=True)
    d_ns = pd.to_datetime(tb['S_Date']).values.astype('datetime64[ns]')
    ram = tb[ALLM].values.astype('float64'); ras = tb[ALLS].values.astype('float64'); n = len(ram); del tb
    months = pd.period_range(pd.Timestamp(d_ns.min()).to_period('M') + 1,
                             pd.Timestamp(d_ns.max()).to_period('M'), freq='M')
    anchors = []
    for mo in months:
        cut = int(np.searchsorted(d_ns, np.datetime64(mo.start_time, 'ns'), side='left'))
        if cut >= N_TRAIN and cut + M_SCORE <= n:
            anchors.append((str(mo), cut))
    if len(anchors) > MAX_ANCHORS:
        anchors = [anchors[i] for i in np.linspace(0, len(anchors) - 1, MAX_ANCHORS).astype(int)]

    rows, cost, diag = [], [], []
    for wname, (s, e) in WINDOWS.items():
        j0, j1 = s - 1, e                                                  # column slice for points s..e
        for mkey, cut in anchors:
            W = ram[cut - N_TRAIN:cut, j0:j1] - ras[cut - N_TRAIN:cut, j0:j1]
            lo, hi = norm_fit(W); Wn = apply_norm(W, lo, hi)
            realN = ram[cut:cut + M_SCORE, j0:j1] - ras[cut:cut + M_SCORE, j0:j1]
            Rn = apply_norm(realN, lo, hi)
            # fit estimators (+cost)
            fitted, dtr, dreal = {}, {}, {}
            for m in ['E', 'R', 'LW']:
                t0 = time.perf_counter(); loc, cov = fit_est(Wn, m); dt = time.perf_counter() - t0
                fitted[m] = (loc, cov); dtr[m] = maha(loc, cov, Wn); dreal[m] = maha(loc, cov, Rn)
                cost.append({'item': item, 'window': wname, 'method': m, 'fit_sec': dt})
            thr = {m: float(np.percentile(dtr[m], 100 - BUDGET)) for m in ['E', 'R', 'LW']}
            insample = {m: float((dtr[m] >= thr[m]).mean()) for m in ['E', 'R', 'LW']}
            n_target = int((dtr['E'] >= thr['E']).sum())                    # singles' actual in-sample count
            tE_or, tR_or, q_or, c_or = or_match_count(dtr['E'], dtr['R'], n_target)  # matched-count OR
            insample['OR'] = c_or / len(dtr['E'])
            # diagnostic for the "Robust threshold explosion" hypothesis (review point 2): is the
            # self-cal threshold set by the trimmed outliers, i.e. thr_R >> a typical anomaly distance?
            cur_diag = {'item': item, 'window': wname, 'month': mkey,
                        'thr_E': thr['E'], 'thr_R': thr['R'], 'thr_ratio_R_over_E': thr['R'] / (thr['E'] + 1e-12),
                        'insample_E': insample['E'], 'insample_R': insample['R'], 'insample_OR': insample['OR'],
                        'n_target': n_target, 'or_union_count': c_or, 'q_or': q_or}
            diag.append(cur_diag)
            # real-normal flags -> specificity / flag rate ; complementarity on real normals
            flag_real = {'E': dreal['E'] >= thr['E'], 'R': dreal['R'] >= thr['R'], 'LW': dreal['LW'] >= thr['LW']}
            flag_real['OR'] = (dreal['E'] >= tE_or) | (dreal['R'] >= tR_or)
            # synthetic battery (avg over seeds)
            for seed in SEEDS:
                batt = make_battery(ram[cut:cut + M_SCORE, j0:j1] + 0, ras[cut:cut + M_SCORE, j0:j1] + 0,
                                    seed=seed, screen_frac=SCREEN_FRAC, include_benign=True)
                at, sv = batt['atype'], batt['severity']
                Sn = apply_norm(batt['calc'], lo, hi)
                dS = {m: maha(*fitted[m], Sn) for m in ['E', 'R', 'LW']}
                flag_S = {'E': dS['E'] >= thr['E'], 'R': dS['R'] >= thr['R'], 'LW': dS['LW'] >= thr['LW'],
                          'OR': (dS['E'] >= tE_or) | (dS['R'] >= tR_or)}
                benign = at == 'benign_offset'
                if seed == SEEDS[0]:
                    # test point 2: is B_drift(k=2) distance below R's (possibly inflated) threshold
                    # while above E's? (thr_R already logged; here record the anomaly-distance side)
                    bd2 = (at == 'B_drift') & (sv == 2)
                    if bd2.any():
                        cur_diag['medDist_Bdrift_k2_E'] = float(np.median(dS['E'][bd2]))
                        cur_diag['medDist_Bdrift_k2_R'] = float(np.median(dS['R'][bd2]))
                        cur_diag['Bdrift_k2_below_thrR'] = float((dS['R'][bd2] < thr['R']).mean())
                        cur_diag['Bdrift_k2_below_thrE'] = float((dS['E'][bd2] < thr['E']).mean())
                for m in METHODS:
                    row = {'item': item, 'window': wname, 'method': m, 'month': mkey, 'seed': seed,
                           'q_or': q_or if m == 'OR' else np.nan}
                    if seed == SEEDS[0]:
                        row['spec'] = float((~flag_real[m]).mean()); row['flag_rate_real'] = float(flag_real[m].mean())
                        row['insample_flag'] = insample[m]
                    # per-type recall (sev>=2) and k=1
                    main = np.isin(at, POS_TYPES) & (sv >= 2)
                    row['recall_main'] = float(flag_S[m][main].mean()) if main.any() else np.nan
                    for t in POS_TYPES:
                        sub = (at == t) & (sv >= 2)
                        row[f'recall_{t}'] = float(flag_S[m][sub].mean()) if sub.any() else np.nan
                    k1 = np.isin(at, POS_TYPES) & (sv == 1)
                    row['recall_k1'] = float(flag_S[m][k1].mean()) if k1.any() else np.nan
                    bd1 = (at == 'B_drift') & (sv == 1)
                    row['recall_Bdrift_k1'] = float(flag_S[m][bd1].mean()) if bd1.any() else np.nan
                    row['benign_flag'] = float(flag_S[m][benign].mean()) if benign.any() else np.nan
                    rows.append(row)
            # complementarity (real normals + battery main), seed 0 only, E vs R
            batt0 = make_battery(ram[cut:cut + M_SCORE, j0:j1], ras[cut:cut + M_SCORE, j0:j1],
                                 seed=SEEDS[0], screen_frac=SCREEN_FRAC, include_benign=False)
            Sn0 = apply_norm(batt0['calc'], lo, hi); main0 = np.isin(batt0['atype'], POS_TYPES) & (batt0['severity'] >= 2)
            fE = maha(*fitted['E'], Sn0)[main0] >= thr['E']; fR = maha(*fitted['R'], Sn0)[main0] >= thr['R']
            rows.append({'item': item, 'window': wname, 'method': 'COMPL', 'month': mkey, 'seed': SEEDS[0],
                         'compl_both': float((fE & fR).mean()), 'compl_Eonly': float((fE & ~fR).mean()),
                         'compl_Ronly': float((~fE & fR).mean()), 'compl_neither': float((~fE & ~fR).mean()),
                         'real_Eonly_flag': float((flag_real['E'] & ~flag_real['R']).mean()),
                         'real_Ronly_flag': float((flag_real['R'] & ~flag_real['E']).mean())})
    del ram, ras; gc.collect()
    return rows, cost, diag, len(anchors)


def main():
    os.makedirs(OUT, exist_ok=True)
    items = forpaper_items()
    print(f'ForPaper items ({len(items)}); windows={list(WINDOWS)}; methods={METHODS}', flush=True)
    all_rows, all_cost, all_diag, meta = [], [], [], {}
    for it in items:
        print(f'{it} ...', flush=True)
        r, c, dg, na = run_item(it); all_rows += r; all_cost += c; all_diag += dg; meta[it] = na
        print(f'   anchors={na}', flush=True)
    series = pd.DataFrame(all_rows); series.to_csv(os.path.join(OUT, 'series.csv'), index=False)
    pd.DataFrame(all_cost).groupby(['window', 'method']).fit_sec.mean().reset_index() \
        .to_csv(os.path.join(OUT, 'compute_cost.csv'), index=False)
    # calibration + Robust-threshold-inflation diagnostics (review points 1 & 2)
    diag = pd.DataFrame(all_diag); diag.to_csv(os.path.join(OUT, 'calibration_diagnostics.csv'), index=False)

    det = series[series.method.isin(METHODS)].copy()
    metric_cols = ['spec', 'flag_rate_real', 'recall_main', 'recall_k1', 'recall_Bdrift_k1', 'benign_flag'] + \
                  [f'recall_{t}' for t in POS_TYPES]
    by_item = det.groupby(['item', 'window', 'method'])[metric_cols].mean().reset_index()
    by_item.to_csv(os.path.join(OUT, 'summary_by_item.csv'), index=False)
    # overall: mean and worst-item (min) across items
    ov = by_item.groupby(['window', 'method']).agg(
        spec_mean=('spec', 'mean'), spec_min=('spec', 'min'),
        recall_main_mean=('recall_main', 'mean'), recall_main_min=('recall_main', 'min'),
        recall_Bdrift_mean=('recall_B_drift', 'mean'), recall_Bdrift_min=('recall_B_drift', 'min'),
        recall_k1_mean=('recall_k1', 'mean')).reset_index().round(4)
    ov.to_csv(os.path.join(OUT, 'summary_overall.csv'), index=False)
    series[series.method == 'COMPL'].groupby(['item', 'window']).mean(numeric_only=True).reset_index() \
        .to_csv(os.path.join(OUT, 'complementarity.csv'), index=False)

    json.dump({'items': items, 'windows': WINDOWS, 'N_train': N_TRAIN, 'M_score': M_SCORE,
               'budget_pct': BUDGET, 'seeds': SEEDS, 'methods': METHODS, 'anchors': meta,
               'date': '2026-07-16'}, open(os.path.join(OUT, 'run_config.json'), 'w'), indent=2)

    pd.set_option('display.width', 240)
    print('\n===== overall (window x method): B_drift is the discriminator =====')
    print(ov.to_string(index=False))
    print('\n-> ', OUT)


if __name__ == '__main__':
    main()
