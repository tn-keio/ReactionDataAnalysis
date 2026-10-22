# Temporal robustness re-run with SCREENED + CONTEMPORANEOUS synthetic anomalies (trustworthy version).
# Fixes the two flaws of 20260706_TemporalRobustness_AllItems (fixed all-period, unscreened battery):
#   (1) contemporaneous: the battery is generated from the SCORING block (the incoming samples), not a
#       frozen all-period sample -> no period mismatch with the moving window model.
#   (2) inlier screening: the base curves are the robust-inliers of that block, so the synthetic
#       'clean' ground truth is genuinely normal (a random real curve may itself be an anomaly).
# Operationally faithful: train on the trailing N=500 (the past), FREEZE, score the SUBSEQUENT block
# (the future). Calibration = SELF (P99.5 on the training block). Metrics per anchor, per method (E/R):
#   recall_main(k>=2), recall_k1, per-type recall  (on the screened contemporaneous battery from S)
#   spec  = specificity on the REAL normal curves of S (the operational false-alarm rate)
# Writes to a NEW folder; 20260706 is left untouched.  No TensorFlow.

import os, sys, json, gc
import numpy as np, pandas as pd
import pyarrow.dataset as ds, pyarrow.compute as pc
from sklearn.covariance import EmpiricalCovariance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from contemporaneous_synth import make_battery, POS_TYPES

BASE = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
OUT = os.path.join(BASE, 'Results', '20260708_TemporalRobustness_screened')

ITEMS = {
    'ALB': 'ALB', 'B2M': 'B2M', 'C3': 'C3', 'C4': 'C4', 'CA': 'Ca', 'CRE': 'CRE', 'CRP': 'CRP',
    'CU': 'Cu', 'DB': 'DB', 'FE': 'Fe', 'FER': 'FER', 'GA': 'GA', 'GAALB': 'GAALB', 'GLU': 'GLU',
    'HDL-C': 'HDL-C', 'IGA': 'IgA', 'IGG': 'IgG', 'IGM': 'IgM', 'IP': 'IP', 'IgG4': 'IgG4',
    'KL-6': 'KL-6', 'LDL-C': 'LDL-C', 'LRG': 'LRG', 'Li': 'Li', 'MMP-3': 'MMP-3', 'TB': 'TB',
    'TC': 'TC', 'TG': 'TG', 'TP': 'TP', 'UA': 'UA', 'UIBC': 'UIBC', 'VALP': 'VALP', 'VCM': 'VCM',
    'ZN': 'Zn', 'mTP': 'mTP',
}
N = 500                # training window (trailing)
M_SCORE = 1000         # subsequent block scored (real normals + base for contemporaneous battery)
SCREEN_FRAC = 0.9      # keep the 90% most-inlier curves as the synthetic 'clean' base
BUDGET = 99.5          # self-calibration operating point
MIN_ANCHORS = 4
METHODS = ['E', 'R']
RAM = [f'RAM{i}' for i in range(11, 39)]
RAS = [f'RAS{i}' for i in range(11, 39)]


def norm_fit(a): return float(a.min()), float(a.max())
def apply_norm(a, lo, hi): return np.clip((a - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def mcd_like(D, keep=0.95, iters=5):
    loc = D.mean(0); cov = np.cov(D.T); n = int(len(D) * keep)
    for _ in range(iters):
        d = np.sum((D - loc) @ np.linalg.pinv(cov) * (D - loc), axis=1)
        core = D[np.argsort(d)[:n]]; loc, cov = core.mean(0), np.cov(core.T)
    return loc, cov


def maha(loc, cov, X):
    P = np.linalg.pinv(cov)
    return np.sum((X - loc) @ P * (X - loc), axis=1)


def fit_model(Wn, method):
    if method == 'E':
        e = EmpiricalCovariance().fit(Wn); return e.location_, e.covariance_
    return mcd_like(Wn)


def run_item(synth_item, proc_name):
    d = ds.dataset(PROC, partitioning='hive')
    tb = d.to_table(columns=['S_Date'] + RAM + RAS, filter=pc.field('A_Name') == proc_name).to_pandas()
    tb = tb.sort_values('S_Date').reset_index(drop=True)
    d_ns = pd.to_datetime(tb['S_Date']).values.astype('datetime64[ns]')
    ram = tb[RAM].values.astype('float64'); ras = tb[RAS].values.astype('float64')
    calc = ram - ras; n = len(calc); del tb
    months = pd.period_range(pd.Timestamp(d_ns.min()).to_period('M') + 1,
                             pd.Timestamp(d_ns.max()).to_period('M'), freq='M')
    rows = []
    for mo in months:
        cut = int(np.searchsorted(d_ns, np.datetime64(mo.start_time, 'ns'), side='left'))
        if cut < N or cut + M_SCORE > n:
            continue
        W = calc[cut - N:cut]; lo, hi = norm_fit(W); Wn = apply_norm(W, lo, hi)
        s_real = calc[cut:cut + M_SCORE]
        batt = make_battery(ram[cut:cut + M_SCORE], ras[cut:cut + M_SCORE], seed=0,
                            screen_frac=SCREEN_FRAC, include_benign=True)
        at = batt['atype']; sev = batt['severity']
        main = np.isin(at, POS_TYPES) & (sev >= 2); k1 = np.isin(at, POS_TYPES) & (sev == 1)
        tmask = {t: (at == t) & (sev >= 2) for t in POS_TYPES}
        Sn = apply_norm(batt['calc'], lo, hi); Rn = apply_norm(s_real, lo, hi)
        row = {'item': synth_item, 'month': str(mo), 'cut': cut}
        for m in METHODS:
            loc, cov = fit_model(Wn, m); thr = float(np.percentile(maha(loc, cov, Wn), BUDGET))
            sc = maha(loc, cov, Sn); rn = maha(loc, cov, Rn)
            row[f'{m}_recall_main'] = float((sc[main] >= thr).mean()) if main.any() else np.nan
            row[f'{m}_recall_k1'] = float((sc[k1] >= thr).mean()) if k1.any() else np.nan
            row[f'{m}_spec'] = float((rn < thr).mean())
            for t in POS_TYPES:
                row[f'{m}_rec_{t}'] = float((sc[tmask[t]] >= thr).mean()) if tmask[t].any() else np.nan
        rows.append(row)
    del calc, ram, ras; gc.collect()
    if len(rows) < MIN_ANCHORS:
        return {'item': synth_item, 'proc': proc_name, 'status': 'insufficient', 'n_anchors': len(rows)}, []
    df = pd.DataFrame(rows)
    rec = {'item': synth_item, 'proc': proc_name, 'status': 'ok', 'n_anchors': len(df), 'n_real': int(n)}
    for m in METHODS:
        for key in ['recall_main', 'recall_k1', 'spec']:
            rec[f'{m}_{key}_mean'] = round(float(df[f'{m}_{key}'].mean()), 3)
            rec[f'{m}_{key}_min'] = round(float(df[f'{m}_{key}'].min()), 3)
    return rec, rows


def make_figures(series_df, summ_df):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ok = summ_df[summ_df.status == 'ok'].copy().sort_values('R_recall_main_min')
    items = ok['item'].tolist(); y = np.arange(len(items)); h = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.34 * len(items) + 1.5))
    for a, (metric, title) in zip(ax, [('recall_main_min', 'min recall_main(k>=2) over anchors'),
                                        ('spec_min', 'min specificity (real normals) over anchors')]):
        a.barh(y - h/2, ok[f'E_{metric}'], h, color='#d62728', label='Empirical')
        a.barh(y + h/2, ok[f'R_{metric}'], h, color='#e377c2', label='Robust')
        a.set_yticks(y); a.set_yticklabels(items, fontsize=7); a.invert_yaxis()
        a.set_xlim(0, 1.02); a.axvline(0.9, color='0.6', ls=':', lw=0.8)
        a.set_title(title); a.grid(alpha=0.3, axis='x'); a.legend(fontsize=8, loc='lower right')
    fig.suptitle('Screened + contemporaneous temporal robustness (worst anchor per item)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(OUT, 'overview_screened.png'), dpi=170); plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--items', nargs='+', default=list(ITEMS.keys()))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    json.dump({'N': N, 'M_score': M_SCORE, 'screen_frac': SCREEN_FRAC, 'budget': BUDGET,
               'calibration': 'self', 'synth': 'contemporaneous+screened (from subsequent block)',
               'note': 'trustworthy re-run of 20260706_AllItems; that folder left untouched',
               'date': '2026-07-08'}, open(os.path.join(OUT, 'run_config.json'), 'w'), indent=2)
    results, series = [], []
    todo = [x for x in args.items if x in ITEMS]
    for i, it in enumerate(todo, 1):
        print(f'[{i}/{len(todo)}] {it} ...', flush=True)
        try:
            rec, rows = run_item(it, ITEMS[it])
        except Exception as ex:
            rec, rows = {'item': it, 'proc': ITEMS[it], 'status': f'error: {type(ex).__name__}: {ex}'}, []
            print('   ERROR', ex, flush=True)
        results.append(rec); series.extend(rows)
        if rec['status'] == 'ok':
            print(f"   anchors={rec['n_anchors']}  E recall_main.min={rec['E_recall_main_min']:.2f} "
                  f"spec.min={rec['E_spec_min']:.2f} | R recall_main.min={rec['R_recall_main_min']:.2f} "
                  f"spec.min={rec['R_spec_min']:.2f}", flush=True)
        else:
            print('   ', rec['status'], flush=True)
    summ = pd.DataFrame(results); summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    if series:
        pd.DataFrame(series).to_csv(os.path.join(OUT, 'series_all.csv'), index=False)
        make_figures(pd.DataFrame(series), summ)
    print('\nDONE ->', OUT)


if __name__ == '__main__':
    main()
