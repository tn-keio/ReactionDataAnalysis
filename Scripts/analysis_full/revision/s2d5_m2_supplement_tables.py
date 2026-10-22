# Stage 2d-5: Markdown tables for Supplementary §S8.5-S8.6 from the result files
# (summary statistics only; no new estimation).
#   Table S8-2  k50 per method (median over 84 assay-type combinations, censored at > 8 sigma-bar)   <- s2d2
#   Table S8-3  recall of kinetic slowing / premature termination by level (N = 500, P99.5)        <- v2 JSONL (+ prozone)
#   Table S8-4  per assay: photometric points, f, calibration slope, median value, three limits     <- s2a2, s2d2
#   Table S8-5  mean recall at the smallest magnitude reaching each limit                          <- s2d2 + s2d4 (affected curves)
#   Table S8-6  DB and CRP: absolute concentration change (median, at k = 1, r = 0.8, q = 0.8)      <- s2a2
# Output: Results/20261021_Rev_M2_SupplementTables/m2_supplement_tables.md
# py3.14.
import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(BASE, 'Results')
YS = os.path.join(RES, '20261020_Rev_M2_Yardsticks')
BP = os.path.join(RES, '20261020_Rev_M2_BiasPhotometric')
AFF = os.path.join(RES, '20261021_Rev_M2_AffectedRecall', 'affected_summary.csv')
V2 = os.path.join(RES, '20260917_Rev_v2Benchmark')
PZ = os.path.join(RES, '20260917_Rev_v2Benchmark_prozone', 'main', 'Reports', 'comparison.jsonl')
OUT = os.path.join(RES, '20261021_Rev_M2_SupplementTables')
M5 = ['maha', 'maha_robust', 'pca', 'lof', 'dae_elbow']
H5 = ['Mahalanobis (empirical)', 'Mahalanobis (robust)', 'PCA', 'LOF', 'Compact dense AE (elbow)']
NAMES = {'maha': 'Mahalanobis (empirical)', 'maha_robust': 'Mahalanobis (robust)', 'pca': 'PCA', 'lof': 'LOF',
         'dae_elbow': 'Compact dense AE (elbow)', 'cae8': 'Compact convolutional AE (z = 8)',
         'ae2d_ens': 'AE ⊕ 2D-CNN (ensemble)', 'ae': '1D-AE', 'twod_only': '2D-CNN → iForest',
         'ae2d_and': 'AE ∧ 2D-CNN (AND)', 'ocsvm': 'One-Class SVM', 'iforest': 'Isolation Forest',
         'cnn1d': '1D-CNN → iForest'}
TYPES = [('A_spike', 'Spike*'), ('B_drift', 'Drift'), ('C_hump', 'Hump'), ('E_baseline', 'Baseline shift'),
         ('S_saturation', 'Premature termination*'), ('W_timewarp', 'Kinetic slowing*')]
AFF_T = {'A_spike', 'S_saturation', 'W_timewarp'}


def f2(v, d=2):
    if pd.isna(v):
        return '—'
    return f'{v:.3f}' if (abs(v) < 0.01 and v != 0) else f'{v:.{d}f}'


def t2():
    k = pd.read_csv(os.path.join(YS, 'k50_summary_13.csv')).set_index('method')
    k = k.drop(index='dae_oracle').sort_values('k50_cell_median')
    rows = ['| Method | Median k50 (σ̄) | Combinations not reaching 50% within 8σ̄ (of 84) |', '|---|---:|---:|']
    for m, r in k.iterrows():
        med = '> 8' if not np.isfinite(r['k50_cell_median']) else f"{r['k50_cell_median']:.2f}"
        rows.append(f"| {NAMES[m]} | {med} | {int(r['n_cells_censored'])} |")
    return '\n'.join(rows)


def jsonl():
    recs = {}
    for g in ('main', 'ca', 'm2'):
        for l in open(os.path.join(V2, g, 'Reports', 'comparison.jsonl'), encoding='utf-8'):
            if l.strip():
                d = json.loads(l); recs[(d['item'], d['method'], d['N'])] = d
    pz = {}
    for l in open(PZ, encoding='utf-8'):
        if l.strip():
            d = json.loads(l); pz[(d['item'], d['method'], d['N'])] = d
    return recs, pz


def t3():
    recs, pz = jsonl()
    z = pd.read_csv(os.path.join(YS, 'dae_z_selection.csv'))
    ze = {r['item']: f"dae{int(r['z_elbow'])}" for _, r in z[z.N == 500].iterrows()}
    items = sorted({k[0] for k in recs}); pitems = sorted({k[0] for k in pz})
    def mean(src, its, m, key):
        v = []
        for it in its:
            r = src.get((it, ze[it] if m == 'dae_elbow' else m, 500))
            if r is not None and key in r['budgets']['99.5']:
                v.append(r['budgets']['99.5'][key]['mean'])
        assert len(v) == len(its), (m, key, len(v))
        return np.mean(v)
    rows = ['| Disturbance and level | n | ' + ' | '.join(H5) + ' |', '|---|---:|' + '---:|' * 5]
    for r in (0.95, 0.9, 0.8, 0.7, 0.5):
        rows.append(f'| Kinetic slowing, r = {r:g} | {len(items)} | ' + ' | '.join(f2(mean(recs, items, m, f"recall_W_r{r:g}"), 3) for m in M5) + ' |')
    for q in (0.98, 0.95, 0.9, 0.8):
        rows.append(f'| Premature termination, q = {q:g} | {len(items)} | ' + ' | '.join(f2(mean(recs, items, m, f"recall_S_q{q:g}"), 3) for m in M5) + ' |')
    for q in (0.5, 0.2):
        rows.append(f'| Premature termination, q = {q:g} | {len(pitems)} | ' + ' | '.join(f2(mean(pz, pitems, m, f"recall_S_q{q:g}"), 3) for m in M5) + ' |')
    return '\n'.join(rows), pitems


def med_fmt(v):
    return f'{v:.0f}' if v >= 100 else f'{v:.3g}'


def t4():
    pts = pd.read_csv(os.path.join(BP, 'points_used.csv')).set_index('item')
    cal = pd.read_csv(os.path.join(BP, 'calibration_fit.csv')).set_index('item')
    lim = pd.read_csv(os.path.join(YS, 'm2_limits_three_yardsticks.csv')).set_index('item')
    aps = pd.read_csv(os.path.join(YS, 'yardstick_inputs', 'eflm_aps_panel21.csv')).set_index('item')
    rows = ['| Assay | Baseline points | End points | f | Calibration slope (per abs. unit) | Median value | EFLM desirable bias (%) | EFLM desirable TEa (%) | CLIA rule | CLIA at median (%) |',
            '|---|---|---|---:|---:|---:|---:|---:|---|---:|']
    for it in pts.index:
        l = lim.loc[it]
        db = aps.loc[it, 'desirable_bias_pct'] if it in aps.index else np.nan
        te = aps.loc[it, 'TEa_pct'] if it in aps.index else np.nan
        rule = l['clia_rule'] if isinstance(l['clia_rule'], str) else 'none'
        note = ' †' if it in ('DB', 'CRP') else (' ‡' if it == 'KL-6' else '')
        rows.append(f"| {it}{note} | {pts.loc[it, 'base_points']} | {pts.loc[it, 'end_points']} | {pts.loc[it, 'f']:.3f} | "
                    f"{cal.loc[it, 'slope_block_median']:.4g} | {med_fmt(cal.loc[it, 'median_value'])} | {f2(db)} | {f2(te)} | "
                    f"{rule.replace('±', '±')} | {f2(l['CLIA_pct'], 1)} |")
    return '\n'.join(rows)


def t5():
    lr = pd.read_csv(os.path.join(YS, 'm2_recall_at_limit_long.csv'))
    lr = lr[lr.bias_def == 'curves_with_substantive_change']
    af = pd.read_csv(AFF)
    rows = ['| Limit | Anomaly | Assays reaching the limit | ' + ' | '.join(H5) + ' |', '|---|---|---:|' + '---:|' * 5]
    for ys in ('EFLM desirable bias', 'EFLM desirable TEa', 'CLIA (reference)'):
        for t, lab in TYPES:
            if t in AFF_T:
                g = af[(af.yardstick == ys) & (af.anomaly_type == t)].set_index('method')
                if g.empty:
                    r = lr[(lr.yardstick == ys) & (lr.anomaly_type == t)].iloc[0]
                    n = f"{int(r['n_items_reached'])}/{int(r['n_items_evaluated'])}"; vals = ['—'] * 5
                else:
                    n = f"{int(g['n_items_reached'].iloc[0])}/{int(g['n_items_evaluated'].iloc[0])}"
                    vals = [f2(g.loc[m, 'recall_aff_mean']) for m in M5]
            else:
                r = lr[(lr.yardstick == ys) & (lr.anomaly_type == t)].iloc[0]
                n = f"{int(r['n_items_reached'])}/{int(r['n_items_evaluated'])}"
                vals = [f2(r[m]) if r['n_items_reached'] > 0 else '—' for m in M5]
            rows.append(f"| {ys if t == 'A_spike' else ''} | {lab} | {n} | " + ' | '.join(vals) + ' |')
    return '\n'.join(rows)


def t6():
    bs = pd.read_csv(os.path.join(BP, 'bias_summary_photometric.csv'))
    rows = ['| Assay | Spike* (k = 1) | Drift (k = 1) | Hump (k = 1) | Baseline shift (k = 1) | Premature termination* (q = 0.8) | Kinetic slowing* (r = 0.8) |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for it in ('DB', 'CRP'):
        cells = []
        for t, s in (('A_spike', 1.0), ('B_drift', 1.0), ('C_hump', 1.0), ('E_baseline', 1.0), ('S_saturation', 0.8), ('W_timewarp', 0.8)):
            r = bs[(bs['item'] == it) & (bs.anomaly_type == t) & np.isclose(bs.severity, s)]
            assert len(r) == 1, (it, t)
            col = 'abs_change_react_median' if t in AFF_T else 'abs_change_median'
            cells.append(f'{r[col].iloc[0]:.3f}')
        rows.append(f'| {it} (mg/dL) | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    s3, pitems = t3()
    md = ['## Table S8-2', t2(), '', '## Table S8-3', s3, f'(q = 0.5 and 0.2: {", ".join(pitems)})', '',
          '## Table S8-4', t4(), '', '## Table S8-5', t5(), '', '## Table S8-6', t6(), '']
    open(os.path.join(OUT, 'm2_supplement_tables.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
