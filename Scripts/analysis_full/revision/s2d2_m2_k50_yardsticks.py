# Stage 2d-2: the supplement table with three yardsticks
# and k50 for all thirteen Table 1 methods (cae8 and the classical/DL methods that s2d did not cover).
#   (1) k50 at N = 500 for the 13 methods (+ dae_oracle as the S5 upper bound), with the corrected compact-AE
#       selection (s1z4: dae_z_selection_fixed.csv).
#   (2) s2d's clinically-relevant-severity analysis run three times, with the limit taken from
#       EFLM desirable bias (main), EFLM desirable TEa, and CLIA acceptance limits (reference).
#       CLIA limits (42 CFR 493.931 routine chemistry, 493.927 general immunology; current eCFR) are used in
#       percent at the item's median reported value (the first 15,000 date-sorted curves, as in
#       run_paper_method_comparison.py), because s2a2 expresses the bias relative to that median: for a "% or units,
#       whichever is greater" rule the greater of the two at the median is used (TB 56%, CRE 24%; the percent
#       component is listed alongside), and a limit stated only in units (Ca) is converted.  Analytes without a CLIA
#       criterion stay NaN.  DB and CRP (median near zero) are not compared in percent.
#   (3) k50 summary with the non-reaching cells kept: per (item, type) cell, a method that never reaches 50%
#       within k <= 8 is right-censored at > 8; the median over all 84 cells is reported with the number of
#       censored cells, next to s2d's mean of the per-item pooled k50 (which drops non-reaching cells).
#       (The two-stage nanmean hides partial failures, e.g. iForest.)
# s2d is imported unchanged; only its module globals (V2, BAT, METHODS, YARDSTICK, load_recs) are redirected.
# Output: Results/20261020_Rev_M2_Yardsticks/  (summary statistics only)
# py3.14 (no TensorFlow).
import os
import sys
import json
import shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2d_clinical_severity as D     # noqa: E402

V2_SRC = D.V2                          # JSONL inputs (20260917_Rev_v2Benchmark)
BAT_SRC = D.BAT
OUT = os.path.join(D.RES, '20261020_Rev_M2_Yardsticks')
SEL = os.path.join(D.RES, '20261013_Rev_M3_CompactSelFix', 'dae_z_selection_fixed.csv')
PROC = os.path.join(D.BASE, 'Data', 'BigData', 'processed')
N_BENCH = 15000                        # 10,000 train pool + 2,000 holdout + 3,000 analysis
METHODS13 = ['maha', 'maha_robust', 'pca', 'dae_elbow', 'cae8', 'lof', 'ae2d_ens', 'ae', 'twod_only',
             'ae2d_and', 'ocsvm', 'iforest', 'cnn1d', 'dae_oracle']
YARDSTICKS = ['desirable_bias_pct', 'TEa_pct', 'CLIA_pct']   # main, EFLM TEa, CLIA (reference)
BIAS_SRC = os.path.join(D.RES, '20261020_Rev_M2_BiasPhotometric', 'bias_summary_photometric.csv')
BIAS_DEF = {'step_incl': 'all_curves', 'react_only': 'curves_with_substantive_change'}
NEAR_ZERO = {'DB', 'CRP'}              # left out of the percent comparison (median value near zero; s2a2)
PROZONE = os.path.join(D.RES, '20260917_Rev_v2Benchmark_prozone', 'main', 'Reports', 'comparison.jsonl')
Q_PROZONE = [0.5, 0.2]
# (percent, absolute limit, unit).  None = no CLIA criterion for this analyte.
CLIA = {'TP': (8, None, ''), 'ALB': (8, None, ''), 'TB': (20, 0.4, 'mg/dL'), 'DB': None,
        'Ca': (None, 1.0, 'mg/dL'), 'IP': (10, 0.3, 'mg/dL'), 'CRE': (10, 0.2, 'mg/dL'), 'UA': (10, None, ''),
        'CRP': None,                     # only high-sensitivity CRP is listed (493.927)
        'TC': (10, None, ''), 'HDL-C': (20, 6, 'mg/dL'), 'LDL-C': (20, None, ''),   # LDL: direct measurement
        'TG': (15, None, ''), 'Fe': (15, None, ''), 'IgG': (20, None, ''), 'KL-6': None,
        'IgA': (20, None, ''), 'IgM': (20, None, ''), 'GLU': (8, 6, 'mg/dL'), 'Zn': None, 'Cu': None}


def jsonl_recs():
    out = {}
    for g in ('main', 'ca', 'm2'):
        for l in open(os.path.join(V2_SRC, g, 'Reports', 'comparison.jsonl'), encoding='utf-8'):
            if l.strip():
                d = json.loads(l)
                out[(d['item'], d['method'], d['N'])] = d
    # the two premature-termination levels (q = 0.5, 0.2) of the six immunoturbidimetric/control items were run
    # separately (20260917_Rev_v2Benchmark_prozone); add their recall keys to the main records
    for l in open(PROZONE, encoding='utf-8'):
        if l.strip():
            d = json.loads(l)
            r = out.get((d['item'], d['method'], d['N']))
            if r is not None:
                for q in Q_PROZONE:
                    key = f'recall_S_q{q:g}'
                    if key in d['budgets']['99.5']:
                        r['budgets']['99.5'][key] = d['budgets']['99.5'][key]
    return out


def bench_median(item):
    df = pd.read_parquet(PROC, columns=['S_Date', 'Value'], filters=[('A_Name', '==', item)])
    v = df.sort_values('S_Date', kind='mergesort')['Value'].iloc[:N_BENCH]
    return float(np.nanmedian(v)), int(v.notna().sum())


def clia_table(items):
    rows = []
    for it in items:
        c = CLIA.get(it)
        if c is None:
            rows.append(dict(item=it, clia_rule='not specified', median_value=np.nan, n_values=0, CLIA_pct=np.nan))
            continue
        pct, ab, unit = c
        med, n = bench_median(it)
        ab_pct = 100 * ab / med if ab is not None and med > 0 else np.nan
        rule = ' or '.join(s for s in (f'±{pct}%' if pct else '', f'±{ab} {unit}' if ab else '') if s)
        rows.append(dict(item=it, clia_rule=rule + (' (greater)' if pct and ab else ''),
                         median_value=med, n_values=n, abs_limit_pct_at_median=ab_pct,
                         percent_component=pct if pct else np.nan,
                         # the bias is a change relative to the median value, so the limit that applies AT the median
                         # is used: the greater of the percent and the absolute part
                         CLIA_pct=np.nanmax([pct if pct else np.nan, ab_pct])))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    bat = os.path.join(OUT, 'yardstick_inputs')
    os.makedirs(bat, exist_ok=True)
    shutil.copyfile(SEL, os.path.join(OUT, 'dae_z_selection.csv'))
    # bias: s2a2's two-point endpoint computation (photometric points, liquid-volume correction, empirical
    # calibration, full k grid) replaces s2a's bias_summary (k <= 0.5 only; approximations).  s2d reads 'median'
    # as tag step_incl and 'react_median' as react_only; combine() relabels them (BIAS_DEF).
    bs = pd.read_csv(BIAS_SRC)
    bs[~bs['item'].isin(NEAR_ZERO)].to_csv(os.path.join(bat, 'bias_summary.csv'), index=False)
    D.NONADD = dict(D.NONADD, S_saturation=D.NONADD['S_saturation'] + Q_PROZONE)   # s2d skips q absent for an item
    aps = pd.read_csv(os.path.join(BAT_SRC, 'eflm_aps_panel21.csv'))
    clia = clia_table(aps['item'].tolist())
    clia.to_csv(os.path.join(OUT, 'clia_limits_panel21.csv'), index=False)
    aps.merge(clia[['item', 'CLIA_pct']], on='item', how='left').to_csv(
        os.path.join(bat, 'eflm_aps_panel21.csv'), index=False)

    D.load_recs = jsonl_recs
    D.BAT, D.METHODS = bat, METHODS13
    # one folder per yardstick: s2d names its files only '' (TEa) / '_desirablebias', and renaming over an
    # existing file on the Drive mount left stale tails in an earlier run, so nothing is renamed or replaced
    for ys in YARDSTICKS:
        sub = os.path.join(OUT, ys)
        os.makedirs(sub, exist_ok=True)
        shutil.copyfile(SEL, os.path.join(sub, 'dae_z_selection.csv'))
        print(f'\n######## yardstick={ys}', flush=True)
        D.V2, D.YARDSTICK = sub, ys
        D.main()
    k = pd.read_csv(os.path.join(OUT, YARDSTICKS[0], 'k50_by_item.csv'))
    s = k.groupby('method')['k50_pooled'].agg(['mean', 'max', lambda v: int(v.isna().sum())])
    s.columns = ['k50_pooled_mean', 'k50_pooled_worst', 'n_items_never_50pct']
    cells = k.melt(id_vars=['item', 'method'], value_vars=[f'k50_{t}' for t in D.TYPES], value_name='k50')
    cens = cells.groupby('method')['k50'].agg(
        n_cells='size', n_cells_censored=lambda v: int(v.isna().sum()),
        k50_cell_median=lambda v: float(np.median(v.fillna(np.inf))))      # inf = never within k <= 8
    s = s.join(cens)
    s.reindex(METHODS13).to_csv(os.path.join(OUT, 'k50_summary_13.csv'))
    print(s.reindex(METHODS13).round(3).to_string())
    print(clia.to_string(index=False))
    combine(aps, clia)


def combine(aps, clia):
    """Supplement-ready tables: (a) the three limits per item; (b) per yardstick x anomaly type x bias definition,
    the number of items whose approximate bias reaches the limit within the tested grid, and each method's mean
    recall at that severity (the same quantities as s2d's clinical_summary, in one long table)."""
    lim = aps[['item', 'desirable_bias_pct', 'TEa_pct']].merge(
        clia[['item', 'clia_rule', 'median_value', 'CLIA_pct', 'percent_component']], on='item', how='left')
    lim.loc[lim['item'].isin(NEAR_ZERO), ['desirable_bias_pct', 'TEa_pct', 'CLIA_pct']] = np.nan   # not compared in %
    lim['percent_comparison'] = (~lim['item'].isin(NEAR_ZERO)
                                 & lim[['desirable_bias_pct', 'TEa_pct', 'CLIA_pct']].notna().any(axis=1))   # KL-6: none
    lim.to_csv(os.path.join(OUT, 'm2_limits_three_yardsticks.csv'), index=False)
    rows = []
    for ys, sub, f in (('EFLM desirable bias', 'desirable_bias_pct', 'clinical_severity_desirablebias.csv'),
                       ('EFLM desirable TEa', 'TEa_pct', 'clinical_severity.csv'),
                       ('CLIA (reference)', 'CLIA_pct', 'clinical_severity_desirablebias.csv')):
        cl = pd.read_csv(os.path.join(OUT, sub, f))
        assert (cl['yardstick'] == sub).all() and cl.groupby(['anomaly_type', 'bias_def']).size().max() <= 21
        for (t, b), g in cl.groupby(['anomaly_type', 'bias_def']):
            hit = g.dropna(subset=['severity_at_TEa'])
            r = dict(yardstick=ys, anomaly_type=t, bias_def=BIAS_DEF[b], n_items_evaluated=int(len(g)), n_items_reached=int(len(hit)))
            for m in METHODS13:
                r[m] = hit[m].mean() if len(hit) else np.nan
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'm2_recall_at_limit_long.csv'), index=False)


if __name__ == '__main__':
    main()
