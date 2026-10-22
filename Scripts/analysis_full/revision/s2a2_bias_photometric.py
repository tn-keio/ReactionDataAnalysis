# Stage 2a-2: the effect of each synthetic perturbation on
# the REPORTED result, computed the way the analyzer computes a two-point endpoint result, replacing s2a's two
# approximations (bias_approx: all pre-addition points vs the last 2; bias_react: the first post-addition point vs
# the last 2, which discards the reaction that happens at reagent-2 addition).
#   dA = mean(calc at the end points) - f * mean(calc at the base points)
#     base / end points : Data/20261020_PhotometricPoints.csv (the instrument settings; n points counted
#                         backwards from the given point, inclusive)
#     f                 : liquid-volume correction (S + R1) / (S + R1 + R2) from Data/parameter_mapped.csv, applied
#                         when the base points precede reagent-2 addition, i.e. base point <= R2_PRE_POINT (19, the
#                         last photometric point before reagent 2 on this analyzer); otherwise f = 1 (CRP, KL-6)
#     calc              : RAM - RAS (bichromatic), as everywhere in the study
#   Bias per perturbed curve, against the clean curve of the same base_id, as a concentration change relative to
#   the item's median reported value (a typical patient sample):  bias = b * (dA' - dA0) / C_med
#     b      : calibration slope (concentration per absorbance unit).  The intercept (reagent blank, -K * S1) cancels
#              in the concentration change, so it is not used; reconstructing each curve's concentration as a + b dA0
#              went to zero or below for low-valued curves.
#     b      : median of per-(analyzer, month) slopes of Value on dA (blocks of >= 50 curves, each a 4-MAD-trimmed
#              OLS), which absorbs reagent-lot and calibration changes; a single trimmed global slope is reported
#     C_med  : median reported value of the first 15,000 date-sorted curves (the benchmark set)
#   Summaries over the 200 base curves:
#     'median'       : median |bias| over all curves (what a typical perturbed curve does to the result)
#     'react_median' : median |bias| over the curves with a substantive change: additive types, |dd| at k = 1 >= 5% of
#                      the item's largest (the perturbation falls on a read point; fixed across k); non-additive
#                      types, |dd| >= 1% of the median reaction
#     signed_median, p05, p95, frac_curves_changed, abs_change_median (concentration units) are also written.
#   Additive types: dd = k * dA(ram_k1 - ram0) exactly (no subtraction of large numbers).
#   A relative bias is not meaningful where C_med is near zero (DB 0.08 mg/dL, CRP 0.13 mg/dL); they are flagged
#   (NEAR_ZERO) and s2d2 leaves them out of the percent comparison; abs_change_median is reported for them.
#   Severity grid: additive types (A, B, C, E) are linear in k (the battery scales one k = 1 perturbation; s2a
#   asserts this), so every k of s2d's grid 0.02-8 is built as ram0 + k * (ram_k1 - ram0); W_timewarp and
#   S_saturation are taken from s2a.build_item (unchanged).  Battery per item = the one Table 1 used
#   (Ca: the clean Ca battery; ALB, IP: the module-2 clean battery; others: main).
# Output: Results/20261020_Rev_M2_BiasPhotometric/{bias_summary_photometric.csv, calibration_fit.csv, points_used.csv}
# py3.14 (no TensorFlow).  Summary statistics only.
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2a_make_battery_v2 as A     # noqa: E402  (build_item, RAM, RAS, P0, SYN_DIR, SRC, POS)

BASE = A.BASE
OUT = os.path.join(BASE, 'Results', '20261020_Rev_M2_BiasPhotometric')
POINTS = os.path.join(BASE, 'Data', '20261020_PhotometricPoints.csv')
PARAM = os.path.join(BASE, 'Data', 'parameter_mapped.csv')
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
K_FULL = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 8]
N_BENCH = 15000
R2_PRE_POINT = 19                     # last photometric point before reagent-2 addition, all items
MIN_BLOCK_N = 50                      # curves per (analyzer, month) block for a slope
NEAR_ZERO = {'DB', 'CRP'}             # median value near zero (0.08, 0.13 mg/dL); see header
BATTERY = {'Ca': 'ca', 'ALB': 'm2', 'IP': 'm2'}          # else 'main'


def cols(point, n):
    """Indices into the 5-38 arrays of the n points ending at `point` (inclusive)."""
    pts = list(range(point - n + 1, point + 1))
    assert min(pts) >= A.P0 and max(pts) <= 38, f'points {pts} outside the 5-38 window'
    return [p - A.P0 for p in pts]


def delta_a(calc, sp):
    return calc[:, sp['end']].mean(1) - sp['f'] * calc[:, sp['base']].mean(1)


def settings():
    pts = pd.read_csv(POINTS)
    par = pd.read_csv(PARAM).set_index('ItemName')
    out, rows = {}, []
    for _, r in pts.iterrows():
        it, key = r['item'], r['item'].upper()
        g = BATTERY.get(it, 'main')
        s, r1, r2 = (float(par.loc[key, c]) for c in ('S.vol', 'R1.vol', 'R2.vol'))
        pre = int(r['base_point']) <= R2_PRE_POINT
        f = (s + r1) / (s + r1 + r2) if pre else 1.0
        out[it] = dict(base=cols(int(r['base_point']), int(r['base_n_points'])),
                       end=cols(int(r['end_point']), int(r['end_n_points'])), f=f, group=g, key=key)
        rows.append(dict(item=it, battery=g, base_points=f"{int(r['base_point']) - int(r['base_n_points']) + 1}-{int(r['base_point'])}",
                         end_points=f"{int(r['end_point']) - int(r['end_n_points']) + 1}-{int(r['end_point'])}",
                         base_before_R2=pre, S=s, R1=r1, R2=r2, f=round(f, 4)))
    return out, pd.DataFrame(rows)


def calibrate(it, sp):
    df = pd.read_parquet(PROC, columns=['SystemName', 'S_Date', 'Value'] + A.RAM + A.RAS,
                         filters=[('A_Name', '==', it)])
    df = df.sort_values('S_Date', kind='mergesort').iloc[:N_BENCH]
    df = df[np.isfinite(pd.to_numeric(df['Value'], errors='coerce'))]
    calc = df[A.RAM].values.astype('float64') - df[A.RAS].values.astype('float64')
    x, y = delta_a(calc, sp), df['Value'].values.astype('float64')
    b_global, a_global, r2_global, keep_global = trimmed_fit(x, y)
    blocks = pd.DataFrame({'x': x, 'y': y, 'sys': df['SystemName'].values,
                           'mo': pd.to_datetime(df['S_Date']).dt.to_period('M').astype(str).values})
    slopes, r2s = [], []
    for _, g in blocks.groupby(['sys', 'mo']):
        if len(g) >= MIN_BLOCK_N and g['x'].std() > 0:
            bb, _, rr, _ = trimmed_fit(g['x'].values, g['y'].values)
            slopes.append(bb); r2s.append(rr)
    return dict(item=it, n=int(len(x)), median_value=float(np.median(y)),
                slope_block_median=float(np.median(slopes)), slope_block_p25=float(np.percentile(slopes, 25)),
                slope_block_p75=float(np.percentile(slopes, 75)), n_blocks=len(slopes),
                r2_block_median=float(np.median(r2s)), slope_global=b_global, intercept_global=a_global,
                r2_global=r2_global, kept_global=float(keep_global.mean()), near_zero_median=it in NEAR_ZERO)


def trimmed_fit(x, y, k=4.0):
    """OLS of y on x after dropping |residual - median| > k * 1.4826 * MAD of a first OLS fit.  Without trimming a
    few off-line results (diluted re-runs, flagged values) dominated the slope (e.g. IP 0.0107 vs 0.0022, R^2 0.82);
    keeping the central bulk also gives the local slope around typical values for curved immunoassay calibrations."""
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    mad = np.median(np.abs(res - np.median(res)))
    keep = np.abs(res - np.median(res)) <= k * 1.4826 * mad if mad > 0 else np.ones(len(x), bool)
    b, a = np.polyfit(x[keep], y[keep], 1)
    r2 = 1 - np.sum((y[keep] - a - b * x[keep]) ** 2) / np.sum((y[keep] - y[keep].mean()) ** 2)
    return b, a, r2, keep


def main():
    os.makedirs(OUT, exist_ok=True)
    sp_all, pts_used = settings()
    pts_used.to_csv(os.path.join(OUT, 'points_used.csv'), index=False)
    print(pts_used.to_string(index=False))
    cal = pd.DataFrame([calibrate(it, sp) for it, sp in sp_all.items()]).set_index('item')
    cal.to_csv(os.path.join(OUT, 'calibration_fit.csv'))
    print(cal.round(4).to_string())
    batteries = {g: pd.read_parquet(os.path.join(A.SYN_DIR, fn)) for g, fn in A.SRC.items()}
    summ = []
    for it, sp in sp_all.items():
        df = batteries[sp['group']]
        df = df[df['item'] == sp['key']].reset_index(drop=True)
        clean = A.by_base(df, 'clean', 0.0)
        ram0, ras0 = clean[A.RAM].values, clean[A.RAS].values
        d0 = delta_a(ram0 - ras0, sp)
        b_loc, c_med = cal.loc[it, 'slope_block_median'], cal.loc[it, 'median_value']
        tol = 1e-2 * np.median(np.abs(d0))             # non-additive types: a change of >= 1% of the median reaction
        changes = []                                    # (type, k, dd per base curve, 'substantive change' mask)
        for t in A.POS:
            k1 = A.by_base(df, t, 1.0)
            assert (k1.base_id.values == clean.base_id.values).all()
            dd1 = delta_a(k1[A.RAM].values - ram0, sp)  # RAS is unchanged by the additive types: calc change = d1
            # spikes and humps have Gaussian tails, so a tiny tolerance counted near-misses as hits;
            # a hit is fixed at k = 1 as |dd1| >= 5% of the largest |dd1| of the item, the same for every k
            hit = np.abs(dd1) >= 0.05 * np.abs(dd1).max() if np.abs(dd1).max() > 0 else np.zeros(len(dd1), bool)
            for k in K_FULL:
                changes.append((t, k, k * dd1, hit))
        ext, _ = A.build_item(df)                       # W_timewarp, S_saturation exactly as in s2a
        for (t, k), g in ext[ext.anomaly_type.isin(['W_timewarp', 'S_saturation'])].groupby(['anomaly_type', 'severity']):
            g = g.set_index('base_id').loc[clean.base_id.values]
            dd = delta_a(g[A.RAM].values - g[A.RAS].values, sp) - d0
            changes.append((t, k, dd, np.abs(dd) > tol))
        for t, k, dd, ch in changes:
            bl = b_loc * dd / c_med                     # concentration change relative to the median value
            summ.append(dict(group='main', item=sp['key'], battery=sp['group'], anomaly_type=t, severity=float(k),
                             median=float(np.median(np.abs(bl))),
                             react_median=float(np.median(np.abs(bl[ch]))) if ch.any() else 0.0,
                             signed_median=float(np.median(bl)), p05=float(np.percentile(bl, 5)),
                             p95=float(np.percentile(bl, 95)), frac_curves_changed=float(ch.mean()),
                             abs_change_median=float(np.median(np.abs(b_loc * dd))),
                             abs_change_react_median=float(np.median(np.abs(b_loc * dd[ch]))) if ch.any() else 0.0,
                             near_zero_median=it in NEAR_ZERO))
    s = pd.DataFrame(summ)
    # s2d reads 'median' (tag step_incl) and 'react_median' (tag react_only); s2d2 relabels the tags (BIAS_DEF)
    s.to_csv(os.path.join(OUT, 'bias_summary_photometric.csv'), index=False)
    chk = s[(s.anomaly_type == 'B_drift')].pivot(index='item', columns='severity', values='median')
    print('\nbias (% of the median value), B_drift, block slope:\n', (100 * chk).round(2).to_string())


if __name__ == '__main__':
    main()
