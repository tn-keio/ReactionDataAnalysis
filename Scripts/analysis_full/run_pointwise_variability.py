# METHOD-AGNOSTIC justification of the reaction-point window (start point).
#
# The Mahalanobis detection sweep (run_reaction_points_forpaper) shows start=5 is good FOR THAT method.
# But the choice of usable data points should hold regardless of which detector is later used. This
# script justifies the window using only descriptive, model-free statistics of the curves: it shows
# that each item's point-wise VARIABILITY is large at points 1-4 (air-bubble instability during the
# initial ultrasonic mixing) and settles to a low, stable floor from ~point 5 -- no anomaly-detection
# algorithm involved.
#
# PRIMARY metric (the one that drives the figure and the recommended start point):
#   within-curve local roughness |2nd difference| = |x[i-1]-2x[i]+x[i+1]|, taken at a HIGH percentile
#   across samples and normalized by the item's endpoint IQR. The 2nd difference removes level and
#   linear slope, so it does NOT react to smooth real signal (flat or sloped baseline, real kinetics)
#   and isolates the non-smooth jitter that air bubbles cause. Per-item recommended start point =
#   1 + (last point whose relative roughness exceeds SETTLE_K x the item's settled floor).
#
#   Percentile robustness: computed at p99, p99.5 and p99.9. p99.5 matches the P99.5 operating
#   threshold used by the detector, directly answering "could a <=1% delayed-bubble tail hide below the
#   descriptive statistic but be flagged at P99.5?". (Answer, empirically: the conclusion is invariant
#   p99 -> p99.5.) NOTE: roughness catches JITTERY instability; a purely SMOOTH offset in a small
#   fraction of curves would not raise it (that residual is left to the detector; see report S5.3).
#
# AUXILIARY (provenance only, NOT used for the decision): across-sample detrended dispersion
#   (robust SD of residual after subtracting each curve's settled baseline). It is CONFOUNDED by real
#   early kinetics (e.g. LDL-C's early hump inflates it), so it is saved as a column but no settling
#   point is derived from it. No TensorFlow.

import os, sys, glob, json
import numpy as np, pandas as pd
import pyarrow.dataset as ds, pyarrow.compute as pc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
PARAM = os.path.join(BASE, 'Data', 'parameter_mapped.csv')
OUT = os.path.join(BASE, 'Results', '20260716_ReactionPoints_Usage_ForPaper')

ALLM = [f'RAM{i}' for i in range(1, 39)]
ALLS = [f'RAS{i}' for i in range(1, 39)]
NMAX = 10000
FLOOR_PTS = list(range(11, 19))     # 0-indexed points 12..19 -> stable baseline reference
BASE_PTS = list(range(9, 19))       # 0-indexed points 10..19 -> per-curve baseline
SETTLE_K = 2.0                      # settled = within K x floor


def forpaper_items():
    parts = {os.path.basename(p).replace('A_Name=', '') for p in glob.glob(os.path.join(PROC, 'A_Name=*'))}
    low = {p.lower(): p for p in parts}
    fp = pd.read_csv(PARAM); fp = fp[fp['ForPaper'] == 1]['ItemName'].tolist()
    return sorted(low[x.lower()] for x in fp)


def load_calc(item):
    d = ds.dataset(PROC, partitioning='hive')
    t = d.to_table(columns=['S_Date'] + ALLM + ALLS, filter=pc.field('A_Name') == item).to_pandas()
    t = t.sort_values('S_Date').reset_index(drop=True)
    if len(t) > NMAX:
        s = len(t) // 2 - NMAX // 2; t = t.iloc[s:s + NMAX]
    return t[ALLM].values.astype('float64') - t[ALLS].values.astype('float64')


def rsd(x, axis=0):
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def main():
    os.makedirs(OUT, exist_ok=True)
    items = forpaper_items()
    print(f'ForPaper items ({len(items)})', flush=True)
    PCTLS = [('99', 99.0), ('995', 99.5), ('999', 99.9)]     # p99.5 matches the P99.5 operating point
    rows, starts = [], []
    for it in items:
        cc = load_calc(it)
        scale = np.percentile(cc[:, -1], 75) - np.percentile(cc[:, -1], 25) + 1e-9   # endpoint IQR
        # PRIMARY: within-curve roughness |2nd diff| at several high percentiles (pts 2..37)
        d2 = np.abs(cc[:, :-2] - 2 * cc[:, 1:-1] + cc[:, 2:])
        rough_p50 = np.median(d2, 0) / scale * 100
        rough = {lab: np.percentile(d2, q, 0) / scale * 100 for lab, q in PCTLS}
        # AUXILIARY (provenance only, confounded by real kinetics -> not used for the decision)
        disp = rsd(cc - np.median(cc[:, BASE_PTS], axis=1, keepdims=True), axis=0) / scale * 100
        for p in range(1, 39):
            r = {'item': it, 'point': p, 'disp_detrended_aux': disp[p - 1]}
            if 2 <= p <= 37:
                r['rough_p50'] = rough_p50[p - 2]
                for lab, _ in PCTLS:
                    r[f'rough_p{lab}'] = rough[lab][p - 2]
            rows.append(r)
        # recommended start from the roughness metric, at each percentile (single source of truth)
        rec = {'item': it}
        for lab, _ in PCTLS:
            s = pd.Series(rough[lab], index=range(2, 38))
            floor = np.median(s.loc[12:19]); rel = s / floor
            exc = rel.loc[2:18][rel.loc[2:18] > SETTLE_K]
            rec[f'start_p{lab}'] = (int(exc.index.max()) if len(exc) else 1) + 1
            if lab == '99':
                rec['floor_pct_endpointIQR'] = round(float(floor), 2)
        starts.append(rec)
        print(f"  {it:6s} start (p99/p99.5/p99.9) = {rec['start_p99']}/{rec['start_p995']}/{rec['start_p999']}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'pointwise_variability.csv'), index=False)
    sdf = pd.DataFrame(starts).rename(columns={'start_p99': 'recommended_start'})
    sdf = sdf[['item', 'floor_pct_endpointIQR', 'recommended_start', 'start_p995', 'start_p999']] \
        .sort_values('recommended_start')
    sdf.to_csv(os.path.join(OUT, 'recommended_start_by_item.csv'), index=False)
    json.dump({'items': items, 'n_max': NMAX, 'primary_metric': '|2nd diff| roughness relative to floor',
               'percentiles': [q for _, q in PCTLS], 'settle_rule': f'1 + last point with rel roughness > {SETTLE_K}x floor(pts12-19)',
               'date': '2026-07-16'}, open(os.path.join(OUT, 'pointwise_variability_config.json'), 'w'), indent=2)

    print('\n===== recommended start (p99 primary) distribution =====')
    print(sdf['recommended_start'].value_counts().sort_index().to_string())
    print('\n===== percentile robustness (items with start > 5) =====')
    for col, name in [('recommended_start', 'p99'), ('start_p995', 'p99.5'), ('start_p999', 'p99.9')]:
        n_ok = int((sdf[col] <= 5).sum())
        print(f'  {name:6s}: {n_ok}/21 items settle by <=5 ; start>5: {sdf[sdf[col] > 5]["item"].tolist()}')
    print('\n-> ', OUT)


if __name__ == '__main__':
    main()
