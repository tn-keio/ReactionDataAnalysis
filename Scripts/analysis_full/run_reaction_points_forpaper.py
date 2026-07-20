# TASK 2 (paper edition) -- reaction photometric-point window, restricted to the ForPaper items only.
#
# The original 20260716_ReactionPoints_Usage used a spread that included GA/C3/IgG4, which are NOT in
# the ForPaper=1 panel. For the paper's Supplementary figure we re-run on the 21 ForPaper endpoint
# items so every curve shown belongs to the paper panel (and no item-selection bias remains).
#
# Same operationally-faithful methodology as run_reaction_points_usage (reused verbatim): per-point
# curvature p50/p95/p99 noise profile, and a window-start sweep measuring specificity + reaction-region
# and early-region synthetic recall (self-cal P99.5 Mahalanobis, rolling anchors, contemporaneous
# inlier-screened battery). No TensorFlow.

import os, sys, json, glob
import numpy as np, pandas as pd
import pyarrow.dataset as ds, pyarrow.compute as pc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# reuse the exact detection pipeline + battery from the original script
from run_reaction_points_usage import run_item, ALLM, ALLS, START_GRID, END, N_TRAIN, M_SCORE, METHODS

BASE = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
PARAM = os.path.join(BASE, 'Data', 'parameter_mapped.csv')
OUT = os.path.join(BASE, 'Results', '20260716_ReactionPoints_Usage_ForPaper')


def forpaper_items():
    """ForPaper=1 item names mapped (case-insensitively) to processed A_Name partitions."""
    parts = {os.path.basename(p).replace('A_Name=', '') for p in glob.glob(os.path.join(PROC, 'A_Name=*'))}
    low = {p.lower(): p for p in parts}
    fp = pd.read_csv(PARAM)
    fp = fp[fp['ForPaper'] == 1]['ItemName'].tolist()
    return sorted(low[x.lower()] for x in fp)


def noise_tails(items):
    d = ds.dataset(PROC, partitioning='hive')
    recs = []
    for it in items:
        tb = d.to_table(columns=['S_Date'] + ALLM + ALLS, filter=pc.field('A_Name') == it).to_pandas()
        if len(tb) == 0:
            continue
        tb = tb.sort_values('S_Date').reset_index(drop=True)
        if len(tb) > 8000:
            s = len(tb) // 2 - 4000; tb = tb.iloc[s:s + 8000]
        cc = tb[ALLM].values.astype('float64') - tb[ALLS].values.astype('float64')
        d2 = np.abs(cc[:, :-2] - 2 * cc[:, 1:-1] + cc[:, 2:])          # curvature at interior points 2..37
        scale = np.percentile(cc[:, -1], 75) - np.percentile(cc[:, -1], 25) + 1e-9
        p50 = np.median(d2, 0); p95 = np.percentile(d2, 95, 0); p99 = np.percentile(d2, 99, 0)
        for pt in range(2, 38):
            j = pt - 2
            recs.append({'item': it, 'point': pt, 'p50': 100 * p50[j] / scale,
                         'p95': 100 * p95[j] / scale, 'p99': 100 * p99[j] / scale})
        print(f'  noise {it:6s} done', flush=True)
    return pd.DataFrame(recs)


def main():
    os.makedirs(OUT, exist_ok=True)
    items = forpaper_items()
    print(f'ForPaper items ({len(items)}): {items}', flush=True)

    # Panel A source: per-point jitter p50/p95/p99 (all ForPaper items)
    tails = noise_tails(items)
    tails.to_csv(os.path.join(OUT, 'noise_profile_tails.csv'), index=False)

    # Panel B source: window-start detection sweep (all ForPaper items)
    all_rows, meta = [], {}
    for it in items:
        print(f'sweep {it} ...', flush=True)
        rows, na = run_item(it); all_rows += rows; meta[it] = na
        print(f'   anchors={na}', flush=True)
    series = pd.DataFrame(all_rows); series.to_csv(os.path.join(OUT, 'series.csv'), index=False)
    agg = (series.groupby(['item', 'method', 'start', 'n_points'])
           .agg(spec=('spec', 'mean'), recall_react=('recall_react', 'mean'),
                recall_early=('recall_early', 'mean'), n_anch=('spec', 'size'))
           .reset_index().round(4))
    agg.to_csv(os.path.join(OUT, 'summary_by_start.csv'), index=False)

    json.dump({'items': items, 'start_grid': START_GRID, 'end': END, 'N_train': N_TRAIN,
               'M_score': M_SCORE, 'anchors': meta, 'source': 'ForPaper=1 items only', 'date': '2026-07-16'},
              open(os.path.join(OUT, 'run_config.json'), 'w'), indent=2)

    pd.set_option('display.width', 220)
    print('\n===== spec / recall vs start (Empirical, mean over ForPaper items) =====')
    e = agg[agg.method == 'E'].groupby('start').agg(
        spec=('spec', 'mean'), recall_react=('recall_react', 'mean'), recall_early=('recall_early', 'mean')).round(3)
    print(e.to_string())
    print('\n-> ', OUT)


if __name__ == '__main__':
    main()
