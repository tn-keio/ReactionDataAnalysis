# Stage 1j: supplementary Table S5 — the per-method review
# budget under the deployment protocol — re-computed WITHOUT np.clip.
#
# ForPaper/Suppl10_ReviewBurden/compare_methods_unified.py (module "C") measures, for six classical
# detectors, the flag rate of the rolling recent-N model with a self-calibrated P99.5 threshold over
# the whole record.  Its apply_norm clips.  Nothing is modified here: C.apply_norm,
# C.PANEL and C.OUT are replaced at run time, exactly as ForPaper/Rev09_MethodBudget_Panel21 does.
#
# Acceptance check: the unclipped empirical-Mahalanobis flag rate must be close to the unclipped
# Table S3 value (1.28% of curves, 88.2 flagged curves/day; Results/20260917_Rev_M1_S11S14_noclip).
# The two protocols differ only in how blocks are sampled (S5 uses <=57 evenly spaced anchors), so a
# few per cent of relative difference is expected — the submitted versions differ by 1.29 vs 1.34%.
# Output: Results/20260918_Rev_M1_TableS5_noclip/  (the submitted run is left untouched)
# py3.14 (no TensorFlow).

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
FORPAPER = os.path.join(BASE, 'ForPaper')
for p in (os.path.join(FORPAPER, 'Suppl10_ReviewBurden'), FORPAPER):
    if p not in sys.path:
        sys.path.insert(0, p)
import compare_methods_unified as C      # noqa: E402
from _common import panel as P           # noqa: E402

OUT = os.path.join(BASE, 'Results', '20260918_Rev_M1_TableS5_noclip')


def apply_norm_noclip(a, lo, hi):
    return (a - lo) / (hi - lo + 1e-8)


def load_lean(a_name):
    """Same (dates, calc) as C.load, built column by column.

    C.load materialises the arrow table, a pandas copy and two more float64 matrices, which peaked at
    several GB for the largest assays and had the process killed.  Here the sorted row order is applied
    per column, so the peak is the final matrix plus one column.  Verified identical to C.load (--verify).
    """
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    d = ds.dataset(C.PROC, partitioning='hive')
    tb = d.to_table(columns=['S_Date'] + C.RAM + C.RAS, filter=pc.field('A_Name') == a_name)
    raw = pd.to_datetime(tb.column('S_Date').to_numpy(zero_copy_only=False)).values
    # pandas' sort_values defaults to kind='quicksort'; ties (same timestamp) must be ordered the same
    # way as in C.load, so use numpy's quicksort rather than arrow's stable sort.
    order = np.argsort(raw, kind='quicksort')
    dates = pd.to_datetime(raw[order])
    calc = np.empty((tb.num_rows, len(C.RAM)), dtype='float64')
    for j, (cm, cs) in enumerate(zip(C.RAM, C.RAS)):
        calc[:, j] = tb.column(cm).to_numpy(zero_copy_only=False)[order].astype('float64')
        calc[:, j] -= tb.column(cs).to_numpy(zero_copy_only=False)[order].astype('float64')
    del tb, order
    import gc
    gc.collect()
    return dates, calc


def verify_loader(item):
    """C.load vs load_lean on one item: rows, order and values must match exactly."""
    import numpy as np
    d1, c1 = C.load(item)
    d2, c2 = load_lean(item)
    same = (len(d1) == len(d2)) and bool((np.asarray(d1) == np.asarray(d2)).all()) and np.array_equal(c1, c2)
    print(f'loader check {item}: n={len(d1)} identical={same} max|diff|={np.abs(c1 - c2).max() if len(c1) else 0}')
    if not same:
        raise SystemExit('lean loader does not reproduce C.load')


def merge(root):
    """Concatenate the per-item tables and rebuild the panel summary (same formulas as C.main)."""
    files = sorted(glob.glob(os.path.join(root, 'per_item', '*', 'method_budget_unified.csv')))
    if not files:
        raise SystemExit('no per-item outputs to merge')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(root, 'method_budget_unified.csv'), index=False, float_format='%.4f')
    agg = df.groupby(['n_train', 'method']).agg(flags_per_day=('flags_per_day', 'sum'),
                                                n_scored=('n_scored', 'sum')).reset_index()
    tot = df.groupby('item')['tests_per_calendar_day'].first().sum()
    agg['rate_pct'] = 100 * agg['flags_per_day'] / tot
    agg['recall'] = [C.RECALL[n][m] for n, m in zip(agg['n_train'], agg['method'])]
    agg.to_csv(os.path.join(root, 'method_budget_summary.csv'), index=False, float_format='%.4f')
    print(f'merged {len(files)} items, {tot:,.0f} tests/day')
    for N in sorted(agg['n_train'].unique()):
        x = agg[agg['n_train'] == N].sort_values('flags_per_day', ascending=False)
        print(f'--- N={N} ---')
        for _, r in x.iterrows():
            print('  {:26s} {:8.1f} flags/day  {:6.3f}%'.format(C.LABEL[r['method']],
                                                                r['flags_per_day'], r['rate_pct']))
        lo, hi = x['flags_per_day'].min(), x['flags_per_day'].max()
        print('  spread {:.1f}-{:.1f} ({:.2f}x)'.format(lo, hi, hi / lo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', type=int, default=21, choices=[13, 21])
    ap.add_argument('--clip', choices=['on', 'off'], default='off')
    # one process per item keeps the peak memory to a single assay: the wrapped script loads the
    # whole record of each item, and running all 21 in one process was killed by the OS.
    ap.add_argument('--items', nargs='+', default=None)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--verify-loader', metavar='ITEM', default=None)
    ap.add_argument('--lean', action='store_true', help='use the memory-lean loader (needed for big assays)')
    a = ap.parse_args()
    root = OUT if a.clip == 'off' else OUT + '_clipcheck'
    if a.verify_loader:
        return verify_loader(a.verify_loader)
    if a.merge:
        return merge(root)
    if a.lean:
        C.load = load_lean
    if a.clip == 'off':
        C.apply_norm = apply_norm_noclip
    panel = [(proc, proc) for proc, _ in P.panel(a.panel)]       # this script keys the partition on A_Name
    C.PANEL = [x for x in panel if a.items is None or x[0] in a.items]
    if not C.PANEL:
        raise SystemExit(f'no panel item matches {a.items}')
    C.OUT = os.path.join(root, 'per_item', C.PANEL[0][0]) if a.items else root
    os.makedirs(C.OUT, exist_ok=True)
    print(f'Table S5 re-run | panel={a.panel} | items={[i for i, _ in C.PANEL]} | clip={a.clip}')
    print(f'out={C.OUT}', flush=True)
    C.main()
    print('\nACCEPTANCE CHECK: compare the empirical-Mahalanobis rate with the unclipped Table S3 '
          '(1.28% of curves) in Results/20260917_Rev_M1_S11S14_noclip/ReviewBurden_Panel21.')


if __name__ == '__main__':
    main()
