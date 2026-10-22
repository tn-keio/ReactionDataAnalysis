# Stage 1w: S11.2 monthly flag rates at BOTH training sizes.
#
# s1g re-ran Table S3 without np.clip, but the wrapped measure_review_burden_unified.py only writes the
# monthly series for N=500.  S11.2 also quotes N=5,000 monthly values (IgM 2023-06..11, IgA 2024-04)
# and per-item N=500/N=5,000 ratios.  This script reuses B.load / B.rolling_flags / B.volumes unchanged
# and only swaps B.apply_norm, writing per-item rates and monthly rates for N=500 and N=5,000.
#   --clip off : all 21 items (revision values)
#   --clip on  : repro check against the submitted S11.2 text (default items IgM, IgA)
# Output: Results/20261013_Rev_M1_BurdenMonthly/{clip,noclip}/{rates_by_item.csv, monthly_by_item_N.csv}
# py3.14 (no TensorFlow).

import os
import sys
import argparse
import gc
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
FORPAPER = os.path.join(BASE, 'ForPaper')
for p in (os.path.join(FORPAPER, 'Suppl10_ReviewBurden'), FORPAPER):
    if p not in sys.path:
        sys.path.insert(0, p)
import measure_review_burden_unified as B     # noqa: E402
from _common import panel as P                # noqa: E402

OUT = os.path.join(BASE, 'Results', '20261013_Rev_M1_BurdenMonthly')
ORIG_APPLY_NORM = B.apply_norm


def apply_norm_noclip(a, lo, hi):
    return (a - lo) / (hi - lo + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', choices=['on', 'off'], required=True)
    ap.add_argument('--items', default='', help='comma-separated subset (default: all 21 for off, IgM,IgA for on)')
    args = ap.parse_args()
    B.apply_norm = ORIG_APPLY_NORM if args.clip == 'on' else apply_norm_noclip
    items = [proc for proc, _ in P.PANEL21]
    if args.items:
        items = args.items.split(',')
    elif args.clip == 'on':
        items = ['IgM', 'IgA']
    out = os.path.join(OUT, 'clip' if args.clip == 'on' else 'noclip')
    os.makedirs(out, exist_ok=True)
    vol = B.volumes()
    rows, monthly = [], []
    for item in items:
        dates, calc = B.load(item)
        v = vol[item.upper()]
        rec = dict(item=item, n_curves=len(calc), tests_per_calendar_day=v['per_day'])
        for N in B.N_GRID:
            f = B.rolling_flags(calc, N)
            ok = ~np.isnan(f)
            rec[f'rate_N{N}_pct'] = 100 * float(f[ok].mean())
            rec[f'flags_per_day_N{N}'] = rec[f'rate_N{N}_pct'] / 100 * v['per_day']
            df = pd.DataFrame({'month': [str(m) for m in np.asarray(dates.to_period('M'))[ok]], 'flag': f[ok]})
            g = df.groupby('month').agg(n=('flag', 'size'), rate_pct=('flag', lambda x: 100 * x.mean())).reset_index()
            g = g[g['n'] >= B.MIN_MONTH_N]
            g['item'] = item; g['N'] = N
            monthly.append(g)
        rows.append(rec)
        print(f"{item:6} N500 {rec['rate_N500_pct']:.2f}%  N5000 {rec['rate_N5000_pct']:.2f}%", flush=True)
        del dates, calc; gc.collect()
    pd.DataFrame(rows).to_csv(os.path.join(out, 'rates_by_item.csv'), index=False, float_format='%.4f')
    pd.concat(monthly).to_csv(os.path.join(out, 'monthly_by_item_N.csv'), index=False, float_format='%.4f')
    print('->', out)


if __name__ == '__main__':
    main()
