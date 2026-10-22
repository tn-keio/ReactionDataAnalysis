# Stage 1l: are the unexplained fixed-reference clusters (Ca, ALB, LDL-C) a stale anchor, or a real
# change that the control material does not show?
#
# s1k found that a fixed reference anchored on each assay's earliest 500 curves flags >=20% of curves
# in 27 months for Ca, 20 for LDL-C and 12 for ALB with no documented event, while 16 of 21 assays have
# 0-1 such months.  Before calling those false alerts we check the two external series we already have:
#   * the saline-blank control for the same assay (Results/20260726_SalineBlank_QC/saline_measurements.csv)
#   * the reagent-lot change dates (Data/20210301-20251231_reagent-comments.csv)
# and we ask whether the curve-flag rise (a) starts at a lot change, (b) coincides with a blank shift,
# or (c) is a slow monotone drift invisible to the blank - which recalibration would absorb into the
# reported result, and which therefore argues for periodic re-anchoring rather than for an alert.
# Output: Results/20260918_Rev_FixedReference_Panel21/drift_vs_qc_{item}.csv + printed summary.
# py3.14 (no TensorFlow).

import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(BASE, 'Results', '20260918_Rev_FixedReference_Panel21')
SALINE = os.path.join(BASE, 'Results', '20260726_SalineBlank_QC', 'saline_measurements.csv')
REAGENT = os.path.join(BASE, 'Data', '20210301-20251231_reagent-comments.csv')
ABBR = {'Ca': 'CA', 'ALB': 'ALB', 'LDL-C': 'LDL-C', 'IP': 'IP', 'Fe': 'FE', 'IgM': 'IGM', 'IgG': 'IGG'}


def monthly_blank(item):
    s = pd.read_csv(SALINE, dtype=str)
    s = s[s['a_name'] == item]
    if s.empty:
        return None
    s['d'] = pd.to_datetime(s['a_date'], errors='coerce')
    s['v'] = pd.to_numeric(s['value'], errors='coerce')
    s = s.dropna(subset=['d', 'v'])
    g = s.groupby(s['d'].dt.to_period('M')).agg(blank_n=('v', 'size'), blank_median=('v', 'median'),
                                                blank_p90=('v', lambda x: x.quantile(0.9))).reset_index()
    g['month'] = g['d'].astype(str)
    return g[['month', 'blank_n', 'blank_median', 'blank_p90']]


def lot_months(item):
    rg = pd.read_csv(REAGENT, encoding='utf-8-sig')
    rg['d'] = pd.to_datetime(rg['管理日'], errors='coerce')
    ab = ABBR.get(item, item)
    rg = rg[rg['abbr'].astype(str).str.upper() == ab.upper()]
    islot = rg['コメント'].astype(str).str.contains('Lot|New', case=False, regex=True)
    rg = rg[islot].dropna(subset=['d'])
    return sorted({str(p) for p in rg['d'].dt.to_period('M')})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=['Ca', 'ALB', 'LDL-C', 'IgM'])
    a = ap.parse_args()
    for item in a.items:
        f = os.path.join(OUT, f'monthly_{item}.csv')
        if not os.path.exists(f):
            print(f'{item}: no monthly file'); continue
        m = pd.read_csv(f)
        b = monthly_blank(item)
        if b is not None:
            m = m.merge(b, on='month', how='left')
        lots = lot_months(item)
        m['lot_change_month'] = m['month'].isin(lots)
        m.to_csv(os.path.join(OUT, f'drift_vs_qc_{item}.csv'), index=False)
        hi = m[m.rate >= 0.20]
        print(f'\n=== {item}: {len(m)} months, {len(hi)} with >=20% flagged, lot-change months: {len(lots)}')
        if b is not None and m['blank_median'].notna().any():
            lo_m = m[m.rate < 0.20]['blank_median'].dropna()
            hi_m = hi['blank_median'].dropna()
            print(f'  saline blank median: months <20% flagged {lo_m.median():.3f} (n={len(lo_m)}) | '
                  f'months >=20% {hi_m.median():.3f} (n={len(hi_m)})')
            c = m[['rate', 'blank_median']].dropna()
            if len(c) > 5:
                print(f'  Spearman(flag rate, blank median) = {c.corr(method="spearman").iloc[0, 1]:.2f} over {len(c)} months')
        else:
            print('  no saline-blank series for this assay')
        first = hi.month.min() if len(hi) else None
        near = []
        if first is not None:
            f0 = pd.Period(first, freq='M')
            near = [l for l in lots if abs((pd.Period(l, freq='M') - f0).n) <= 1]
        print(f'  first month >=20%: {first or "-"}; lot changes within +/-1 month of it: {near}')
        # monotone trend test on the flag-rate series (Spearman vs time)
        mm = m.reset_index(drop=True)
        rho = pd.Series(range(len(mm))).corr(mm['rate'], method='spearman')
        print(f'  monotone trend over the record: Spearman(month index, flag rate) = {rho:.2f}')


if __name__ == '__main__':
    main()
