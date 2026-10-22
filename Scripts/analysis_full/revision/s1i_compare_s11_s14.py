# Stage 1i: submitted vs unclipped for the analyses built on measure_review_burden_unified.py.
#   S11 Table S3 : Results/20260726_ReviewBurden_Panel21  vs 20260917_Rev_M1_S11S14_noclip/ReviewBurden_Panel21
#   S14 / 3.12   : Results/20260726_FixedReference_IgEvent, _PerAnalyzer_LotAligned, _LotStratified
# py3.14.

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(BASE, 'Results')
NEW = os.path.join(RES, '20260917_Rev_M1_S11S14_noclip')
PAIRS = [('ReviewBurden_Panel21', 'review_burden_unified.csv', ['item']),
         ('FixedReference_IgEvent', 'fixed_reference_monthly.csv', ['item', 'month']),
         ('FixedReference_IgEvent', 'fixed_reference_weekly.csv', None),
         ('PerAnalyzer_LotAligned', 'per_analyzer_verdict.csv', None),
         ('LotStratified', 'lot_stratified_daily.csv', None)]


def main():
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    out = []
    for name, fn, key in PAIRS:
        a = pd.read_csv(os.path.join(RES, f'20260726_{name}', fn))
        b = pd.read_csv(os.path.join(NEW, name, fn))
        num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
        same_shape = a.shape == b.shape
        print(f'\n## {name}/{fn}  shape sub={a.shape} new={b.shape}')
        if same_shape and key is None:
            d = (b[num] - a[num]).abs().max()
            print('max |diff| per numeric column:\n', d[d > 0].to_string() if (d > 0).any() else '  none (identical)')
            nonnum = [c for c in a.columns if c not in num]
            ch = {c: int((a[c].astype(str) != b[c].astype(str)).sum()) for c in nonnum}
            print('changed non-numeric cells:', {k: v for k, v in ch.items() if v})
        elif key is not None:
            m = a.merge(b, on=key, suffixes=('_sub', '_new'), how='outer', indicator=True)
            print('merge:', m['_merge'].value_counts().to_dict())
            for c in num:
                if c in key:
                    continue
                dd = (m[f'{c}_new'] - m[f'{c}_sub']).abs()
                if dd.max() > 0:
                    print(f'  {c}: max|diff|={dd.max():.4f}  n_changed={(dd > 0).sum()}')
            if name == 'ReviewBurden_Panel21':
                cols = ['item', 'rate_N500_pct_sub', 'rate_N500_pct_new', 'flags_per_day_N500_sub',
                        'flags_per_day_N500_new', 'rate_N5000_pct_sub', 'rate_N5000_pct_new']
                t = m[cols].copy(); t['d_perday'] = t.flags_per_day_N500_new - t.flags_per_day_N500_sub
                print(t.round(3).to_string(index=False))
                print('TOTAL flags/day N500 sub=%.1f new=%.1f ; N5000 sub=%.1f new=%.1f' % (
                    m.flags_per_day_N500_sub.sum(), m.flags_per_day_N500_new.sum(),
                    m.flags_per_day_N5000_sub.sum(), m.flags_per_day_N5000_new.sum()))
                t.to_csv(os.path.join(NEW, name, 'compare_sub_vs_noclip.csv'), index=False)
            if name == 'FixedReference_IgEvent':
                for it in ('IgG', 'IgM'):
                    s = m[m.item == it]
                    print(f'  {it}: peak sub={s.rate_pct_sub.max():.1f}% new={s.rate_pct_new.max():.1f}% ; '
                          f'months>=20% sub={(s.rate_pct_sub >= 20).sum()} new={(s.rate_pct_new >= 20).sum()}')
        else:
            print('  shapes differ -> inspect manually')
        if name == 'PerAnalyzer_LotAligned':
            print(a.to_string(index=False)); print(b.to_string(index=False))


if __name__ == '__main__':
    main()
