# Per-analyzer, lot-aligned comparison of the reaction-curve signal against the saline-blank QC.
#
# WHY THIS SUPERSEDES Rev11/Rev12.  Both earlier analyses pooled the three analyzers. Reagent lots
# are not changed on all three on the same day -- for IgM the defective lot 000383 reached #2 on
# 11 May 2023, #3 in mid-May, and #1 on 1 June 2023 -- so pooling smears three differently-timed
# onsets into one blurred ramp. That flatters neither signal in particular, but it makes any
# lead-time estimate unreliable, because the pooled onset date is an average of three events rather
# than a measurement of one. Each analyzer must be treated as its own experiment and anchored on its
# own lot-change date.
#
# For each (assay, analyzer) this script produces one weekly table containing:
#   * the reaction-curve flag rate from a fixed-reference model fitted on THAT analyzer's earliest
#     500 curves (so the reference is internal to the analyzer, not borrowed from another),
#   * the saline-blank median from the same analyzer,
#   * the reagent lot in force,
# and reports the first week each signal crosses its pre-specified alert, expressed as days from
# that analyzer's own switch to the implicated lot.
#
# per_analyzer_verdict.csv records the FIRST week each signal crosses its alert. The tables in
# Supplementary S14.5-S14.6 and Fig. S12 apply a two-consecutive-week persistence rule to the weekly table
# written here (Scripts/analysis_full/revision/s1z14_tables_s14_5_6.py); for IgG the two rules give
# different weeks. In the revision this module is run with unclipped normalization through
# Scripts/analysis_full/revision/s1g_burden_fixedref_noclip.py (arm rev13) and s1z11_igg_lot406.py.
# The saline measurements it reads (Results/20260726_SalineBlank_QC/saline_measurements.csv) are
# extracted from the raw instrument exports, which are not published.
#
# py3.14 (pandas + sklearn, no TF).

import os
import sys
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FORPAPER = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(FORPAPER, 'Suppl10_ReviewBurden'))
sys.path.insert(0, FORPAPER)

import measure_review_burden_unified as B   # noqa: E402
from _common import panel as P              # noqa: E402

import pyarrow.dataset as ds
import pyarrow.compute as pc

CURVE_ALERT = 20.0      # % of curves flagged
N_TRAIN = 500
MIN_WEEK_N = 5

# assay -> (implicated lot, display window)
EVENTS = {
    'IgM': ('000383', '2023-03-01', '2023-12-31'),
    'IgG': (None,     '2024-08-01', '2025-06-30'),   # lot determined from the data
}


def load_curves(a_name, analyzer):
    """Date-sorted calc curves for one assay on one analyzer."""
    d = ds.dataset(os.path.join(P.BASE, 'Data', 'BigData', 'processed'), partitioning='hive')
    tb = d.to_table(columns=['S_Date', 'SystemName'] + B.RAM + B.RAS,
                    filter=pc.field('A_Name') == a_name).to_pandas()
    tb['sys'] = tb['SystemName'].astype(str).str.replace(
        'LABOSPECT 008 ALPHA ', '', regex=False).str.strip()
    tb = tb[tb['sys'] == analyzer]
    if tb.empty:
        return None, None
    tb['S_Date'] = pd.to_datetime(tb['S_Date'])
    tb = tb.sort_values('S_Date').reset_index(drop=True)
    calc = tb[B.RAM].values.astype('float64') - tb[B.RAS].values.astype('float64')
    return pd.to_datetime(tb['S_Date'].values), calc


def fixed_reference(dates, calc, n_train):
    train = calc[:n_train]
    lo, hi = B.norm_fit(train)
    W = B.apply_norm(train, lo, hi)
    e = B.EmpiricalCovariance().fit(W)
    thr = np.percentile(B.maha(e.location_, e.covariance_, W), B.BUDGET)
    d = B.maha(e.location_, e.covariance_, B.apply_norm(calc[n_train:], lo, hi))
    return dates[n_train:], d >= thr


def saline(item):
    src = os.path.join(P.RESULTS, '20260726_SalineBlank_QC', 'saline_measurements.csv')
    df = pd.read_csv(src, dtype=str)
    df = df[df['a_name'] == item]
    df['dt'] = pd.to_datetime(df['a_date'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['dt', 'value'])
    return df[~df['s_id'].str.contains('Cal', na=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=list(EVENTS))
    args = ap.parse_args()
    out = P.out_dir('PerAnalyzer_LotAligned')
    rows, verdicts = [], []

    for item in args.items:
        lot_of_interest, lo, hi = EVENTS[item]
        sal = saline(item)
        # saline baseline/alert from the assay's first 180 days, pooled (a property of the assay)
        ref = sal[sal['dt'] <= sal['dt'].min() + pd.Timedelta(days=180)]['value']
        base = ref.median()
        mad = (ref - base).abs().median()
        alert = max(base + 5 * (mad if mad > 0 else (ref.std() or 0.1)), base + 1.0)

        for analyzer in ['#1', '#2', '#3']:
            s = sal[sal['analyzer'] == analyzer].copy()
            if s.empty:
                continue
            s['wk'] = s['dt'].dt.to_period('W').dt.start_time

            # per-analyzer lot switch dates
            s2 = s.sort_values('dt')
            s2 = s2[s2['r1_lot'].notna() & (s2['r1_lot'] != '')]
            chg = s2[s2['r1_lot'] != s2['r1_lot'].shift()][['dt', 'r1_lot']]
            lot = lot_of_interest
            if lot is None:      # pick the lot in force at the peak saline week
                pk = s[(s['dt'] >= lo) & (s['dt'] <= hi)]
                if not pk.empty:
                    wk = pk.groupby('wk')['value'].median().idxmax()
                    cand = pk[pk['wk'] == wk]['r1_lot'].dropna()
                    lot = cand.mode().iloc[0] if len(cand) else None
            sw = chg[chg['r1_lot'] == lot]['dt'].min() if lot else pd.NaT

            dates, calc = load_curves(item, analyzer)
            if dates is None or len(calc) < 3 * N_TRAIN:
                print(f'  {item} {analyzer}: too few curves'); continue
            d2, flag = fixed_reference(dates, calc, N_TRAIN)
            cw = (pd.DataFrame({'dt': d2, 'flag': flag})
                  .assign(wk=lambda x: x['dt'].dt.to_period('W').dt.start_time)
                  .groupby('wk')['flag'].agg(['count', 'mean']))
            cw = cw[cw['count'] >= MIN_WEEK_N]
            cw['curve_pct'] = 100 * cw['mean']

            sw_ = s.groupby('wk')['value'].agg(['count', 'median', 'max'])
            sw_ = sw_[sw_['count'] >= 3]

            m = cw[['count', 'curve_pct']].rename(columns={'count': 'curve_n'}).join(
                sw_.rename(columns={'count': 'sal_n', 'median': 'sal_med', 'max': 'sal_max'}),
                how='outer').sort_index()
            m = m.loc[lo:hi]

            ch = m[m['curve_pct'] >= CURVE_ALERT].index.min() if (m['curve_pct'] >= CURVE_ALERT).any() else pd.NaT
            sh = m[m['sal_med'] >= alert].index.min() if (m['sal_med'] >= alert).any() else pd.NaT

            print(f'\n{"="*92}\n{item}  analyzer {analyzer}   lot {lot}  switched {sw:%Y-%m-%d}'
                  f'   saline alert {alert:.2f}\n{"="*92}')
            print(f'{"week":12} {"curve n":>8} {"flag %":>8} | {"sal n":>6} {"med":>6} {"max":>6}  lot')
            for wk, r in m.iterrows():
                lots = s[(s['wk'] == wk)]['r1_lot'].dropna()
                lt = ','.join(sorted({x[-4:] for x in lots.unique() if x})) or '-'
                f1 = '' if pd.isna(r['curve_pct']) else f"{r['curve_pct']:8.1f}"
                f2 = '' if pd.isna(r['sal_med']) else f"{r['sal_med']:6.2f}"
                f3 = '' if pd.isna(r['sal_max']) else f"{r['sal_max']:6.2f}"
                n1 = '' if pd.isna(r['curve_n']) else f"{int(r['curve_n']):8d}"
                n2 = '' if pd.isna(r['sal_n']) else f"{int(r['sal_n']):6d}"
                mk = ('  <=curve' if wk == ch else '') + ('  <=saline' if wk == sh else '')
                print(f'{wk:%Y-%m-%d}  {n1} {f1} | {n2} {f2} {f3}  {lt}{mk}')
                rows.append(dict(item=item, analyzer=analyzer, week=f'{wk:%Y-%m-%d}',
                                 lot_in_force=lt, curve_n=r['curve_n'], curve_pct=r['curve_pct'],
                                 sal_n=r['sal_n'], sal_med=r['sal_med'], sal_max=r['sal_max']))

            v = dict(item=item, analyzer=analyzer, lot=lot,
                     lot_switch=f'{sw:%Y-%m-%d}' if pd.notna(sw) else '',
                     curve_alert=f'{ch:%Y-%m-%d}' if pd.notna(ch) else '',
                     saline_alert=f'{sh:%Y-%m-%d}' if pd.notna(sh) else '',
                     curve_days_after_lot=(ch - sw).days if pd.notna(ch) and pd.notna(sw) else None,
                     saline_days_after_lot=(sh - sw).days if pd.notna(sh) and pd.notna(sw) else None,
                     lead_days=(sh - ch).days if pd.notna(ch) and pd.notna(sh) else None)
            verdicts.append(v)
            print(f"\n  lot switch {v['lot_switch']} | curve alert {v['curve_alert']} "
                  f"(+{v['curve_days_after_lot']}d) | saline alert {v['saline_alert']} "
                  f"(+{v['saline_days_after_lot']}d) | lead {v['lead_days']}d")

    pd.DataFrame(rows).to_csv(os.path.join(out, 'per_analyzer_weekly.csv'), index=False)
    vd = pd.DataFrame(verdicts)
    vd.to_csv(os.path.join(out, 'per_analyzer_verdict.csv'), index=False)
    print('\n\n=== SUMMARY (days from that analyzer\'s own lot switch) ===')
    print(vd.to_string(index=False))


if __name__ == '__main__':
    main()
