# Stage 1z14 (S14.5, S14.6): the per-analyzer timing tables, written directly from the weekly series.
# per_analyzer_lot.py reports the FIRST week each indicator crosses its level, but S14.5 states the rule
# used for the tables and Fig. S12: an indicator alerts at the first week it reaches its level AND stays at
# or above it for two consecutive weeks (as make_figS12.py). Without persistence the IgG verdicts differ
# (curve alert on #1 two weeks earlier, a single noisy blank week on #2 eight weeks earlier), so the
# tables cannot be read off per_analyzer_verdict.csv. This script applies the stated rule to the unclipped
# weekly series and writes both tables.
# Inputs (unclipped): IgM from s1g rev13, IgG from s1z11 with the lot fixed to 000406 (the curve and blank
# series are identical to the data-chosen run; only the lot anchor differs); the saline blank levels are
# recomputed from the saline measurements exactly as in per_analyzer_lot.py.
# "Curve baseline" (S14.6) is the range of the weekly curve flag rate from the start of the display window
# to the week before the implicated lot first entered service on that analyzer, rounded to whole percent
# as in the table; the CSV also carries the 10th-90th percentile. The peak of
# an analyzer that never alerts is taken from the weeks on or after the implicated lot entered service.
# The saline level reproduces per_analyzer_lot.py line for line, so the two scripts cannot disagree.
# Output: Results/20261025_Rev_TablesS14_5_6/{table_s14_5.csv, table_s14_6.csv, tables_s14_5_6.md}
# py3.14 (no TensorFlow).
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(BASE, 'Results')
OUT = os.path.join(RES, '20261025_Rev_TablesS14_5_6')

WEEKLY = {'IgM': os.path.join(RES, '20260917_Rev_M1_S11S14_noclip', 'PerAnalyzer_LotAligned'),
          'IgG': os.path.join(RES, '20261020_Rev_IgG_Lot406', 'lot406')}
SALINE = os.path.join(RES, '20260726_SalineBlank_QC', 'saline_measurements.csv')

CURVE_ALERT = 20.0
PERSIST_WEEKS = 2
LOT = {'IgM': '0383', 'IgG': '0406'}
# display windows of Fig. S12 (b)-(g); the alert search runs inside them
WIN = {'IgM': ('2023-04-01', '2023-11-30'), 'IgG': ('2024-09-01', '2025-05-31')}
ANALYZERS = ['#1', '#2', '#3']


def saline_alert(item):
    """Baseline median + 5*MAD over the assay's first 180 days, floored at baseline + 1.0."""
    df = pd.read_csv(SALINE, dtype=str)
    df = df[df['a_name'] == item]
    df['dt'] = pd.to_datetime(df['a_date'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['dt', 'value'])
    df = df[~df['s_id'].str.contains('Cal', na=False)]
    ref = df[df['dt'] <= df['dt'].min() + pd.Timedelta(days=180)]['value']
    base = ref.median()
    mad = (ref - base).abs().median()
    return max(base + 5 * (mad if mad > 0 else (ref.std() or 0.1)), base + 1.0)


def sustained(weeks, values, level, k=PERSIST_WEEKS):
    """First week at or above `level` for k consecutive CALENDAR weeks (a missing week breaks the run)."""
    ok = np.asarray(values, float) >= level          # NaN compares False
    wk = weeks.reset_index(drop=True)
    for i in range(len(ok) - k + 1):
        if ok[i:i + k].all() and (wk[i + k - 1] - wk[i]).days == 7 * (k - 1):
            return wk[i]
    return None


def first(weeks, values, level):
    ok = np.asarray(values, float) >= level
    return weeks.iloc[int(ok.argmax())] if ok.any() else None


def ymd(t):
    return f'{t:%Y-%m-%d}' if t is not None and pd.notna(t) else ''


def rows_for(item):
    w = pd.read_csv(os.path.join(WEEKLY[item], 'per_analyzer_weekly.csv'), dtype={'lot_in_force': str})
    w['week'] = pd.to_datetime(w['week'])
    v = pd.read_csv(os.path.join(WEEKLY[item], 'per_analyzer_verdict.csv'), dtype={'lot': str})
    level = saline_alert(item)
    lo, hi = WIN[item]
    out = []
    for az in ANALYZERS:
        s = w[(w['item'] == item) & (w['analyzer'] == az)].sort_values('week')
        s = s[(s['week'] >= lo) & (s['week'] <= hi)]
        tc = sustained(s['week'], s['curve_pct'], CURVE_ALERT)
        ts = sustained(s['week'], s['sal_med'], level)
        on = s['lot_in_force'].fillna('').str.contains(LOT[item])
        if not on.any():
            raise SystemExit(f'{item} {az}: lot {LOT[item]} never in force inside {lo}..{hi}')
        lot_week = s.loc[on, 'week'].min()
        pre = s[s['week'] < lot_week]['curve_pct']
        if pre.empty:
            raise SystemExit(f'{item} {az}: no baseline weeks before {lot_week:%Y-%m-%d}')
        vv = v[v['analyzer'] == az].iloc[0]
        out.append(dict(
            item=item, analyzer=az, lot=vv['lot'], lot_switch=vv['lot_switch'],
            lot_first_week=ymd(lot_week), saline_level=round(level, 2),
            curve_alert=ymd(tc), saline_alert=ymd(ts),
            curve_lead_days=(ts - tc).days if tc is not None and ts is not None else None,
            curve_peak_pct=round(s[s['week'] >= lot_week]['curve_pct'].max(), 1),
            baseline_min=round(pre.min(), 1), baseline_p10=round(pre.quantile(.1), 1),
            baseline_p90=round(pre.quantile(.9), 1), baseline_max=round(pre.max(), 1), baseline_weeks=len(pre),
            # the first-crossing rule of per_analyzer_lot.py, for comparison
            curve_first_cross=ymd(first(s['week'], s['curve_pct'], CURVE_ALERT)),
            saline_first_cross=ymd(first(s['week'], s['sal_med'], level))))
    return pd.DataFrame(out)


def lead(r):
    if pd.notna(r['curve_lead_days']):
        return f"{int(r['curve_lead_days'])} d"
    if not r['curve_alert']:
        return 'not detected'                # the curve never alerted
    return 'blank never alerted'


def main():
    os.makedirs(OUT, exist_ok=True)
    t5, t6 = rows_for('IgM'), rows_for('IgG')
    t5.to_csv(os.path.join(OUT, 'table_s14_5.csv'), index=False)
    t6.to_csv(os.path.join(OUT, 'table_s14_6.csv'), index=False)

    md = ['## S14.5 (IgM, lot 000383)', '',
          '| Analyzer | Lot switch | Curve alert | Saline-blank alert | Curve lead |', '|---|---|---|---|---:|']
    md += [f"| {r.analyzer} | {r.lot_switch} | {r.curve_alert} | {r.saline_alert} | {lead(r)} |" for _, r in t5.iterrows()]
    md += ['', f"Saline-blank level {t5['saline_level'].iloc[0]:.2f}; curve level {CURVE_ALERT:.0f}%; "
           f"alert = first week at or above the level for {PERSIST_WEEKS} consecutive weeks.", '',
           '## S14.6 (IgG, lot 000406)', '',
           '| Analyzer | Curve baseline | Curve alert | Saline-blank alert | Curve lead |',
           '|---|---:|---|---|---:|']
    for _, r in t6.iterrows():
        ca = r.curve_alert or f"never - peak {r.curve_peak_pct:.1f}%"
        md.append(f"| {r.analyzer} | {r.baseline_min:.0f}-{r.baseline_max:.0f}% "
                  f"| {ca} | {r.saline_alert} | {lead(r)} |")
    md += ['', f"Saline-blank level {t6['saline_level'].iloc[0]:.2f}. Curve baseline: range of the weekly curve "
           f"flag rate from {WIN['IgG'][0]} to the week before lot 000406 first entered service on that analyzer "
           f"({', '.join(f'{n} weeks on {a}' for a, n in zip(t6['analyzer'], t6['baseline_weeks']))})."]
    with open(os.path.join(OUT, 'tables_s14_5_6.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    cols = ['item', 'analyzer', 'curve_alert', 'curve_first_cross', 'saline_alert', 'saline_first_cross']
    print('\nTwo-week rule against first crossing:')
    print(pd.concat([t5, t6])[cols].to_string(index=False))


if __name__ == '__main__':
    main()
