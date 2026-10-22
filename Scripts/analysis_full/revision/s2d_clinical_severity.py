# Stage 2d (Reviewer #2, Methods-2): tie detection to the reported-result bias.
#   (1) k50: the severity at which each method's recall crosses 50% (log-linear interpolation on the
#       k grid 0.02..8), per item and anomaly type, plus the Mahalanobis norm at that severity.
#   (2) clinically relevant severity: the smallest grid severity whose |bias| reaches the EFLM
#       desirable bias for that assay, and every method's recall there.  Bias is bracketed by the two
#       approximations (bias_approx = includes the reagent-addition step, bias_react = reaction phase only).
# Inputs: Results/20260917_Rev_v2Benchmark (per_item_v2.csv is not enough -> read the JSONL directly),
#         Results/20260917_Rev_M2_BatteryV2/{bias_summary.csv, eflm_desirable_bias_panel21.csv}
# Output: Results/20260917_Rev_v2Benchmark/{k50_by_item.csv, clinical_severity.csv, clinical_summary.csv}
# py3.14.

import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(BASE, 'Results')
V2 = os.path.join(RES, '20260917_Rev_v2Benchmark')
BAT = os.path.join(RES, '20260917_Rev_M2_BatteryV2')
K = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 8]
TYPES = {'A_spike': K, 'B_drift': K, 'C_hump': K, 'E_baseline': K}
NONADD = {'W_timewarp': [0.95, 0.9, 0.8, 0.7, 0.5], 'S_saturation': [0.98, 0.95, 0.9, 0.8]}
METHODS = ['maha', 'maha_robust', 'pca', 'lof', 'ae', 'twod_only', 'ae2d_ens', 'dae_elbow', 'dae_oracle']
N_MAIN = 500
YARDSTICK = os.environ.get('REV_YARDSTICK', 'TEa_pct')     # 'TEa_pct' (sporadic single-result error) or 'desirable_bias_pct'


def load_recs():
    out = {}
    for g in ('main', 'ca', 'm2'):
        p = os.path.join(V2, g, 'Reports', 'comparison.jsonl')
        for l in open(p, encoding='utf-8'):
            if l.strip():
                d = json.loads(l)
                out[(d['item'], d['method'], d['N'])] = d
    return out


def recall(rec, key):
    b = rec['budgets']['99.5']
    return b[key]['mean'] if key in b else np.nan


def k50(ks, rs):
    """Smallest severity where recall reaches 0.5, log-linearly interpolated; NaN if never/always."""
    ks, rs = np.asarray(ks, float), np.asarray(rs, float)
    ok = ~np.isnan(rs)
    ks, rs = ks[ok], rs[ok]
    if len(ks) < 2 or rs.max() < 0.5:
        return np.nan
    if rs[0] >= 0.5:
        return float(ks[0])          # already above 50% at the smallest severity tested
    for i in range(1, len(ks)):
        if rs[i] >= 0.5:
            x0, x1, y0, y1 = np.log(ks[i - 1]), np.log(ks[i]), rs[i - 1], rs[i]
            return float(np.exp(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)))
    return np.nan


def main():
    recs = load_recs()
    # 'dae_elbow' / 'dae_oracle' are not methods in the JSONL: resolve them to the z chosen per (item, N)
    zsel = pd.read_csv(os.path.join(V2, 'dae_z_selection.csv'))
    zmap = {(r['item'], int(r['N'])): (int(r['z_elbow']), int(r['z_oracle'])) for _, r in zsel.iterrows()}
    for (it, N), (ze, zo) in zmap.items():
        for tag, z in (('dae_elbow', ze), ('dae_oracle', zo)):
            src = recs.get((it, f'dae{z}', N))
            if src is not None:
                recs[(it, tag, N)] = src
    # yardstick: desirable TOTAL allowable error, appropriate for a single erroneous result (a sporadic,
    # per-sample error), rather than desirable bias, which is a specification for systematic error.
    aps = os.path.join(BAT, 'eflm_aps_panel21.csv')
    eflm = pd.read_csv(aps if os.path.exists(aps) else
                       os.path.join(BAT, 'eflm_desirable_bias_panel21.csv')).set_index('item')
    bias = pd.read_csv(os.path.join(BAT, 'bias_summary.csv'))
    bias = bias[bias.group == 'main'] if 'group' in bias else bias
    items = sorted({k[0] for k in recs})
    syn = {'CRE': 'CRE', 'Ca': 'CA', 'Cu': 'CU', 'Fe': 'FE', 'Zn': 'ZN', 'IgG': 'IGG', 'IgA': 'IGA', 'IgM': 'IGM'}

    # ---- (1) k50 ----
    rows = []
    for it in items:
        for m in METHODS:
            r = recs.get((it, m, N_MAIN))
            if r is None:
                continue
            row = dict(item=it, method=m)
            for t, ks in TYPES.items():
                row[f'k50_{t}'] = k50(ks, [recall(r, f'recall_{t}_k{k:g}') for k in ks])
            row['k50_pooled'] = np.nanmean([row[f'k50_{t}'] for t in TYPES])
            rows.append(row)
    k50df = pd.DataFrame(rows)
    k50df.to_csv(os.path.join(V2, 'k50_by_item.csv'), index=False)

    # ---- (2) clinically relevant severity ----
    out = []
    for it in items:
        col = YARDSTICK if YARDSTICK in eflm.columns else 'desirable_bias_pct'
        db = eflm.loc[it, col] if it in eflm.index else np.nan
        sname = syn.get(it, it)
        bi = bias[bias.item == sname]
        for t, ks in list(TYPES.items()) + list(NONADD.items()):
            bt = bi[bi.anomaly_type == t]
            if bt.empty or np.isnan(db):
                continue
            for col, tag in (('median', 'step_incl'), ('react_median', 'react_only')):
                b = bt.set_index('severity')[col].abs() * 100          # % bias
                hit = [k for k in ks if k in b.index and b[k] >= db]
                if not hit:
                    sev = np.nan
                else:
                    sev = max(hit) if t in NONADD else min(hit)        # NONADD: smaller q/r = larger bias
                rec = dict(item=it, anomaly_type=t, bias_def=tag, yardstick=YARDSTICK, limit_pct=db,
                           severity_at_TEa=sev, bias_at_severity=float(b[sev] if sev in b.index else np.nan))
                if not np.isnan(sev):
                    key = f'recall_{t}_k{sev:g}' if t in TYPES else (
                        f'recall_W_r{sev:g}' if t == 'W_timewarp' else f'recall_S_q{sev:g}')
                    for m in METHODS:
                        r = recs.get((it, m, N_MAIN))
                        rec[m] = recall(r, key) if r else np.nan
                out.append(rec)
    cl = pd.DataFrame(out)
    tag = '' if YARDSTICK == 'TEa_pct' else '_desirablebias'
    cl.to_csv(os.path.join(V2, f'clinical_severity{tag}.csv'), index=False)
    summ = cl.dropna(subset=['severity_at_TEa']).groupby(['anomaly_type', 'bias_def'])[METHODS].agg(['mean', 'min'])
    summ.to_csv(os.path.join(V2, f'clinical_summary{tag}.csv'))
    with pd.option_context('display.width', 300, 'display.max_columns', 40):
        print('== k50 (panel mean / worst item), N=500 ==')
        g = k50df.groupby('method')[[c for c in k50df if c.startswith('k50')]].agg(['mean', 'max'])
        print(g.round(3).to_string())
        print(f'\n== recall at the severity where the approximate deviation reaches the EFLM {YARDSTICK} ==')
        print(summ.round(3).to_string())
        print('\n== how many items have such a severity within the tested grid ==')
        print(cl.groupby(['anomaly_type', 'bias_def']).severity_at_TEa.apply(lambda v: int(v.notna().sum())).to_string())


if __name__ == '__main__':
    main()
