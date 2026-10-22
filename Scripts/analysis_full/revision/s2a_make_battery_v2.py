# Stage 2a: EXTRA synthetic rows for the v2 benchmark, appended at evaluation
# time to the three submitted batteries (which are NOT modified):
#   (a) sub-1-sigma additive anomalies  A/B/C/E at k in K_LOW
#       = clean + k * (row_k1 - clean), per base curve.  Exact: in all three generators the spike
#       positions are drawn once before the k loop and every perturbation is linear in A = k*sigbar.
#   (b) W_timewarp  : kinetic slowing of the REACTION PHASE only, applied to BOTH wavelengths:
#                     X'(t) = X(ts + r (t - ts)) for t > ts, X in {RAM, RAS},
#                     ts = t0 + 1 = first point after the second-reagent addition step (linear interp.).
#                     The addition step itself stays instantaneous (warping from t0 would smear it).
#   (c) S_saturation: the measured calc signal's reaction phase is hard-capped at fraction q of its own span:
#       calc' = min(calc, calc[ts] + q*(end - calc[ts])) (max(...) for falling), t >= ts; RAM' = RAS + calc'
#       (a deliberately simplified worst-case shape, not a kinetic model)
#   (revised 2026-09-17: the first version measured the span from the pre-addition baseline, so the
#    reagent-addition step dominated the span and the cap/warp distorted the step itself)
# For (a) RAS is left unchanged (as in the submitted generators); (b) warps both wavelengths; (c) acts on calc.
# Per row we also store an approximate reported-result bias on the calc channel:
#   bias_approx  : delta = mean(calc[last 2]) - mean(calc[pre-addition points])   (includes the addition step)
#   bias_react   : delta = mean(calc[last 2]) - calc[ts]                           (reaction phase only)
#   Both are approximations of the endpoint read-out; the instrument's photometric points are not used.
# (an approximation of the two-point endpoint read-out; the instrument's photometric points are not used).
#
# onset t0 and sigbar are recovered from each battery's own k=1 rows (identical rule for all three):
#   sigbar = max over points of the B_drift k=1 perturbation   (unit drift shape reaches 1 at the end)
#   t0     = (first index where the E_baseline k=1 perturbation falls below sigbar) - 1
#   (validated against endpoint_normal_profiles_p5_38.json for the main battery)
#
# Output (new files): Data/BigData/synthetic_anomalies/rev2026_extra_{main,ca,m2}.parquet
#                     Results/20260917_Rev_M2_BatteryV2/{battery_profile.csv, bias_summary.csv}
# py3.14, no TensorFlow.

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SYN_DIR = os.path.join(BASE, 'Data', 'BigData', 'synthetic_anomalies')
OUT = os.path.join(BASE, 'Results', '20260917_Rev_M2_BatteryV2')
SRC = {'main': 'endpoint_synthetic_anomalies_p5_38.parquet',
       'ca': 'endpoint_synthetic_anomalies_p5_38_Ca_clean.parquet',
       'm2': 'endpoint_synthetic_anomalies_p5_38_m2clean.parquet'}
P0 = 5
RAM = [f'RAM{i}' for i in range(P0, 39)]
RAS = [f'RAS{i}' for i in range(P0, 39)]
T = len(RAM)
POS = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']
K_LOW = [0.02, 0.05, 0.1, 0.2, 0.5]
R_WARP = [0.95, 0.9, 0.8, 0.7, 0.5]
Q_SAT = [0.98, 0.95, 0.9, 0.8]
# Severe premature termination is only plausible for the immunoturbidimetric assays (antigen excess is
# one of its mechanisms); adding these levels to every item would invalidate the cached image features
# of the whole panel for nothing.  NOTE: this is NOT a faithful model of prozone - real antigen excess
# also speeds up the early reaction and can hook downwards; see M2_Request.md (2026-09-19).
Q_PROZONE = [0.5, 0.2]
IMMUNO = {'IGG', 'IGA', 'IGM', 'CRP', 'KL-6', 'CRE'}     # CRE is the non-immuno control item
META = ['item', 'base_id', 'anomaly_type', 'severity', 'phase', 'inject_point', 'achieved_dev_sigma', 'direction']


def by_base(df, atype, sev):
    d = df[(df.anomaly_type == atype) & np.isclose(df.severity, sev)].sort_values('base_id')
    return d


def endpoint_delta(calc, t0):
    b = calc[:, :max(t0, 1)].mean(1)
    return calc[:, -2:].mean(1) - b


def reaction_delta(calc, ts):
    return calc[:, -2:].mean(1) - calc[:, ts]


def build_item(df):
    clean = by_base(df, 'clean', 0.0)
    ids = clean.base_id.values
    ram0 = clean[RAM].values; ras0 = clean[RAS].values
    k1 = {t: by_base(df, t, 1.0) for t in POS}
    for t in POS:
        assert (k1[t].base_id.values == ids).all(), f'base_id mismatch for {t}'
    dB = k1['B_drift'][RAM].values - ram0
    dE = k1['E_baseline'][RAM].values - ram0
    sigbar = float(np.median(dB.max(1)))
    assert np.allclose(dB.max(1), sigbar, rtol=1e-6), 'B_drift k=1 amplitude not constant'
    shapeE = np.median(dE, 0) / sigbar
    # E shape = 1 for IDX < onset and clip((onset + 2 - IDX)/2, 0, 1) after, i.e. still 1 AT onset:
    # the first index below 1 is onset + 1.
    t0 = int(np.argmax(shapeE < 1 - 1e-9)) - 1
    direction = int(clean.direction.iloc[0])
    ts = t0 + 1
    delta0 = endpoint_delta(ram0 - ras0, t0)
    rdelta0 = reaction_delta(ram0 - ras0, ts)
    rows, prof = [], dict(sigbar=sigbar, onset_idx=t0, onset_point=P0 + t0, direction=direction,
                          n_base=len(ids), delta_median=float(np.median(delta0)),
                          react_delta_median=float(np.median(rdelta0)),
                          step_median=float(np.median((ram0 - ras0)[:, ts] - (ram0 - ras0)[:, :max(t0, 1)].mean(1))))

    def add(atype, sev, ram, phase, ras=None):
        ras = ras0 if ras is None else ras
        calc = ram - ras
        with np.errstate(divide='ignore', invalid='ignore'):
            bias = np.where(delta0 != 0, endpoint_delta(calc, t0) / delta0 - 1.0, np.nan)
            rbias = np.where(rdelta0 != 0, reaction_delta(calc, ts) / rdelta0 - 1.0, np.nan)
        dev = np.abs(ram - ram0).max(1) / sigbar
        blk = pd.DataFrame({'item': clean.item.values, 'base_id': ids, 'anomaly_type': atype,
                            'severity': float(sev), 'phase': phase, 'inject_point': -1,
                            'achieved_dev_sigma': dev, 'direction': direction, 'bias_approx': bias,
                            'bias_react': rbias})
        rows.append(pd.concat([blk, pd.DataFrame(ram, columns=RAM), pd.DataFrame(ras, columns=RAS)], axis=1))

    for t in POS:
        d1 = k1[t][RAM].values - ram0
        for k in K_LOW:
            add(t, k, ram0 + k * d1, f'lowk:{k1[t].phase.iloc[0]}')
    x = np.arange(T, dtype=float)
    for r in R_WARP:
        src = np.where(x > ts, ts + r * (x - ts), x)
        warp = lambda A: np.vstack([np.interp(src, x, c) for c in A])
        add('W_timewarp', r, warp(ram0), f'timewarp r={r}', ras=warp(ras0))   # both wavelengths
    calc0 = ram0 - ras0                       # saturation is imposed on the measured (calc) signal
    b = calc0[:, ts]; e = calc0[:, -2:].mean(1); span = e - b
    after = (x >= ts)[None, :]
    qs = Q_SAT + (Q_PROZONE if str(clean.item.iloc[0]) in IMMUNO else [])
    for q in qs:
        cap = (b + q * span)[:, None]
        capped = np.where(span[:, None] >= 0, np.minimum(calc0, cap), np.maximum(calc0, cap))
        sat = np.where(after, capped, calc0)
        add('S_saturation', q, ras0 + sat, f'saturation q={q} (calc; hard cap)')
    # sanity: the scaled k=1 row reproduces the stored k=1 row exactly
    for t in POS:
        assert np.allclose(ram0 + 1.0 * (k1[t][RAM].values - ram0), k1[t][RAM].values)
    return pd.concat(rows, ignore_index=True), prof


def main():
    os.makedirs(OUT, exist_ok=True)
    profs, bias = [], []
    for g, fn in SRC.items():
        src = pd.read_parquet(os.path.join(SYN_DIR, fn))
        outs = []
        for it, df in src.groupby('item'):
            # 'main' holds every endpoint item (incl. CA/ALB/IP); the evaluation picks the rows per loader group
            ext, prof = build_item(df.reset_index(drop=True))
            outs.append(ext); profs.append(dict(group=g, item=it, **prof))
            s = ext.groupby(['anomaly_type', 'severity']).agg(
                median=('bias_approx', 'median'), react_median=('bias_react', 'median'),
                react_p05=('bias_react', lambda v: np.nanpercentile(v, 5)),
                react_p95=('bias_react', lambda v: np.nanpercentile(v, 95)),
                n_nan=('bias_react', lambda v: int(np.isnan(v).sum()))).reset_index()
            s.insert(0, 'item', it); s.insert(0, 'group', g); bias.append(s)
        pd.concat(outs, ignore_index=True)[META + ['bias_approx', 'bias_react'] + RAM + RAS].to_parquet(
            os.path.join(SYN_DIR, f'rev2026_extra_{g}.parquet'), index=False)
        print(f'{g}: {len(outs)} items written', flush=True)
    pd.DataFrame(profs).to_csv(os.path.join(OUT, 'battery_profile.csv'), index=False)
    pd.concat(bias).to_csv(os.path.join(OUT, 'bias_summary.csv'), index=False)
    print(pd.DataFrame(profs).to_string())


if __name__ == '__main__':
    main()
