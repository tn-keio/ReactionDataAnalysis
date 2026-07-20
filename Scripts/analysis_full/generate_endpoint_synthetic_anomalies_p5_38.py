# Generate item-specific synthetic anomaly data for endpoint assays -- POINTS 5-38 version.
# Copyright (C) 2026
#
# Identical logic to generate_endpoint_synthetic_anomalies.py (which is fixed at points 11-38, T=28),
# but built on the CONFIRMED reaction-point window 5-38 (T=34).  Everything is parameterized by the
# RAM/RAS column lists and T, so the anomaly library (A_spike/B_drift/C_hump/D_kinetic/E_baseline/
# benign_offset), the phase-relative placement, and the k*sigma severities are unchanged in definition.
#
# Output (separate filenames; the 11-38 files are preserved):
#   Data/BigData/synthetic_anomalies/endpoint_synthetic_anomalies_p5_38.parquet
#   Data/BigData/synthetic_anomalies/endpoint_normal_profiles_p5_38.json
#
# Run with py3.14 (default `py`) -- no TensorFlow needed.

import os
import json
import numpy as np
import pandas as pd

OUT_DIR = 'Data/BigData/synthetic_anomalies'
os.makedirs(OUT_DIR, exist_ok=True)

P0 = 5                                     # window start point (confirmed)
RAM = [f'RAM{i}' for i in range(P0, 39)]
RAS = [f'RAS{i}' for i in range(P0, 39)]
T = 39 - P0                                # reaction points 5..38 -> index 0..33  (T=34)
IDX = np.arange(T)
N_PROFILE = 2000
M_BASE = 200
K_GRID = [1, 2, 3, 5, 8]
D_LAG = [1, 2, 3, 4, 5]
BENIGN_K = [2, 5, 8]
SETTLE = 3
SEED = 42

MP = {'CA': 'Ca', 'FE': 'Fe', 'CU': 'Cu', 'ZN': 'Zn',
      'IGG': 'IgG', 'IGA': 'IgA', 'IGM': 'IgM', 'RPR-2': 'RPR'}


def characterize(normal_raw):
    mu = normal_raw.mean(0)
    sd = normal_raw.std(0)
    dmu = np.abs(np.diff(mu))
    onset = int(np.argmax(dmu))
    b0, b1 = 0, max(0, onset - 1)
    r0, r1 = onset, min(T - 1, onset + SETTLE)
    p0, p1 = min(T - 1, onset + SETTLE + 1), T - 1
    direction = int(np.sign(mu[p0:p1 + 1].mean() - mu[b0:b1 + 1].mean()) or 1)
    sigbar = float(np.median(sd)) + 1e-9
    span = float(abs(mu[p0:p1 + 1].mean() - mu[b0:b1 + 1].mean()))
    return dict(mu=mu, sd=sd, onset=onset, baseline=[b0, b1], rise=[r0, r1],
               plateau=[p0, p1], direction=direction, sigbar=sigbar, span=span)


def unit_shapes(prof):
    onset = prof['onset']; p0 = prof['plateau'][0]
    shapes = {}
    denom = max(1, (T - 1) - onset)
    shapes['B_drift'] = np.clip((IDX - onset) / denom, 0, 1)
    hump = np.exp(-((IDX - (onset + 2)) / 1.6) ** 2) - 0.5 * np.exp(-((IDX - (p0 + 3)) / 2.0) ** 2)
    shapes['C_hump'] = hump / (np.abs(hump).max() + 1e-9)
    e = np.where(IDX < onset, 1.0, np.clip((onset + 2 - IDX) / 2.0, 0, 1))
    shapes['E_baseline'] = e
    return shapes


def main():
    rng = np.random.RandomState(SEED)
    csv = pd.read_csv('Data/parameter_mapped.csv')
    items = csv[csv['type'].isin(['1point', '2point'])]['ItemName'].astype(str).tolist()

    all_rows = []
    profiles = {}
    for it in items:
        p = MP.get(it, it)
        df = pd.read_parquet('Data/BigData/processed', columns=RAM + RAS,
                             filters=[('A_Name', '==', p)])
        if len(df) < 50:
            print('skip (too few)', it); continue
        n = len(df)
        perm = rng.permutation(n)
        prof_idx = perm[:min(N_PROFILE, n)]
        base_idx = perm[:min(M_BASE, n)]
        RAM_all = df[RAM].values.astype(np.float64)
        RAS_all = df[RAS].values.astype(np.float64)
        prof = characterize(RAM_all[prof_idx])
        shapes = unit_shapes(prof)
        sigbar = prof['sigbar']; onset = prof['onset']
        r0 = prof['rise'][0]

        base_ram = RAM_all[base_idx]               # (M,34)
        base_ras = RAS_all[base_idx]
        M = len(base_ram)
        spike_pt = rng.randint(max(r0, 1), T, size=M)

        def add_rows(atype, sev, phase, RAM_arr, RAS_arr, inj_pt):
            dev = np.abs(RAM_arr - base_ram).max(axis=1) / sigbar
            for b in range(M):
                row = {'item': it, 'base_id': int(b), 'anomaly_type': atype,
                       'severity': float(sev), 'phase': phase,
                       'inject_point': int(inj_pt[b]) if inj_pt is not None else -1,
                       'achieved_dev_sigma': float(dev[b]),
                       'direction': prof['direction']}
                for j, c in enumerate(RAM): row[c] = float(RAM_arr[b, j])
                for j, c in enumerate(RAS): row[c] = float(RAS_arr[b, j])
                all_rows.append(row)

        add_rows('clean', 0.0, 'none', base_ram, base_ras, None)
        for k in K_GRID:
            A = k * sigbar
            pert = np.zeros((M, T))
            for b in range(M):
                pert[b] = A * np.exp(-((IDX - spike_pt[b]) / 0.9) ** 2)
            add_rows('A_spike', k, 'rise/plateau', base_ram + pert, base_ras, spike_pt)
            add_rows('B_drift', k, 'onset->end', base_ram + A * shapes['B_drift'][None, :], base_ras, None)
            add_rows('C_hump', k, 'rise/plateau', base_ram + A * shapes['C_hump'][None, :], base_ras, None)
            add_rows('E_baseline', k, 'baseline', base_ram + A * shapes['E_baseline'][None, :], base_ras, None)
        for L in D_LAG:
            lagged = np.empty_like(base_ram)
            lagged[:, :L] = base_ram[:, :1]
            lagged[:, L:] = base_ram[:, :T - L]
            add_rows('D_kinetic', L, f'lag={L}pt', lagged, base_ras, None)
        for k in BENIGN_K:
            A = k * sigbar
            add_rows('benign_offset', k, 'whole', base_ram + A, base_ras + A, None)

        profiles[it] = {'parquet_name': p, 'n_total': int(n),
                        'mu_raw': prof['mu'].round(3).tolist(),
                        'sigma_raw': prof['sd'].round(3).tolist(),
                        'onset_idx': prof['onset'], 'onset_point': int(P0 + prof['onset']),
                        'baseline_idx': prof['baseline'], 'rise_idx': prof['rise'],
                        'plateau_idx': prof['plateau'], 'direction': prof['direction'],
                        'sigma_bar': round(sigbar, 4), 'span': round(prof['span'], 3)}
        print(f'{it:8} n={n:>9,} onset=pt{P0+prof["onset"]:<2} dir={prof["direction"]:+d} '
              f'sigbar={sigbar:8.2f} span={prof["span"]:8.1f}')

    out = pd.DataFrame(all_rows)
    meta_cols = ['item', 'base_id', 'anomaly_type', 'severity', 'phase',
                 'inject_point', 'achieved_dev_sigma', 'direction']
    out = out[meta_cols + RAM + RAS]
    pq = os.path.join(OUT_DIR, 'endpoint_synthetic_anomalies_p5_38.parquet')
    out.to_parquet(pq, index=False)
    with open(os.path.join(OUT_DIR, 'endpoint_normal_profiles_p5_38.json'), 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=1)

    print(f'\nItems: {len(profiles)} | rows: {len(out):,} | window points {P0}-38 (T={T})')
    print('anomaly_type counts:'); print(out['anomaly_type'].value_counts().to_string())
    print(f'Saved: {pq}')


if __name__ == '__main__':
    main()
