# Ca-only synthetic anomaly battery from a CLEAN, CONTEMPORANEOUS window (analyzer #2, date >= 2021-09-01,
# i.e. AFTER the Aug-2021 optical event and BEFORE #1/#3 come online).  Same generation logic as
# generate_endpoint_synthetic_anomalies_p5_38.py (points 5-38, T=34), restricted to that window, so the
# 'clean'/base curves are contemporaneous with the Ca re-run's training window.
# Output: Data/BigData/synthetic_anomalies/endpoint_synthetic_anomalies_p5_38_Ca_clean.parquet  (item='CA')
# Run with py3.14 (no TF).
import os, json
import numpy as np
import pandas as pd

OUT_DIR = 'Data/BigData/synthetic_anomalies'
P0 = 5
RAM = [f'RAM{i}' for i in range(P0, 39)]
RAS = [f'RAS{i}' for i in range(P0, 39)]
T = 39 - P0                      # 34
IDX = np.arange(T)
N_PROFILE = 2000; M_BASE = 200
K_GRID = [1, 2, 3, 5, 8]; D_LAG = [1, 2, 3, 4, 5]; BENIGN_K = [2, 5, 8]; SETTLE = 3; SEED = 42
ITEM = 'CA'; PROC_NAME = 'Ca'
WIN_START = '2021-09-01'         # after Aug-2021 event
WIN_END = '2022-04-30'           # CONTEMPORANEOUS with the re-run training window (all #2; before #1/#3);
                                 # base curves must be contemporaneous (full #2 history drifts -> 30% FP)


def characterize(normal_raw):
    mu = normal_raw.mean(0); sd = normal_raw.std(0); dmu = np.abs(np.diff(mu))
    onset = int(np.argmax(dmu)); b0, b1 = 0, max(0, onset - 1)
    r0, r1 = onset, min(T - 1, onset + SETTLE)
    p0, p1 = min(T - 1, onset + SETTLE + 1), T - 1
    direction = int(np.sign(mu[p0:p1 + 1].mean() - mu[b0:b1 + 1].mean()) or 1)
    sigbar = float(np.median(sd)) + 1e-9
    span = float(abs(mu[p0:p1 + 1].mean() - mu[b0:b1 + 1].mean()))
    return dict(mu=mu, sd=sd, onset=onset, baseline=[b0, b1], rise=[r0, r1],
               plateau=[p0, p1], direction=direction, sigbar=sigbar, span=span)


def unit_shapes(prof):
    onset = prof['onset']; p0 = prof['plateau'][0]; shapes = {}
    denom = max(1, (T - 1) - onset)
    shapes['B_drift'] = np.clip((IDX - onset) / denom, 0, 1)
    hump = np.exp(-((IDX - (onset + 2)) / 1.6) ** 2) - 0.5 * np.exp(-((IDX - (p0 + 3)) / 2.0) ** 2)
    shapes['C_hump'] = hump / (np.abs(hump).max() + 1e-9)
    e = np.where(IDX < onset, 1.0, np.clip((onset + 2 - IDX) / 2.0, 0, 1))
    shapes['E_baseline'] = e
    return shapes


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_parquet('Data/BigData/processed', columns=['S_Date', 'SystemName'] + RAM + RAS,
                         filters=[('A_Name', '==', PROC_NAME)])
    d = pd.to_datetime(df['S_Date'])
    sysv = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    m = (sysv == '#2') & (d >= WIN_START) & (d <= WIN_END)
    df = df[m.values].copy()
    print(f'Ca clean window (#2, {WIN_START}..{WIN_END}): {len(df):,} rows, '
          f'{pd.to_datetime(df["S_Date"]).min().date()}..{pd.to_datetime(df["S_Date"]).max().date()}')

    all_rows = []
    n = len(df); perm = rng.permutation(n)
    prof_idx = perm[:min(N_PROFILE, n)]; base_idx = perm[:min(M_BASE, n)]
    RAM_all = df[RAM].values.astype(np.float64); RAS_all = df[RAS].values.astype(np.float64)
    prof = characterize(RAM_all[prof_idx]); shapes = unit_shapes(prof)
    sigbar = prof['sigbar']; r0 = prof['rise'][0]
    base_ram = RAM_all[base_idx]; base_ras = RAS_all[base_idx]; M = len(base_ram)
    spike_pt = rng.randint(max(r0, 1), T, size=M)

    def add_rows(atype, sev, phase, RAM_arr, RAS_arr, inj_pt):
        dev = np.abs(RAM_arr - base_ram).max(axis=1) / sigbar
        for b in range(M):
            row = {'item': ITEM, 'base_id': int(b), 'anomaly_type': atype, 'severity': float(sev),
                   'phase': phase, 'inject_point': int(inj_pt[b]) if inj_pt is not None else -1,
                   'achieved_dev_sigma': float(dev[b]), 'direction': prof['direction']}
            for j, c in enumerate(RAM): row[c] = float(RAM_arr[b, j])
            for j, c in enumerate(RAS): row[c] = float(RAS_arr[b, j])
            all_rows.append(row)

    add_rows('clean', 0.0, 'none', base_ram, base_ras, None)
    for k in K_GRID:
        A = k * sigbar; pert = np.zeros((M, T))
        for b in range(M): pert[b] = A * np.exp(-((IDX - spike_pt[b]) / 0.9) ** 2)
        add_rows('A_spike', k, 'rise/plateau', base_ram + pert, base_ras, spike_pt)
        add_rows('B_drift', k, 'onset->end', base_ram + A * shapes['B_drift'][None, :], base_ras, None)
        add_rows('C_hump', k, 'rise/plateau', base_ram + A * shapes['C_hump'][None, :], base_ras, None)
        add_rows('E_baseline', k, 'baseline', base_ram + A * shapes['E_baseline'][None, :], base_ras, None)
    for L in D_LAG:
        lagged = np.empty_like(base_ram); lagged[:, :L] = base_ram[:, :1]; lagged[:, L:] = base_ram[:, :T - L]
        add_rows('D_kinetic', L, f'lag={L}pt', lagged, base_ras, None)
    for k in BENIGN_K:
        A = k * sigbar; add_rows('benign_offset', k, 'whole', base_ram + A, base_ras + A, None)

    out = pd.DataFrame(all_rows)
    meta_cols = ['item', 'base_id', 'anomaly_type', 'severity', 'phase',
                 'inject_point', 'achieved_dev_sigma', 'direction']
    out = out[meta_cols + RAM + RAS]
    pq = os.path.join(OUT_DIR, 'endpoint_synthetic_anomalies_p5_38_Ca_clean.parquet')
    out.to_parquet(pq, index=False)
    print(f'onset=pt{P0+prof["onset"]} dir={prof["direction"]:+d} sigbar={sigbar:.2f} span={prof["span"]:.1f}')
    print('anomaly_type counts:'); print(out['anomaly_type'].value_counts().to_string())
    print(f'rows: {len(out):,}  saved: {pq}')


if __name__ == '__main__':
    main()
