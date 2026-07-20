# Clean contemporaneous synthetic batteries for contaminated module-2 items (ALB, IP), same window and
# logic as generate_ca_clean_synthetic.py: analyzer #2, 2021-09-01..2022-04-30 (post Aug-2021 event,
# pre #1/#3).  Output ONE parquet with item in {'ALB','IP'} (proc name == synth code for these).
import os
import numpy as np
import pandas as pd

OUT_DIR = 'Data/BigData/synthetic_anomalies'
P0 = 5
RAM = [f'RAM{i}' for i in range(P0, 39)]; RAS = [f'RAS{i}' for i in range(P0, 39)]
T = 39 - P0; IDX = np.arange(T)
N_PROFILE = 2000; M_BASE = 200
K_GRID = [1, 2, 3, 5, 8]; D_LAG = [1, 2, 3, 4, 5]; BENIGN_K = [2, 5, 8]; SETTLE = 3; SEED = 42
ITEMS = [('ALB', 'ALB'), ('IP', 'IP')]      # (proc_name, synth_code)
WIN_START, WIN_END = '2021-09-01', '2022-04-30'


def characterize(normal_raw):
    mu = normal_raw.mean(0); sd = normal_raw.std(0); dmu = np.abs(np.diff(mu))
    onset = int(np.argmax(dmu)); b0, b1 = 0, max(0, onset - 1)
    r0, r1 = onset, min(T - 1, onset + SETTLE)
    p0, p1 = min(T - 1, onset + SETTLE + 1), T - 1
    direction = int(np.sign(mu[p0:p1 + 1].mean() - mu[b0:b1 + 1].mean()) or 1)
    return dict(mu=mu, sd=sd, onset=onset, baseline=[b0, b1], rise=[r0, r1], plateau=[p0, p1],
               direction=direction, sigbar=float(np.median(sd)) + 1e-9,
               span=float(abs(mu[p0:p1 + 1].mean() - mu[b0:b1 + 1].mean())))


def unit_shapes(prof):
    onset = prof['onset']; p0 = prof['plateau'][0]; shapes = {}
    denom = max(1, (T - 1) - onset)
    shapes['B_drift'] = np.clip((IDX - onset) / denom, 0, 1)
    hump = np.exp(-((IDX - (onset + 2)) / 1.6) ** 2) - 0.5 * np.exp(-((IDX - (p0 + 3)) / 2.0) ** 2)
    shapes['C_hump'] = hump / (np.abs(hump).max() + 1e-9)
    shapes['E_baseline'] = np.where(IDX < onset, 1.0, np.clip((onset + 2 - IDX) / 2.0, 0, 1))
    return shapes


def gen_item(proc, code, all_rows):
    df = pd.read_parquet('Data/BigData/processed', columns=['S_Date', 'SystemName'] + RAM + RAS,
                         filters=[('A_Name', '==', proc)])
    d = pd.to_datetime(df['S_Date'])
    sysv = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    df = df[((sysv == '#2') & (d >= WIN_START) & (d <= WIN_END)).values].copy()
    n = len(df)
    print(f'{code}: clean #2 {WIN_START}..{WIN_END} -> {n:,} rows')
    rng = np.random.RandomState(SEED); perm = rng.permutation(n)
    RAM_all = df[RAM].values.astype(np.float64); RAS_all = df[RAS].values.astype(np.float64)
    prof = characterize(RAM_all[perm[:min(N_PROFILE, n)]]); shapes = unit_shapes(prof)
    sigbar = prof['sigbar']; r0 = prof['rise'][0]
    base_ram = RAM_all[perm[:min(M_BASE, n)]]; base_ras = RAS_all[perm[:min(M_BASE, n)]]; M = len(base_ram)
    spike_pt = rng.randint(max(r0, 1), T, size=M)

    def add(atype, sev, phase, RA, RS, inj):
        dev = np.abs(RA - base_ram).max(axis=1) / sigbar
        for b in range(M):
            row = {'item': code, 'base_id': int(b), 'anomaly_type': atype, 'severity': float(sev),
                   'phase': phase, 'inject_point': int(inj[b]) if inj is not None else -1,
                   'achieved_dev_sigma': float(dev[b]), 'direction': prof['direction']}
            for j, c in enumerate(RAM): row[c] = float(RA[b, j])
            for j, c in enumerate(RAS): row[c] = float(RS[b, j])
            all_rows.append(row)

    add('clean', 0.0, 'none', base_ram, base_ras, None)
    for k in K_GRID:
        A = k * sigbar; pert = np.zeros((M, T))
        for b in range(M): pert[b] = A * np.exp(-((IDX - spike_pt[b]) / 0.9) ** 2)
        add('A_spike', k, 'rise/plateau', base_ram + pert, base_ras, spike_pt)
        add('B_drift', k, 'onset->end', base_ram + A * shapes['B_drift'][None, :], base_ras, None)
        add('C_hump', k, 'rise/plateau', base_ram + A * shapes['C_hump'][None, :], base_ras, None)
        add('E_baseline', k, 'baseline', base_ram + A * shapes['E_baseline'][None, :], base_ras, None)
    for L in D_LAG:
        lg = np.empty_like(base_ram); lg[:, :L] = base_ram[:, :1]; lg[:, L:] = base_ram[:, :T - L]
        add('D_kinetic', L, f'lag={L}pt', lg, base_ras, None)
    for k in BENIGN_K:
        A = k * sigbar; add('benign_offset', k, 'whole', base_ram + A, base_ras + A, None)


def main():
    all_rows = []
    for proc, code in ITEMS:
        gen_item(proc, code, all_rows)
    out = pd.DataFrame(all_rows)
    meta = ['item', 'base_id', 'anomaly_type', 'severity', 'phase', 'inject_point', 'achieved_dev_sigma', 'direction']
    out = out[meta + RAM + RAS]
    pq = os.path.join(OUT_DIR, 'endpoint_synthetic_anomalies_p5_38_m2clean.parquet')
    out.to_parquet(pq, index=False)
    print('items:', out['item'].unique().tolist(), '| rows:', len(out), '| saved', pq)


if __name__ == '__main__':
    main()
