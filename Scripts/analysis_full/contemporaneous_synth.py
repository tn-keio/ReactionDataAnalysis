# Contemporaneous synthetic-anomaly generation (window-local).
#
# Ports the injection logic of generate_endpoint_synthetic_anomalies.py, but re-anchors the normal
# profile (mu, sigma, landmarks, unit shapes) and the base curves to a GIVEN block of real curves,
# instead of a fixed all-period random sample.  This removes the period mismatch between a fixed
# battery and a moving rolling-window model (see ForPaper/05,06,07).
#
# Anomaly is injected on Main (RAM); Sub (RAS) kept original for A-E so the deviation survives in
# Calc = RAM - RAS.  benign_offset shifts both channels equally -> cancels in Calc (a true negative).
# Severity in sigma units (k) using the BLOCK-LOCAL sigma_bar, so "k=1" means 1 local sigma.
#
# No TensorFlow. Deterministic given `seed`.

import numpy as np

T = 28                       # reaction points 11..38 -> index 0..27
IDX = np.arange(T)
SETTLE = 3                   # rise-phase width after onset (points)
K_GRID_DEFAULT = [1, 2, 3, 5, 8]
BENIGN_K_DEFAULT = [2, 5, 8]
D_LAG_DEFAULT = [1, 2, 3, 4, 5]
POS_TYPES = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']


def _robust_center_cov(X, keep=0.9, iters=5):
    """FAST-MCD-like robust mean/cov by iterative trimming (single start)."""
    loc = X.mean(0); cov = np.cov(X.T); n = max(2, int(len(X) * keep))
    for _ in range(iters):
        d = np.sum((X - loc) @ np.linalg.pinv(cov) * (X - loc), axis=1)
        core = X[np.argsort(d)[:n]]; loc, cov = core.mean(0), np.cov(core.T)
    return loc, cov


def screen_inliers(ram_block, ras_block, frac=0.9):
    """Return indices of the `frac` most-inlier curves of the block (by robust Mahalanobis distance
    on Calc = RAM - RAS).  Used to keep the synthetic 'clean' base genuinely normal: a randomly picked
    real curve may itself be an (unlabelled) anomaly, which would corrupt the ground truth."""
    calc = np.asarray(ram_block, float) - np.asarray(ras_block, float)
    loc, cov = _robust_center_cov(calc)
    d = np.sum((calc - loc) @ np.linalg.pinv(cov) * (calc - loc), axis=1)
    k = max(1, int(len(calc) * frac))
    return np.argsort(d)[:k]


def characterize(ram_block):
    """Local normal profile from a block of real Main curves (n, 28).  Same math as the frozen
    generator, but computed on the block passed in (contemporaneous)."""
    mu = ram_block.mean(0)
    sd = ram_block.std(0)
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
    shapes['E_baseline'] = np.where(IDX < onset, 1.0, np.clip((onset + 2 - IDX) / 2.0, 0, 1))
    return shapes


def make_battery(ram_block, ras_block, seed=0, m_base=200, k_grid=K_GRID_DEFAULT,
                 benign_k=BENIGN_K_DEFAULT, d_lag=D_LAG_DEFAULT, include_d=False, include_benign=True,
                 screen_frac=None):
    """Generate a contemporaneous anomaly battery from a real block.

    screen_frac: if set (e.g. 0.9), the profile (mu/sigma/shapes) AND the base curves are drawn only
      from the `screen_frac` most-inlier curves of the block (robust-distance screening), so the
      synthetic 'clean' ground truth is genuinely normal.  If None, no screening (legacy behaviour).

    Returns dict with:
      calc      : (R, 28) synthetic Calc curves (RAM_pert - RAS) for every row
      atype     : (R,) anomaly_type strings ('clean','A_spike',...,'benign_offset')
      severity  : (R,) float severity (k in sigma units; 0 for clean; lag for D)
      base_id   : (R,) int index into the sampled base curves
      profile   : the local profile dict (mu, sigbar, onset, ...)
    Row composition per battery: clean(M) + {A,B,C,E}×len(k_grid)×M [+ D×len(d_lag)×M] [+ benign×len(benign_k)×M].
    """
    rng = np.random.RandomState(seed)
    ram_block = np.asarray(ram_block, float); ras_block = np.asarray(ras_block, float)
    if screen_frac is not None and screen_frac < 1.0:
        keep = screen_inliers(ram_block, ras_block, frac=screen_frac)
        ram_block = ram_block[keep]; ras_block = ras_block[keep]
    n = len(ram_block)
    prof = characterize(ram_block)
    shapes = unit_shapes(prof)
    sigbar = prof['sigbar']; r0 = prof['rise'][0]
    M = min(m_base, n)
    idx = rng.permutation(n)[:M]
    base_ram = ram_block[idx]; base_ras = ras_block[idx]           # (M,28)
    base_calc = base_ram - base_ras
    spike_pt = rng.randint(max(r0, 1), T, size=M)

    calc_rows, atype, severity, base_id = [], [], [], []

    def add(name, sev, ram_pert, ras_arr):
        calc_rows.append(ram_pert - ras_arr)                      # (M,28)
        atype.extend([name] * M); severity.extend([float(sev)] * M); base_id.extend(range(M))

    add('clean', 0.0, base_ram, base_ras)
    for k in k_grid:
        A = k * sigbar
        pert = np.zeros((M, T))
        for b in range(M):
            pert[b] = A * np.exp(-((IDX - spike_pt[b]) / 0.9) ** 2)
        add('A_spike', k, base_ram + pert, base_ras)
        add('B_drift', k, base_ram + A * shapes['B_drift'][None, :], base_ras)
        add('C_hump', k, base_ram + A * shapes['C_hump'][None, :], base_ras)
        add('E_baseline', k, base_ram + A * shapes['E_baseline'][None, :], base_ras)
    if include_d:
        for L in d_lag:
            lagged = np.empty_like(base_ram)
            lagged[:, :L] = base_ram[:, :1]; lagged[:, L:] = base_ram[:, :T - L]
            add('D_kinetic', L, lagged, base_ras)
    if include_benign:
        for k in benign_k:
            A = k * sigbar
            add('benign_offset', k, base_ram + A, base_ras + A)    # both channels -> cancels in Calc

    return dict(calc=np.concatenate(calc_rows, axis=0),
                atype=np.array(atype), severity=np.array(severity, float),
                base_id=np.array(base_id, int), profile=prof)
