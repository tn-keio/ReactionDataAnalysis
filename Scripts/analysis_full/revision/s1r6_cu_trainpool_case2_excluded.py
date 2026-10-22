# Stage 1r-6: the two curves of the second Cu case (initial -5 and 3x repeat 93) lie in
# the Cu training pool of the main benchmark (Table 1) and were used as "normal" training data at N = 5,000 and
# 10,000.  Re-run the deterministic methods (empirical Mahalanobis, iterative-trimming robust, PCA) on Cu with and
# without those two curves.  Replicates run_paper_method_comparison.py (main group, window 5-38, scalar min-max
# WITHOUT clip = rev_common.apply_norm_unclipped, splits 10,000/2,000/3,000, holdout-clean P99.5) without importing
# it (it imports TensorFlow).  "excluded" drops the two curves and keeps N by taking the next curves of the pool.
# Output: Results/20261019_Rev_CuTrainpoolCase2/  (local only)
import os
import json
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA

BASE = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
SYN = os.path.join(BASE, 'Data', 'BigData', 'synthetic_anomalies', 'endpoint_synthetic_anomalies_p5_38.parquet')
CASE_ROWS = os.path.join(BASE, 'Results', '20260925_Rev_CuCase2', 'raw_rows.csv')
OUT = os.path.join(BASE, 'Results', '20261019_Rev_CuTrainpoolCase2')
RAM = [f'RAM{i}' for i in range(5, 39)]; RAS = [f'RAS{i}' for i in range(5, 39)]
N_TR, N_HO, N_AN = 10000, 2000, 3000
POS = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']


def mcd_like(D, keep=0.95, iters=5):
    loc = D.mean(0); cov = np.cov(D.T); n = int(len(D) * keep)
    for _ in range(iters):
        d = np.sum((D - loc) @ np.linalg.pinv(cov) * (D - loc), axis=1)
        core = D[np.argsort(d)[:n]]; loc, cov = core.mean(0), np.cov(core.T)
    return loc, cov


def maha(loc, cov, X):
    P = np.linalg.pinv(cov)
    return np.sum((X - loc) @ P * (X - loc), axis=1)


def scorer(method, Xtr):
    if method == 'maha':
        e = EmpiricalCovariance().fit(Xtr); return lambda X: maha(e.location_, e.covariance_, X)
    if method == 'maha_robust':
        loc, cov = mcd_like(Xtr); return lambda X: maha(loc, cov, X)
    m = PCA(n_components=0.95).fit(Xtr)
    return lambda X: np.mean((X - m.inverse_transform(m.transform(X))) ** 2, axis=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    sid = str(pd.read_csv(CASE_ROWS, dtype=str)['S_ID'].iloc[0]).strip()
    df = pd.read_parquet(PROC, filters=[('A_Name', '==', 'Cu')], columns=['S_Date', 'S_ID'] + RAM + RAS)
    df = df.sort_values('S_Date').reset_index(drop=True).iloc[:N_TR + N_HO + N_AN]
    calc = df[RAM].values.astype('float64') - df[RAS].values.astype('float64')
    lo, hi = float(calc[:N_TR].min()), float(calc[:N_TR].max())
    X = (calc - lo) / (hi - lo + 1e-8)
    pos_case = np.where(df['S_ID'].astype(str).str.strip().values == sid)[0]
    print('case-2 positions in the date-sorted Cu set (0-based):', pos_case.tolist())
    s = pd.read_parquet(SYN)
    s = s[s['item'].astype(str).str.upper() == 'CU'].reset_index(drop=True)
    Xs = ((s[RAM].values.astype('float64') - s[RAS].values.astype('float64')) - lo) / (hi - lo + 1e-8)
    at = s['anomaly_type'].values; sev = s['severity'].values
    main_sel = np.isin(at, POS) & (sev >= 2); clean = at == 'clean'
    ho = X[N_TR:N_TR + N_HO]
    keep = np.setdiff1d(np.arange(N_TR), pos_case)
    rows = []
    for N in (500, 5000, 10000):
        for cond, idx in (('submitted', np.arange(N)), ('case2_excluded', keep[:N])):
            for mth in ('maha', 'maha_robust', 'pca'):
                f = scorer(mth, X[idx]); thr = np.percentile(f(ho), 99.5); fs = f(Xs) >= thr
                r = dict(N=N, condition=cond, n_train=len(idx), case_in_train=int(np.isin(pos_case, idx).sum()),
                         method=mth, recall_main=round(float(fs[main_sel].mean()), 4),
                         specificity_clean=round(float(1 - fs[clean].mean()), 4))
                for t in POS:
                    r[f'recall_{t}'] = round(float(fs[(at == t) & (sev >= 2)].mean()), 4)
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, 'cu_trainpool_case2.csv'), index=False)
    print(out[['N', 'condition', 'n_train', 'case_in_train', 'method', 'recall_main', 'specificity_clean']].to_string(index=False))


if __name__ == '__main__':
    main()
