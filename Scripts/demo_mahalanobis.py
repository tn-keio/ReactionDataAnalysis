# -*- coding: utf-8 -*-
"""
Self-contained demo: empirical Mahalanobis vs classical baselines on the
shipped reaction-curve CSV sample (Data/<ITEM>.csv).

This runs out-of-the-box on the repository sample (up to 2,000 curves/item) and
reproduces the *qualitative* headline result of the paper: empirical-covariance
Mahalanobis distance detects the synthetic anomaly battery, including subtle 1sigma
perturbations, more sensitively than PCA / LOF / Isolation Forest / One-Class SVM,
at a matched (self-calibrated) specificity budget.

NOTE. The published headline numbers use up to 15,000 date-sorted curves/item from
the full dataset (available on request; see README). The 2,000-curve random sample
here gives smaller, noisier numbers but the same ranking. Deep-learning baselines
(TensorFlow) are omitted from this demo; see Scripts/analysis_full/ for the full
benchmark.

Usage:
    python demo_mahalanobis.py            # default item = CRE
    python demo_mahalanobis.py --item TP  # any item present in Data/
Requires: numpy, pandas, scipy, scikit-learn.
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = os.path.join(REPO, 'Data')

PTS = list(range(5, 39))                       # analysis window: points 5-38
RAM = [f'RAM{p}' for p in PTS]
RAS = [f'RAS{p}' for p in PTS]
N_TRAIN = 500
SEVERITIES = [1, 2, 5, 8]                       # k x sigma_bar
N_BASE = 150                                    # clean base curves per anomaly type
RNG = np.random.default_rng(0)


def load_calc(item):
    df = pd.read_csv(os.path.join(DATA, f'{item}.csv'))
    df = df.sort_values('S_Date').reset_index(drop=True)
    calc = df[RAM].to_numpy(float) - df[RAS].to_numpy(float)   # Calc = main - sub
    return calc


def minmax_fit(train):
    lo = train.min(0)
    hi = train.max(0)
    rng = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
    return lo, rng


def inject(base, sigma, kind, k):
    """Inject one anomaly of type `kind` at severity k*sigma onto a raw base curve."""
    x = base.copy()
    T = len(x)
    sbar = float(sigma.mean())
    if kind == 'A_spike':
        p = T // 2
        x[p] += k * sigma[p]
    elif kind == 'B_drift':
        x += np.linspace(0, k * sbar, T)
    elif kind == 'C_hump':
        c = T // 2
        x += k * sbar * np.exp(-((np.arange(T) - c) / (T / 8.0)) ** 2)
    elif kind == 'E_baseline':
        x[:6] += k * sigma[:6].mean()
    return x


def scorers(train_n):
    """Return {name: (fit_scores, score_fn)} trained on normalized training data."""
    out = {}

    emp = EmpiricalCovariance().fit(train_n)
    out['Mahalanobis (empirical)'] = (emp.mahalanobis(train_n),
                                      lambda X: emp.mahalanobis(X))
    try:
        rob = MinCovDet(random_state=0).fit(train_n)
        out['Mahalanobis (robust)'] = (rob.mahalanobis(train_n),
                                       lambda X: rob.mahalanobis(X))
    except Exception:
        pass

    pca = PCA(n_components=min(10, train_n.shape[1])).fit(train_n)
    def pca_err(X):
        return ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(1)
    out['PCA (recon error)'] = (pca_err(train_n), pca_err)

    iso = IsolationForest(random_state=0).fit(train_n)
    out['Isolation Forest'] = (-iso.score_samples(train_n),
                               lambda X: -iso.score_samples(X))

    lof = LocalOutlierFactor(novelty=True).fit(train_n)
    out['LOF'] = (-lof.score_samples(train_n),
                  lambda X: -lof.score_samples(X))

    ocs = OneClassSVM(gamma='scale').fit(train_n)
    out['One-Class SVM'] = (-ocs.score_samples(train_n),
                            lambda X: -ocs.score_samples(X))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--item', default='CRE')
    args = ap.parse_args()

    calc = load_calc(args.item)
    if len(calc) < N_TRAIN + 100:
        raise SystemExit(f'Too few curves for {args.item}: {len(calc)}')

    train = calc[:N_TRAIN]
    clean = calc[N_TRAIN:]                              # held-out clean curves
    sigma = train.std(0)
    lo, rng = minmax_fit(train)
    norm = lambda X: (X - lo) / rng

    train_n = norm(train)
    clean_n = norm(clean)

    # build synthetic anomalies from held-out clean base curves
    base_idx = RNG.choice(len(clean), size=min(N_BASE, len(clean)), replace=False)
    bases = clean[base_idx]
    anomalies = {kind: {k: norm(np.array([inject(b, sigma, kind, k) for b in bases]))
                        for k in SEVERITIES}
                 for kind in ('A_spike', 'B_drift', 'C_hump', 'E_baseline')}

    models = scorers(train_n)

    print(f'\nItem {args.item}: train={len(train)}  clean-eval={len(clean)}  '
          f'window=pts5-38  self-calibrated threshold = P99.5 of training scores\n')
    hdr = f'{"method":26} {"spec":>6} {"recall@1s":>10} {"recall@2s":>10} {"recall@5s":>10} {"recall@8s":>10}'
    print(hdr); print('-' * len(hdr))
    for name, (fit_scores, fn) in models.items():
        thr = np.percentile(fit_scores, 99.5)
        spec = float((fn(clean_n) <= thr).mean())      # clean not flagged
        rec = {}
        for k in SEVERITIES:
            flagged = []
            for kind in anomalies:
                flagged.append(fn(anomalies[kind][k]) > thr)
            rec[k] = float(np.concatenate(flagged).mean())
        print(f'{name:26} {spec:6.3f} {rec[1]:10.3f} {rec[2]:10.3f} {rec[5]:10.3f} {rec[8]:10.3f}')
    print('\n(recall averaged over A_spike/B_drift/C_hump/E_baseline at each severity)\n')


if __name__ == '__main__':
    main()
