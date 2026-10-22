# Stage 1e: the real Cu interference case (S10, Fig. 5)
# re-computed with and without np.clip in the per-point min-max normalization.
#
# Reuses real_anomaly_cu_validation.py unchanged: only its module-global maha_fit, OUT and FIGDIR are
# replaced at run time.  FIGDIR is redirected so the submitted ForPaper figures are NOT overwritten.
# Output: Results/20260917_Rev_M1_CuCase/{clip,noclip}/detection_result.json (+ figures), clip_check.json
# py3.14 (no TensorFlow).  The json contains the stored result value of the case: local use only.

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import real_anomaly_cu_validation as C      # noqa: E402

OUT = os.path.join(C.BASE, 'Results', '20260917_Rev_M1_CuCase')
ORIG_FIT = C.maha_fit


def maha_fit_noclip(train):
    """Identical to C.maha_fit except that the normalization is not clipped."""
    lo, hi = train.min(0), train.max(0)
    nz = lambda X: (X - lo) / (hi - lo + 1e-8)
    ntr = nz(train); mu = ntr.mean(0); P = np.linalg.pinv(np.cov(ntr.T))
    dtr = np.einsum('ij,jk,ik->i', ntr - mu, P, ntr - mu)
    thr = np.percentile(dtr, C.BUDGET)
    return dict(nz=nz, mu=mu, P=P, thr=thr, dtr=dtr, tr_mean=train.mean(0), tr_sd=train.std(0))


def clip_check():
    """How many of the anomaly's window-5-38 points lie outside the training min/max (would be clipped)?"""
    import pandas as pd
    df = pd.read_parquet(C.PROC, columns=['SystemName', 'S_ID', 'S_Date'] + C.RAS + C.RAM,
                         filters=[('A_Name', '==', 'Cu')])
    df['d'] = pd.to_datetime(df['S_Date'])
    df['sys'] = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    anom = C.calc38(df[df['S_ID'] == C.ANOM_SID])[0]
    pre = df[df['d'] < C.ANOM_DATE]
    out = {}
    cols = C.window_cols(5, 38)
    for label, tr in [('single_#3', pre[pre['sys'] == '#3']), ('pool_#2+#3', pre[pre['sys'].isin(['#2', '#3'])])]:
        t = C.calc38(tr.sort_values('d').iloc[-C.N_TRAIN:])[:, cols]
        a = anom[cols]; lo, hi = t.min(0), t.max(0)
        outside = (a < lo) | (a > hi)
        out[label] = dict(points_outside=int(outside.sum()), n_points=len(cols),
                          outside_points=[5 + int(i) for i in np.where(outside)[0]])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    for tag, fn in [('clip', ORIG_FIT), ('noclip', maha_fit_noclip)]:
        C.maha_fit = fn
        C.OUT = os.path.join(OUT, tag)
        C.FIGDIR = os.path.join(OUT, tag, 'Figures')
        print(f'===== {tag} =====', flush=True)
        C.main()
    json.dump(clip_check(), open(os.path.join(OUT, 'clip_check.json'), 'w'), indent=2)
    print(open(os.path.join(OUT, 'clip_check.json')).read())


if __name__ == '__main__':
    main()
