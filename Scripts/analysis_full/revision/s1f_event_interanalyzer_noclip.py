# Stage 1f (Reviewer #2, Methods-1 scope extension): S9 (maintenance / reagent-lot robustness) and
# S4 (inter-analyzer transferability) re-computed WITHOUT np.clip.
#
# Reuses event_robustness_analysis.py and inter_analyzer_analysis.py unchanged: their module-level
# maha_eval / maha_fit (the only places the clip occurs) are replaced by unclipped copies, and
# OUT / RESULTS are redirected to new folders.  Everything else (panel, events, placebo seed, blocks)
# is the submitted code path.
# Output: Results/20260917_Rev_M1_S9S4_noclip/{event,inter}/...
# py3.14 (no TensorFlow).  Both underlying scripts are resumable per item.

import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import event_robustness_analysis as EV      # noqa: E402
import inter_analyzer_analysis as IA        # noqa: E402

OUT = os.path.join(EV.BASE, 'Results', '20260917_Rev_M1_S9S4_noclip')


def ev_maha_eval_noclip(train, pre, post):
    """Copy of EV.maha_eval without np.clip."""
    lo, hi = train.min(0), train.max(0)
    def nz(X): return (X - lo) / (hi - lo + 1e-8)
    ntr = nz(train); mu = ntr.mean(0); P = np.linalg.pinv(np.cov(ntr.T))
    dtr = np.einsum('ij,jk,ik->i', ntr - mu, P, ntr - mu)
    thr = np.percentile(dtr, EV.BUDGET)
    def rate(X):
        nx = nz(X); d = np.einsum('ij,jk,ik->i', nx - mu, P, nx - mu)
        return float((d >= thr).mean())
    pre_rate = rate(pre); post_rate = rate(post)
    rmu, rsd = train.mean(0), train.std(0) + 1e-9
    onset = int(np.argmax(np.abs(np.diff(rmu))))
    dev = np.abs(post.mean(0) - rmu) / rsd
    bmask = np.arange(len(EV.PTS)) <= onset
    return pre_rate, post_rate, float(dev[bmask].mean()), float(dev[~bmask].mean()), EV.PTS[onset]


def ia_maha_fit_noclip(train):
    """Copy of IA.maha_fit without np.clip."""
    lo, hi = train.min(0), train.max(0)
    def nz(X): return (X - lo) / (hi - lo + 1e-8)
    ntr = nz(train); mu = ntr.mean(0); P = np.linalg.pinv(np.cov(ntr.T))
    dtr = np.einsum('ij,jk,ik->i', ntr - mu, P, ntr - mu)
    thr = np.percentile(dtr, IA.BUDGET)
    rmu, rsd = train.mean(0), train.std(0) + 1e-9
    onset = int(np.argmax(np.abs(np.diff(rmu))))
    return dict(nz=nz, mu=mu, P=P, thr=thr, rmu=rmu, rsd=rsd, onset=onset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', nargs='+', default=['event', 'inter'])
    a = ap.parse_args()
    if 'event' in a.which:
        EV.maha_eval = ev_maha_eval_noclip
        EV.OUT = os.path.join(OUT, 'event')
        EV.RESULTS = os.path.join(EV.OUT, os.path.basename(EV.RESULTS))
        os.makedirs(EV.OUT, exist_ok=True)
        print('===== S9 event robustness (noclip) ->', EV.OUT, flush=True)
        EV.main()
    if 'inter' in a.which:
        IA.maha_fit = ia_maha_fit_noclip
        IA.OUT = os.path.join(OUT, 'inter')
        IA.RESULTS = os.path.join(IA.OUT, os.path.basename(IA.RESULTS))
        os.makedirs(IA.OUT, exist_ok=True)
        print('===== S4 inter-analyzer (noclip) ->', IA.OUT, flush=True)
        IA.main()


if __name__ == '__main__':
    main()
