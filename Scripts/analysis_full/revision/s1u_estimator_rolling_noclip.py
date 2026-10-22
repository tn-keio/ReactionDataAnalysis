# Stage 1u: S3.4 "Confirmation under a rolling protocol"
# (worst-item recall E 0.839 / R 0.468 / LW 0.115; Fig. S2(b)-(d)) re-computed with and without np.clip.
#
# Reuses run_estimator_selection.py unchanged: only its module-global apply_norm, norm_fit and OUT are
# replaced at run time.  --clip on reproduces the submitted Results/20260716_EstimatorSelection_E_vs_R
# (repro check); --clip off is the revision value.  The wrappers also count how many curves leave
# [0, 1], split by role (train window / real scoring block / synthetic battery), using the call order
# inside run_item: norm_fit(W) -> apply_norm(W) -> apply_norm(realN) -> apply_norm(battery) x (seeds+1).
# Output: Results/20261012_Rev_M1_EstimatorRolling/{clip,noclip}/ (+ clip_frequency.csv)
# py3.14 (no TensorFlow).

import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import run_estimator_selection as ES      # noqa: E402

OUT = os.path.join(ES.BASE, 'Results', '20261012_Rev_M1_EstimatorRolling')
ORIG_NORM_FIT = ES.norm_fit
ORIG_APPLY_NORM = ES.apply_norm

_state = {'calls': 0}
_counts = {}                      # role -> [n_curves, n_curves_outside]


def norm_fit_counting(a):
    _state['calls'] = 0
    return ORIG_NORM_FIT(a)


def make_apply_norm(clip):
    def apply_norm(a, lo, hi):
        x = (a - lo) / (hi - lo + 1e-8)
        role = {0: 'train', 1: 'real'}.get(_state['calls'], 'synthetic')
        _state['calls'] += 1
        out = ((x < 0.0) | (x > 1.0)).any(1)
        c = _counts.setdefault(role, [0, 0]); c[0] += len(x); c[1] += int(out.sum())
        return np.clip(x, 0.0, 1.0) if clip else x
    return apply_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', choices=['on', 'off'], required=True)
    args = ap.parse_args()
    clip = args.clip == 'on'
    ES.norm_fit = norm_fit_counting
    ES.apply_norm = make_apply_norm(clip)
    ES.OUT = os.path.join(OUT, 'clip' if clip else 'noclip')
    ES.main()
    rows = [dict(role=r, n_curves=n, n_outside=k, frac_outside=k / n if n else np.nan)
            for r, (n, k) in sorted(_counts.items())]
    pd.DataFrame(rows).to_csv(os.path.join(ES.OUT, 'clip_frequency.csv'), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
