# Stage 1x (Reviewer #2, Methods-1 scope extension): S10b.3 / Table S5-3 -- the module-2 items (Ca, ALB, IP)
# on their ORIGINAL (un-re-based) batteries and loaders, classical detectors, with and without np.clip.
#
# Same pipeline as s1b_classical_rerun.py; only a fourth loader group is registered at run time:
# the 'main' loader (ORIG_LOAD_ITEM) and the main p5_38 battery applied to Ca/ALB/IP, i.e. exactly the
# configuration before the 2026-07-18 re-base.  --clip on is the repro check against the submitted
# pre-rebase values (Ca specificity 0.65, ALB 0.88 for empirical Mahalanobis; Table S5-3).
# Output: Results/20261013_Rev_M1_PreRebase_{clip|noclip}/prerebase/Reports/comparison.jsonl
# Run with the TF venv (the pipeline import pulls TensorFlow).

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_common as R          # noqa: E402
M = R.M

CLASSICAL = ['maha', 'maha_robust', 'pca', 'lof', 'iforest', 'ocsvm']
R.GROUPS['prerebase'] = dict(items=[('Ca', 'CA'), ('ALB', 'ALB'), ('IP', 'IP')],
                             syn=R.GROUPS['main']['syn'], loader=R.ORIG_LOAD_ITEM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', choices=['on', 'off'], required=True)
    ap.add_argument('--methods', nargs='+', default=CLASSICAL)
    a = ap.parse_args()
    clip = a.clip == 'on'
    R.set_clip(clip)
    M.METHODS_OUT = list(a.methods)
    out = R.results_dir('20261013_Rev_M1_PreRebase_' + ('clip' if clip else 'noclip'))
    g = R.use_group('prerebase', out, 'prerebase')
    print(f'=== prerebase clip={clip} items={[i[0] for i in g["items"]]} methods={a.methods}', flush=True)
    M.run(g['items'], a.methods, do_list=False)


if __name__ == '__main__':
    main()
