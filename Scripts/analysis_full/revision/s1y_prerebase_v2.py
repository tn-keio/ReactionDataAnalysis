# Stage 1y (Reviewer #2, Methods-1 scope extension): Table S5-3 -- the three module-2 items (Ca, ALB, IP)
# on their ORIGINAL (un-re-based) batteries and loaders, run through the v2 benchmark unchanged.
#
# Reuses s2b_benchmark_v2.py as is (install(): unclipped input, linear AE/CNN outputs, 2D y-range
# [-0.05, 1.05], v2 metrics, compact AEs).  Only a fourth loader group is registered at run time:
#   loader  = rev_common.ORIG_LOAD_ITEM (full history, as before the 2026-07-18 re-base)
#   battery = endpoint_synthetic_anomalies_p5_38.parquet + rev2026_extra_main.parquet
#             (the extra rows of the main battery already cover CA, ALB and IP)
# The 18 non-re-based items are not re-run: their v2 values are the "before" values as well.
# The lr3e4 sensitivity variants are omitted (run on 3 items only in v2; not in Table 1).
# Classical methods are included; they must equal s1x_prerebase_classical.py --clip off (cross-check).
# Output: Results/20261013_Rev_M1_PreRebase_v2/prerebase/Reports/{comparison,dae_selection}.jsonl
# TF venv.

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2b_benchmark_v2 as B2   # noqa: E402  (imports rev_common -> pipeline + TF)
R, M = B2.R, B2.M

R.GROUPS['prerebase'] = dict(items=[('Ca', 'CA'), ('ALB', 'ALB'), ('IP', 'IP')],
                             syn=R.GROUPS['main']['syn'], loader=R.ORIG_LOAD_ITEM)
B2.EXTRA['prerebase'] = B2.EXTRA['main']
METHODS = [m for m in B2.ALL_METHODS if 'lr3e4' not in m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=None, help='one item per process (TF memory, as run_v2_full.ps1)')
    a = ap.parse_args()
    B2.install(); B2.self_test()
    M.METHODS_OUT = list(METHODS)        # resume checks without the omitted lr3e4 variants
    out = R.results_dir('20261013_Rev_M1_PreRebase_v2')
    g = R.use_group('prerebase', out, 'v2prerebase')
    items = [it for it in g['items'] if a.items is None or it[0] in a.items]
    M.load_item = B2.make_loader('prerebase', g['loader'])
    os.makedirs(M.REPORTS, exist_ok=True)
    B2.SEL_LOG = os.path.join(M.REPORTS, 'dae_selection.jsonl')
    print(f'=== v2 prerebase items={[i[0] for i in items]} methods={METHODS} out={M.OUT}', flush=True)
    t0 = time.time()
    M.run(items, METHODS, do_list=False)
    print(f'=== DONE in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
