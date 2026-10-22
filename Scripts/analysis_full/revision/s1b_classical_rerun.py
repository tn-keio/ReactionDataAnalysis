# Stage 1b (Reviewer #2, Methods-1): re-run the NON-deep detectors of the 21-item method comparison
# with and without the np.clip in the min-max normalization.
#
#   --clip on  : reproduction check -- must reproduce the submitted JSONL values exactly
#                (validates that the three loader groups reproduce Table 1's sources).
#   --clip off : the unclipped comparison requested by the reviewer.
#
# Deep-learning detectors are deferred to the Stage-2/3 v2 benchmark (their decoders use a sigmoid
# output, which cannot represent unclipped out-of-range targets; see CodeFix_Register.md).
# Output: Results/20260917_Rev_M1_Classical_{clip|noclip}/<group>/Reports/comparison.jsonl
# Resume-aware (the pipeline skips (item, method, N) already present in the JSONL).

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_common as R          # noqa: E402
M = R.M

CLASSICAL = ['maha', 'maha_robust', 'pca', 'lof', 'iforest', 'ocsvm']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', choices=['on', 'off'], required=True)
    ap.add_argument('--methods', nargs='+', default=CLASSICAL)
    ap.add_argument('--groups', nargs='+', default=list(R.GROUPS))
    ap.add_argument('--items', nargs='+', default=None)
    a = ap.parse_args()
    assert all(m in CLASSICAL for m in a.methods), 'deep methods are not run here'
    clip = a.clip == 'on'
    R.set_clip(clip)
    # run()'s item-level resume test iterates the module-global METHODS_OUT (all 11 methods), so with a
    # subset it never skips.  Narrow it (only used there and in list/image writing, which is off here).
    M.METHODS_OUT = list(a.methods)
    out = R.results_dir('20260917_Rev_M1_Classical_' + ('clip' if clip else 'noclip'))
    for gname in a.groups:
        g = R.use_group(gname, out, 'classical')
        items = [it for it in g['items'] if a.items is None or it[0] in a.items]
        if not items:
            continue
        print(f'=== group={gname} clip={clip} items={[i[0] for i in items]} methods={a.methods}', flush=True)
        M.run(items, a.methods, do_list=False)


if __name__ == '__main__':
    main()
