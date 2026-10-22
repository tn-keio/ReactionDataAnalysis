# Stage 1z6 (Reviewer #2, Methods-1, plan step 2): normalization constants from the N training curves only.
#
# §2.5 says lo/hi come from the 10,000-curve training pool and are fixed across N, which reads oddly next to
# "trained on as few as 500 curves".  Here, for every N, lo/hi are recomputed from train_pool[:N] alone
# (raw calc units, scalar min/max as norm_fit) and EVERY set (train, holdout, synthetic, analysis) is
# re-mapped with them, unclipped.  Everything else is the s1b_classical_rerun.py --clip off pipeline.
# Expected (theory): all six classical detectors are invariant to a common scalar affine map of the inputs
# (Mahalanobis/robust: affine-invariant; PCA: n_components=0.95 and reconstruction-error ranks unchanged;
# LOF: distance ratios; OCSVM gamma='scale': kernel unchanged; iForest: uniform split draws scale with the
# range), so the JSONL should equal Results/20260917_Rev_M1_Classical_noclip up to floating point.
# Also records the share of curves that leave [0, 1] under the N-only constants.
# Output: Results/20261013_Rev_M1_NormFromN/<group>/Reports/comparison.jsonl, out_of_range.csv
# TF venv (rev_common imports the pipeline).

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_common as R          # noqa: E402
M = R.M

CLASSICAL = ['maha', 'maha_robust', 'pca', 'lof', 'iforest', 'ocsvm']
SETS = ('train_pool', 'holdout', 'syn', 'analysis')
ORIG_SCORE_SETS = M.score_sets
_cur = {}
_oor = []


def score_sets_from_N(method, ctx, N, seed):
    lo, hi = ctx['norm']
    raw = {k: ctx[k] * (hi - lo + 1e-8) + lo for k in SETS}           # back to calc units
    loN, hiN = float(raw['train_pool'][:N].min()), float(raw['train_pool'][:N].max())
    c2 = dict(ctx)
    for k in SETS:
        c2[k] = (raw[k] - loN) / (hiN - loN + 1e-8)
    key = (_cur.get('item'), N)
    if key not in {(r['item'], r['N']) for r in _oor}:
        rec = dict(item=key[0], N=N, lo_pool=lo, hi_pool=hi, lo_N=loN, hi_N=hiN)
        for k in ('holdout', 'syn', 'analysis'):
            x = c2[k]; rec[f'oor_curves_{k}'] = float(((x < 0) | (x > 1)).any(1).mean())
        _oor.append(rec)
    return ORIG_SCORE_SETS(method, c2, N, seed)


def main():
    R.set_clip(False)
    M.score_sets = score_sets_from_N
    M.METHODS_OUT = list(CLASSICAL)
    out = R.results_dir('20261013_Rev_M1_NormFromN')
    for gname in R.GROUPS:
        g = R.use_group(gname, out, 'normfromN')
        base_loader = g['loader']

        def loader(proc, synth, _b=base_loader):
            _cur['item'] = proc
            return _b(proc, synth)
        M.load_item = loader
        print(f'=== group={gname} items={[i[0] for i in g["items"]]}', flush=True)
        M.run(g['items'], CLASSICAL, do_list=False)
    pd.DataFrame(_oor).to_csv(os.path.join(out, 'out_of_range.csv'), index=False, float_format='%.6g')
    # ---- compare with the fixed-constant run
    ref = os.path.join(R.results_dir('20260917_Rev_M1_Classical_noclip'))
    rows = []
    for gname in R.GROUPS:
        def jl(p):
            return {(d['item'], d['method'], d['N']): d for d in
                    (json.loads(l) for l in open(p, encoding='utf-8') if l.strip())}
        a = jl(os.path.join(ref, gname, 'Reports', 'comparison.jsonl'))
        b = jl(os.path.join(out, gname, 'Reports', 'comparison.jsonl'))
        for k, rb in b.items():
            ra = a.get(k)
            if ra is None:
                continue
            ba, bb = ra['budgets']['99.5'], rb['budgets']['99.5']
            d = {f: abs(ba[f]['mean'] - bb[f]['mean']) for f in ba if f in bb}
            rows.append(dict(item=k[0], method=k[1], N=k[2], max_abs_diff=max(d.values()),
                             d_recall_main=bb['recall_main']['mean'] - ba['recall_main']['mean'],
                             d_spec=bb['specificity_clean']['mean'] - ba['specificity_clean']['mean'],
                             d_fp05=rb['recall_at_fpr']['0.5']['mean'] - ra['recall_at_fpr']['0.5']['mean']))
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(out, 'diff_vs_fixed_constants.csv'), index=False, float_format='%.6g')
    print(f'compared {len(c)} records')
    print(c.groupby('method')[['max_abs_diff']].max().to_string())
    print(c.groupby('N')[['max_abs_diff']].max().to_string())


if __name__ == '__main__':
    main()
