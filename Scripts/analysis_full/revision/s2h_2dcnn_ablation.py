# Stage 2h: why did 2D-CNN -> iForest change from 0.579 (submitted) to 0.574 (v2)?
#
# Two things changed at once in v2: the input is no longer clipped to [0,1], and the rendering y-range
# was widened from [0,1] to [-0.05,1.05].  This script separates them with ONE additional cell:
#     submitted : clipped   input, y-range [0, 1]          (Results/20260717.. + re-based runs)
#     ablation  : clipped   input, y-range [-0.05, 1.05]   (computed here)
#     v2        : unclipped input, y-range [-0.05, 1.05]   (Results/20260917_Rev_v2Benchmark)
#   ablation - submitted = effect of the y-range;  v2 - ablation = effect of unclipping.
# Items: IgG and IgM (the two largest v2-vs-submitted swings, +0.209 and -0.115) plus CRE and ALB
# (typical items), so the selection is not only the extremes.  Seeds 0-3, N = 500.
# It also runs a paired test over all 21 items on the v2-vs-submitted difference, to show whether the
# panel-mean change is distinguishable from IsolationForest seed noise.
# Output: Results/20260918_Rev_M1_2DCNN_ablation/{ablation.csv, paired_test.txt}
# TF venv.  Rendering ~13 min per item (27k images) -> ~50 min total.

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2b_benchmark_v2 as S2B      # noqa: E402  (pipeline + v2 patches)
R = S2B.R
M = R.M
sys.path.insert(0, os.path.join(R.BASE_DIR, 'ForPaper'))
from _common import panel as P      # noqa: E402

OUT = R.results_dir('20260918_Rev_M1_2DCNN_ablation')
V2 = R.results_dir('20260917_Rev_v2Benchmark')
ITEMS = {'IgG': 'main', 'IgM': 'main', 'CRE': 'main', 'ALB': 'm2'}
SEEDS = [0, 1, 2, 3]
N = 500


def v2_records():
    recs = {}
    for g in ('main', 'ca', 'm2'):
        for l in open(os.path.join(V2, g, 'Reports', 'comparison.jsonl'), encoding='utf-8'):
            if l.strip():
                d = json.loads(l)
                recs[(d['item'], d['method'], d['N'])] = d
    return recs


def paired_test():
    """v2 vs submitted, 2D-CNN -> iForest, over the 21-item panel."""
    new, old = v2_records(), P.load_metrics(21)
    rows = []
    for it in P.proc_names(21):
        b = new[(it, 'twod_only', N)]['budgets']['99.5']
        o = old[(it, 'twod_only', N)]['budgets']['99.5']
        rows.append(dict(item=it, v2=b['recall_main']['mean'], v2_ci=b['recall_main']['ci95'],
                         submitted=o['recall_main']['mean'], submitted_ci=o['recall_main']['ci95']))
    d = pd.DataFrame(rows)
    d['delta'] = d.v2 - d.submitted
    w = wilcoxon(d.v2, d.submitted, alternative='two-sided', zero_method='wilcox')
    txt = (f"2D-CNN -> iForest, N={N}, 21 assays, v2 (unclipped, y-range [-0.05,1.05]) vs submitted\n"
           f"  panel mean: v2 {d.v2.mean():.4f} vs submitted {d.submitted.mean():.4f} "
           f"(delta {d.delta.mean():+.4f})\n"
           f"  per-assay delta: {int((d.delta < 0).sum())} worse / {int((d.delta > 0).sum())} better, "
           f"median {d.delta.median():+.4f}, range {d.delta.min():+.3f}..{d.delta.max():+.3f}\n"
           f"  mean 95% CI over the 4 IsolationForest seeds: v2 {d.v2_ci.mean():.4f}, "
           f"submitted {d.submitted_ci.mean():.4f}\n"
           f"  paired Wilcoxon (two-sided): W={w.statistic}, p={w.pvalue:.3f}\n")
    open(os.path.join(OUT, 'paired_test.txt'), 'w', encoding='utf-8').write(txt)
    d.to_csv(os.path.join(OUT, 'per_item_twod_v2_vs_submitted.csv'), index=False)
    print(txt, flush=True)


def ablation():
    S2B.install()                      # unclipped + v2 render/metrics/score_sets
    R.set_clip(True)                   # ablation cell: CLIPPED input, widened y-range
    recs = v2_records()
    old = P.load_metrics(21)
    rows = []
    for item, gname in ITEMS.items():
        g = R.use_group(gname, OUT, 'ablation_clip_wideylim')
        os.makedirs(M.CACHE, exist_ok=True)     # use_group only sets the path; feats_cached needs the dir
        synth = dict(g['items'])[item]
        M.load_item = S2B.make_loader(gname, g['loader'])
        ctx = M.load_item(item, synth)
        for nm, arr in (('train', ctx['train_pool']), ('holdout', ctx['holdout']),
                        ('analysis', ctx['analysis']), ('syn', ctx['syn'])):
            ctx[f'feat_{nm}'] = M.feats_cached(item, nm, arr)
        vals = []
        for seed in SEEDS:
            sc, _, _ = M.score_sets('twod_only', ctx, N, seed)
            thr = float(np.percentile(sc['holdout'], 99.5))
            m = M.metrics_at(sc['syn'] >= thr, ctx['meta'], sc['holdout'], thr)
            vals.append(m['recall_main'])
        mean = float(np.mean(vals)); ci = float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
        rows.append(dict(item=item, cell='clip+wide_ylim', recall_main=mean, ci95=ci,
                         v2_unclip_wide=recs[(item, 'twod_only', N)]['budgets']['99.5']['recall_main']['mean'],
                         submitted_clip_narrow=old[(item, 'twod_only', N)]['budgets']['99.5']['recall_main']['mean']))
        print(f'  {item}: clip+wide {mean:.3f}±{ci:.3f} | v2 {rows[-1]["v2_unclip_wide"]:.3f} | '
              f'submitted {rows[-1]["submitted_clip_narrow"]:.3f}', flush=True)
        del ctx
        M.tf.keras.backend.clear_session()
    d = pd.DataFrame(rows)
    d['effect_ylim'] = d.recall_main - d.submitted_clip_narrow      # clipped, y-range only
    d['effect_unclip'] = d.v2_unclip_wide - d.recall_main           # unclipping, at the wide y-range
    d.to_csv(os.path.join(OUT, 'ablation.csv'), index=False)
    print(d.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-render', action='store_true', help='paired test only')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    paired_test()
    if not a.skip_render:
        ablation()


if __name__ == '__main__':
    main()
