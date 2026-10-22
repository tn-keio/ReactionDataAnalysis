# Stage 1z13 (Reviewer #2, Methods-1; M1 change list §10.3 G4): how often would the 2D-CNN rendering range
# [-0.05, 1.05] cut a curve?  The v2 benchmark renders each unclipped, scalar min-max normalized curve with the
# y-axis fixed to [-0.05, 1.05]; points outside it are not drawn.  This counts, exactly as s1a counted values
# outside [0, 1], the curves with at least one point outside [-0.05, 1.05] (same loaders, same constants from
# the 10,000-curve training pool, clip off).  s1a is imported unchanged; only the bounds differ.
# Output: Results/20261021_Rev_M1_RenderRange/{render_frequency.csv, render_summary.csv}
# Run with the TF venv (the pipeline import pulls TensorFlow).  CPU only, read-only on data.

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_common as R          # noqa: E402
M = R.M

LO, HI = -0.05, 1.05
OUT = R.results_dir('20261021_Rev_M1_RenderRange')


def frac(X):
    lo = X < LO; hi = X > HI; any_ = lo | hi
    return dict(n_curves=int(len(X)),
                curves_cut=float(any_.any(1).mean()) if len(X) else np.nan,
                points_cut=float(any_.mean()) if len(X) else np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    R.set_clip(False)
    rows = []
    for gname, g in R.GROUPS.items():
        R.use_group(gname, OUT, 'renderfreq')
        for proc, synth in g['items']:
            ctx = M.load_item(proc, synth)
            meta = ctx['meta']
            sets = {'train_pool': ctx['train_pool'], 'holdout': ctx['holdout'], 'analysis': ctx['analysis']}
            for s, X in sets.items():
                rows.append(dict(item=proc, group=gname, set=s, anomaly_type='', severity=np.nan, **frac(X)))
            at = meta['anomaly_type'].values; sv = meta['severity'].values
            for t in sorted(set(at)):
                for k in sorted(set(sv[at == t])):
                    sel = (at == t) & (sv == k)
                    rows.append(dict(item=proc, group=gname, set='synthetic', anomaly_type=t,
                                     severity=float(k), **frac(ctx['syn'][sel])))
            print(f'{proc:6} ({gname}) done', flush=True)
            del ctx
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'render_frequency.csv'), index=False, encoding='utf-8')
    syn = df[df.set == 'synthetic']
    summ = pd.concat([
        df[df.set != 'synthetic'].groupby('set')['curves_cut'].agg(['median', 'max', 'mean']).reset_index()
          .assign(anomaly_type='', severity=np.nan),
        syn.groupby(['anomaly_type', 'severity'])['curves_cut'].agg(['median', 'max', 'mean']).reset_index()
          .assign(set='synthetic')], ignore_index=True)
    summ.to_csv(os.path.join(OUT, 'render_summary.csv'), index=False, encoding='utf-8')
    print(summ.to_string(), flush=True)


if __name__ == '__main__':
    main()
