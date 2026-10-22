# Stage 1a (Reviewer #2, Methods-1): how often are clean and perturbed curves clipped?
#
# Loads every 21-panel item exactly as the submitted method comparison did (same loaders, same
# scalar min-max constants from the 10,000-sample training pool), but with the clip removed, and
# counts values outside [0, 1] -- i.e. the values np.clip would have altered.
# Output: Results/20260917_Rev_M1_ClipFrequency/{clip_frequency.csv, clip_summary.csv}
# Run with the TF venv (the pipeline import pulls TensorFlow).  CPU only, read-only on data.

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_common as R          # noqa: E402
M = R.M

OUT = R.results_dir('20260917_Rev_M1_ClipFrequency')


def frac(X):
    lo = X < 0.0; hi = X > 1.0; any_ = lo | hi
    return dict(n_curves=int(len(X)),
                curves_clipped=float(any_.any(1).mean()) if len(X) else np.nan,
                curves_below0=float(lo.any(1).mean()) if len(X) else np.nan,
                curves_above1=float(hi.any(1).mean()) if len(X) else np.nan,
                points_clipped=float(any_.mean()) if len(X) else np.nan,
                max_excess=float(np.max(np.maximum(X - 1.0, -X))) if len(X) else np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    R.set_clip(False)
    rows = []
    for gname, g in R.GROUPS.items():
        R.use_group(gname, OUT, 'clipfreq')
        for proc, synth in g['items']:
            ctx = M.load_item(proc, synth)
            meta = ctx['meta']
            sets = {'train_pool': ctx['train_pool'], 'train_N500': ctx['train_pool'][:500],
                    'holdout': ctx['holdout'], 'analysis': ctx['analysis']}
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
    df.to_csv(os.path.join(OUT, 'clip_frequency.csv'), index=False, encoding='utf-8')
    # Summary: per set/type, median and max over items of the curve-level clipped fraction
    syn = df[df.set == 'synthetic']
    summ = pd.concat([
        df[df.set != 'synthetic'].groupby('set')['curves_clipped'].agg(['median', 'max', 'mean']).reset_index()
          .assign(anomaly_type='', severity=np.nan),
        syn.groupby(['anomaly_type', 'severity'])['curves_clipped'].agg(['median', 'max', 'mean']).reset_index()
          .assign(set='synthetic')], ignore_index=True)
    summ.to_csv(os.path.join(OUT, 'clip_summary.csv'), index=False, encoding='utf-8')
    print(summ.to_string(), flush=True)


if __name__ == '__main__':
    main()
