# Stage 1z8: intervention test for s1z7.
# s1z7 found that the IP compact-AE holdout P99.5 sits on ~1% of normal curves with a wide within-curve range
# (max - min > 0.3, normalized units), present at the same rate in both windows (~1%), but the post-re-base
# N = 500 training set holds only 4 such curves (max range 0.47; holdout up to 0.92) against 7 (max 0.79) pre.
# Hypothesis: the threshold rise is driven by whether the N training curves cover the amplitude of these curves.
# Test (z = 8, seeds 0-3, unclipped, threshold = holdout P99.5, everything else as s1z2):
#   base    : train_pool[:500]
#   post+W  : post, the 10 widest curves of train_pool[500:] replace the last 10 non-wide curves of the fit part
#   pre-W   : pre, every wide curve (range > 0.3) of train_pool[:500] replaced by the next non-wide curves
# The validation split (last 20%) is left as in base.
# Output: Results/20261014_Rev_M1_IP_ThrDiag/swap.csv
# TF venv.

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_common as R          # noqa: E402
import models_v2 as MV          # noqa: E402
M = R.M

N, Z, SEEDS, W = 500, 8, [0, 1, 2, 3], 0.3
OUT = R.results_dir('20261014_Rev_M1_IP_ThrDiag')
CONF = {'pre': (R.ORIG_LOAD_ITEM, R.GROUPS['main']['syn']),
        'post': (R.GROUPS['m2']['loader'], R.GROUPS['m2']['syn'])}


def rng(X):
    return X.max(1) - X.min(1)


def variants(tag, pool):
    vs = max(1, int(N * 0.2)); fit, val = pool[:N - vs].copy(), pool[N - vs:N]
    out = {'base': (fit, val)}
    wide_fit = rng(fit) > W
    rest = pool[N:]; r_rest = rng(rest)
    if tag == 'post':
        add = rest[np.argsort(r_rest)[::-1][:10]]
        f2 = fit.copy(); repl = np.where(~wide_fit)[0][-10:]; f2[repl] = add
        out['post+W'] = (f2, val)
    else:
        f2 = fit.copy(); repl = np.where(wide_fit)[0]
        f2[repl] = rest[r_rest <= W][:len(repl)]
        out['pre-W'] = (f2, val)
    return out


def main():
    R.set_clip(False)
    rows = []
    for tag, (loader, syn_path) in CONF.items():
        M.SYN = syn_path
        ctx = loader('IP', 'IP'); ho = ctx['holdout']
        at = ctx['meta'].anomaly_type.values; sev = ctx['meta'].severity.values
        for vname, (fit, val) in variants(tag, ctx['train_pool']).items():
            for seed in SEEDS:
                M.set_seeds(seed)
                ae = MV.DenseAE(fit.shape[1], Z)
                ae.train(fit, val, epochs=M.AE_EPOCHS, batch_size=M.BATCH)
                s_ho = ae.score(ho); thr = np.percentile(s_ho, 99.5); s_syn = ae.score(ctx['syn'])
                r = dict(config=tag, variant=vname, seed=seed, n_wide_fit=int((rng(fit) > W).sum()),
                         max_range_fit=float(rng(fit).max()), thr=float(thr),
                         spec_clean=float(1 - (s_syn[at == 'clean'] >= thr).mean()))
                for t in M.POS_TYPES:
                    sel = (at == t) & (sev >= 2)
                    r[f'rec_{t}'] = float((s_syn[sel] >= thr).mean())
                rows.append(r); print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                M.tf.keras.backend.clear_session()
        del ctx
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'swap.csv'), index=False, float_format='%.6g')
    pd.set_option('display.width', 250)
    print(d.groupby(['config', 'variant']).mean(numeric_only=True).drop(columns='seed').round(4).to_string())


if __name__ == '__main__':
    main()
