# Stage 1z7: why does the compact-AE holdout P99.5 threshold at IP
# rise ~4.8x after the module-2 re-base (s1z2: z = 8, 0.00080 -> 0.00383)?  Diagnostic only.
#
# Same set-up as s1z2_ip_rebase_diagnostic.py (unclipped, N = 500, v2 DenseAE, seeds 0-3), z = 8 and 16.
# Per configuration x z x seed:
#   tail      : score quantiles (P50/P90/P99/P99.5/max) on train-fit, validation, holdout, and the unseen
#               in-window normals train_pool[N:]  -> is the heavy tail specific to the holdout, or to the window?
#   decompose : absolute median score per anomaly type (k >= 2)  -> threshold rise vs anomaly-score fall
#   alt thr   : recall with thresholds from validation P99.5, holdout P99, and unseen-normal P99.5
#   who       : the 10 holdout curves above P99.5 (holdout position, PC1-3 scores in train-SD units, PCA-95%
#               residual and Mahalanobis percentile within the holdout); overlap of that set across seeds
# Output: Results/20261014_Rev_M1_IP_ThrDiag/{tail.csv, alt_thr.csv, top_holdout.csv, overlap.csv}
# TF venv.

import os
import sys
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_common as R          # noqa: E402
import models_v2 as MV          # noqa: E402
M = R.M

N = 500
ZS = [8, 16]
SEEDS = [0, 1, 2, 3]
QS = [50, 90, 99, 99.5, 100]
OUT = R.results_dir('20261014_Rev_M1_IP_ThrDiag')
CONF = {'pre': (R.ORIG_LOAD_ITEM, R.GROUPS['main']['syn']),
        'post': (R.GROUPS['m2']['loader'], R.GROUPS['m2']['syn'])}


def main():
    os.makedirs(OUT, exist_ok=True)
    R.set_clip(False)
    tail, alt, top, sets = [], [], [], {}
    for tag, (loader, syn_path) in CONF.items():
        M.SYN = syn_path
        ctx = loader('IP', 'IP')
        Xtr = ctx['train_pool'][:N]; unseen = ctx['train_pool'][N:]; ho = ctx['holdout']
        meta = ctx['meta']; at = meta.anomaly_type.values; sev = meta.severity.values
        vs = max(1, int(N * 0.2)); Xfit, Xval = Xtr[:-vs], Xtr[-vs:]
        # classical descriptors of holdout curves (fit on the N training curves)
        mu = Xtr.mean(0); C = np.cov(Xtr.T); ev, V = np.linalg.eigh(C); ev, V = ev[::-1], V[:, ::-1]
        q = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)
        pcs = (ho - mu) @ V[:, :3] / np.sqrt(ev[:3])
        resid = (((ho - mu) - ((ho - mu) @ V[:, :q]) @ V[:, :q].T) ** 2).mean(1)
        P = np.linalg.pinv(C); md2 = np.einsum('ij,jk,ik->i', ho - mu, P, ho - mu)
        pct = lambda v: pd.Series(v).rank(pct=True).values
        resid_p, md_p = pct(resid), pct(md2)
        for z in ZS:
            for seed in SEEDS:
                M.set_seeds(seed)
                ae = MV.DenseAE(Xtr.shape[1], z)
                ae.train(Xfit, Xval, epochs=M.AE_EPOCHS, batch_size=M.BATCH)
                sc = {'fit': ae.score(Xfit), 'val': ae.score(Xval), 'holdout': ae.score(ho),
                      'unseen': ae.score(unseen)}
                s_syn = ae.score(ctx['syn'])
                for k, v in sc.items():
                    r = dict(config=tag, z=z, seed=seed, set=k, n=len(v))
                    r.update({f'P{qq}': float(np.percentile(v, qq)) for qq in QS})
                    tail.append(r)
                thr = {'holdout_P99.5': np.percentile(sc['holdout'], 99.5),
                       'holdout_P99': np.percentile(sc['holdout'], 99),
                       'val_P99.5': np.percentile(sc['val'], 99.5),
                       'unseen_P99.5': np.percentile(sc['unseen'], 99.5)}
                for tn, tv in thr.items():
                    r = dict(config=tag, z=z, seed=seed, thr_rule=tn, thr=float(tv),
                             spec_clean=float(1 - (s_syn[at == 'clean'] >= tv).mean()),
                             fpr_unseen=float((sc['unseen'] >= tv).mean()))
                    for t in M.POS_TYPES:
                        sel = (at == t) & (sev >= 2)
                        r[f'rec_{t}'] = float((s_syn[sel] >= tv).mean())
                        r[f'med_{t}'] = float(np.median(s_syn[sel]))
                    alt.append(r)
                idx = np.argsort(sc['holdout'])[::-1][:10]
                sets[(tag, z, seed)] = set(idx.tolist())
                for rk, i in enumerate(idx):
                    top.append(dict(config=tag, z=z, seed=seed, rank=rk + 1, ho_pos=int(i),
                                    score=float(sc['holdout'][i]), score_over_ho_med=float(sc['holdout'][i] / np.median(sc['holdout'])),
                                    pc1=float(pcs[i, 0]), pc2=float(pcs[i, 1]), pc3=float(pcs[i, 2]),
                                    resid_pct=float(resid_p[i]), md2_pct=float(md_p[i]),
                                    xmin=float(ho[i].min()), xmax=float(ho[i].max())))
                print(tag, z, seed, {k: round(float(v), 6) for k, v in thr.items()}, flush=True)
                M.tf.keras.backend.clear_session()
        del ctx
    ov = []
    for (tag, z) in itertools.product(CONF, ZS):
        for a, b in itertools.combinations(SEEDS, 2):
            ov.append(dict(config=tag, z=z, seed_a=a, seed_b=b,
                           shared_top10=len(sets[(tag, z, a)] & sets[(tag, z, b)])))
    for name, rows in (('tail', tail), ('alt_thr', alt), ('top_holdout', top), ('overlap', ov)):
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f'{name}.csv'), index=False, float_format='%.6g')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    t = pd.DataFrame(tail).groupby(['config', 'z', 'set']).mean(numeric_only=True).drop(columns='seed')
    print(t.to_string())
    a = pd.DataFrame(alt).groupby(['config', 'z', 'thr_rule']).mean(numeric_only=True).drop(columns='seed')
    print(a.round(4).to_string())
    print(pd.DataFrame(ov).groupby(['config', 'z']).shared_top10.mean())
    tp = pd.DataFrame(top)
    print(tp.groupby(['config', 'z'])[['score_over_ho_med', 'pc1', 'pc2', 'pc3', 'resid_pct', 'md2_pct']].median().round(3))


if __name__ == '__main__':
    main()
