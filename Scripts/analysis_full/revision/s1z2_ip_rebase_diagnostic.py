# Stage 1z2 (Reviewer #2, Methods-1 scope extension): why does the module-2 re-base lower the compact-AE
# worst-item recall at IP from 0.293 (pre) to 0.024 (post)?  Diagnostic only; no table value is produced.
#
# Both configurations unclipped (rev_common.set_clip(False)), N = 500, the v2 DenseAE (s2b_benchmark_v2
# path: MV.DenseAE, same epochs/batch/validation split/seeds), threshold = clean-holdout P99.5:
#   pre  : rev_common.ORIG_LOAD_ITEM  + endpoint_synthetic_anomalies_p5_38.parquet        (full history)
#   post : rerun_m2_clean.clean_load_item + endpoint_synthetic_anomalies_p5_38_m2clean.parquet
#          (analyzer #2, 2021-09-01..2022-04-30)
# Reported per configuration:
#   data      : training-pool spread, PCA spectrum (N = 500), clean-battery shift vs holdout
#   amplitude : sigbar (max |B_drift k=1 - clean| per base curve), in units of the training per-point SD
#   subspace  : share of each anomaly's perturbation energy inside the 95%-variance PCA subspace of the
#               training curves (what a low-dimensional reconstructor can absorb)
#   DAE z     : holdout P99.5, clean-battery scores, per-type recall (k >= 2) and median score/threshold
# Output: Results/20261013_Rev_M1_IP_RebaseDiag/{summary.csv, dae.csv}
# TF venv.

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_common as R          # noqa: E402
import models_v2 as MV          # noqa: E402
M = R.M

N = 500
ZS = [8, 16]
SEEDS = [0, 1, 2, 3]
OUT = R.results_dir('20261013_Rev_M1_IP_RebaseDiag')
CONF = {'pre': (R.ORIG_LOAD_ITEM, R.GROUPS['main']['syn']),
        'post': (R.GROUPS['m2']['loader'], R.GROUPS['m2']['syn'])}


def perturbations(ctx):
    """delta = anomalous - clean curve of the same base_id (normalized units), k >= 2 positives."""
    meta, syn = ctx['meta'], ctx['syn']
    clean = {b: syn[i] for i, (t, b) in enumerate(zip(meta.anomaly_type, meta.base_id)) if t == 'clean'}
    out = {}
    for t in M.POS_TYPES + ['B_drift_k1']:
        tt, sel = (t, (meta.anomaly_type == t) & (meta.severity >= 2)) if t != 'B_drift_k1' else \
                  ('B_drift', (meta.anomaly_type == 'B_drift') & np.isclose(meta.severity, 1))
        idx = np.where(sel.values)[0]
        out[t] = np.array([syn[i] - clean[meta.base_id[i]] for i in idx if meta.base_id[i] in clean])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    R.set_clip(False)
    summ, rows = [], []
    for tag, (loader, syn_path) in CONF.items():
        M.SYN = syn_path
        ctx = loader('IP', 'IP')
        Xtr = ctx['train_pool'][:N]; ho = ctx['holdout']; meta = ctx['meta']; at = meta.anomaly_type.values
        sd = Xtr.std(0); mu = Xtr.mean(0)
        ev, V = np.linalg.eigh(np.cov(Xtr.T)); ev, V = ev[::-1], V[:, ::-1]
        q = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1); Vq = V[:, :q]
        P = np.linalg.pinv(np.cov(Xtr.T))
        md = lambda X: np.einsum('ij,jk,ik->i', X - mu, P, X - mu)
        clean_syn = ctx['syn'][at == 'clean']
        dl = perturbations(ctx)
        s = dict(config=tag, lo=float(ctx['norm'][0]), hi=float(ctx['norm'][1]),
                 train_sd_median=float(np.median(sd)), pc1_share=float(ev[0] / ev.sum()), q95=q,
                 md2_holdout_median=float(np.median(md(ho))), md2_cleansyn_median=float(np.median(md(clean_syn))),
                 cleansyn_mean_shift_in_sd=float(np.median(np.abs(clean_syn.mean(0) - mu) / sd)))
        sig = np.abs(dl['B_drift_k1']).max(1)
        s['sigbar_median'] = float(np.median(sig)); s['sigbar_over_sd'] = float(np.median(sig) / np.median(sd))
        for t in M.POS_TYPES:
            d = dl[t]
            s[f'insub_{t}'] = float(np.median(((d @ Vq) ** 2).sum(1) / (d ** 2).sum(1)))
            s[f'rms_over_sd_{t}'] = float(np.median(np.sqrt((d ** 2).mean(1)) / np.median(sd)))
        summ.append(s); print(s, flush=True)
        vs = max(1, int(N * 0.2))
        for z in ZS:
            for seed in SEEDS:
                M.set_seeds(seed)
                ae = MV.DenseAE(Xtr.shape[1], z)
                ae.train(Xtr[:-vs], Xtr[-vs:], epochs=M.AE_EPOCHS, batch_size=M.BATCH)
                s_ho = ae.score(ho); thr = np.percentile(s_ho, 99.5); s_syn = ae.score(ctx['syn'])
                r = dict(config=tag, z=z, seed=seed, thr=float(thr), train_med=float(np.median(ae.score(Xtr))),
                         ho_med=float(np.median(s_ho)), clean_med=float(np.median(s_syn[at == 'clean'])),
                         spec=float(1 - (s_syn[at == 'clean'] >= thr).mean()))
                for t in M.POS_TYPES:
                    sel = (at == t) & (meta.severity.values >= 2)
                    r[f'rec_{t}'] = float((s_syn[sel] >= thr).mean())
                    r[f'med_over_thr_{t}'] = float(np.median(s_syn[sel]) / thr)
                rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                M.tf.keras.backend.clear_session()
        del ctx
    pd.DataFrame(summ).to_csv(os.path.join(OUT, 'summary.csv'), index=False, float_format='%.6g')
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'dae.csv'), index=False, float_format='%.6g')
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(d.groupby(['config', 'z']).mean(numeric_only=True).drop(columns='seed').round(3).to_string())


if __name__ == '__main__':
    main()
