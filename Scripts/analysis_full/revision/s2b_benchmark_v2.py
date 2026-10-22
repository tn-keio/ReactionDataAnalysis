# Stage 2/3 (Reviewer #2, Methods-1/2/3): the v2 method comparison, 21-item panel.
#
# Differences from the submitted run (everything else is the submitted pipeline, reused verbatim):
#   M1  inputs unclipped for every method (rev_common.set_clip(False));
#       1D-AE and 1D-CNN decoders with LINEAR output (models_v2.linear_outputs);
#       2D-CNN rendering y-range widened from [0, 1] to [-0.05, 1.05] (training-pool min-max + 5%).
#   M2  evaluation battery = submitted battery + rev2026_extra_<group>.parquet
#       (sub-1-sigma A/B/C/E, W_timewarp, S_saturation); metrics extended accordingly.
#   M3  compact dense AEs 'dae1'...'dae16' (34-16-z-16-34, linear) as extra methods; for each fit the
#       clean validation MSE and the recall on an independent SELECTION battery are logged to
#       Reports/dae_selection.jsonl (used later for the elbow rule and the oracle upper bound).
# Output: Results/20260917_Rev_v2Benchmark[_pilot]/<group>/Reports/comparison.jsonl (resume-aware)
# TF venv.  Example:  python s2b_benchmark_v2.py --pilot --items KL-6 TB ALB

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_common as R          # noqa: E402  (imports the pipeline + TF)
M = R.M
import models_v2 as MV          # noqa: E402
import contemporaneous_synth as CS   # noqa: E402

CS.T = len(M.RAM); CS.IDX = np.arange(CS.T)

DAE_Z = [1, 2, 4, 8, 16]
CAE_Z = [2, 8]
LR_TAGS = {'lr3e4': 3e-4}          # learning-rate sensitivity (default is Adam 1e-3)
DAE = [f'dae{z}' for z in DAE_Z] + [f'cae{z}' for z in CAE_Z] \
      + [f'dae8_{t}' for t in LR_TAGS] + [f'cae8_{t}' for t in LR_TAGS]
ALL_METHODS = list(M.METHODS_OUT) + DAE
YLIM = (-0.05, 1.05)
EXTRA = {g: os.path.join(R.SYN_DIR, f'rev2026_extra_{g}.parquet') for g in R.GROUPS}
SEL_LOG = None          # set per group

_ORIG_SCORE_SETS = M.score_sets
_CUR = {}               # current (group, item) for logging


# ------------------------------------------------------------------ M1: 2D rendering range
def render_feats_v2(X, tag=''):
    """Copy of M.render_feats with the y-range widened (the only change)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from tensorflow.keras.applications.resnet50 import preprocess_input
    n, L = X.shape; feats, buf = [], []
    fig = plt.figure(figsize=(M.IMG_PX / 100., M.IMG_PX / 100.), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_ylim(*YLIM); ax.set_xlim(0, L - 1)
    line, = ax.plot(np.zeros(L), color='black', lw=1.5); cv = fig.canvas
    for i in range(n):
        line.set_ydata(X[i]); cv.draw()
        buf.append(np.asarray(cv.buffer_rgba())[:, :, :3].astype('float32'))
        if len(buf) == M.RESNET_BATCH or i == n - 1:
            feats.append(M.resnet().predict(preprocess_input(np.array(buf)), verbose=0)); buf = []
        if n > 3000 and (i + 1) % 3000 == 0:
            print(f'      feat[{tag}] {i + 1}/{n}', flush=True)
    plt.close(fig)
    return np.concatenate(feats, axis=0).astype('float32')


# ------------------------------------------------------------------ M2: extended battery + metrics
def make_loader(group, base_loader):
    extra_all = pd.read_parquet(EXTRA[group])

    def load_v2(proc, synth):
        ctx = base_loader(proc, synth)
        lo, hi = ctx['norm']
        ex = extra_all[extra_all['item'] == synth].reset_index(drop=True)
        assert len(ex), f'no extra battery rows for {synth} in {group}'
        ex_calc = ex[M.RAM].values.astype('float64') - ex[M.RAS].values.astype('float64')
        ctx['n_syn_submitted'] = len(ctx['syn'])
        ctx['syn'] = np.concatenate([ctx['syn'], M.apply_norm(ex_calc, lo, hi)])
        ctx['meta'] = pd.concat([ctx['meta'], ex[['anomaly_type', 'severity', 'base_id']]], ignore_index=True)
        ctx['syn_ram'] = np.concatenate([ctx['syn_ram'], ex[M.RAM].values]);
        ctx['syn_ras'] = np.concatenate([ctx['syn_ras'], ex[M.RAS].values])
        # independent SELECTION battery for the compact-AE oracle: training-pool curves, seed 1,
        # injected on the calc channel (sigma from calc; documented deviation from the RAM-based battery)
        raw_tr = ctx['train_pool'] * (hi - lo + 1e-8) + lo
        sel = CS.make_battery(raw_tr, np.zeros_like(raw_tr), seed=1, include_d=False, include_benign=False)
        ctx['sel'] = M.apply_norm(sel['calc'], lo, hi)
        ctx['sel_meta'] = (sel['atype'], sel['severity'])
        _CUR.update(group=group, item=proc)
        return ctx
    return load_v2


def metrics_v2(flag_syn, meta, s_ho, thr):
    at = meta['anomaly_type'].values; sev = meta['severity'].values
    clean = at == 'clean'; benign = at == 'benign_offset'
    m = {'specificity_clean': float(1 - flag_syn[clean].mean()),
         'fp_clean': float(flag_syn[clean].mean()),
         'fp_benign': float(flag_syn[benign].mean()) if benign.any() else float('nan'),
         'fp_holdout': float((s_ho >= thr).mean()),
         'recall_main': float(flag_syn[np.isin(at, M.POS_TYPES) & (sev >= 2)].mean())}
    for t in M.POS_TYPES:
        sel = at == t
        m[f'recall_{t}'] = float(flag_syn[sel & (sev >= 2)].mean())
        for k in sorted(np.unique(sev[sel])):
            m[f'recall_{t}_k{k:g}'] = float(flag_syn[sel & np.isclose(sev, k)].mean())
    for t, pre in (('D_kinetic', 'D_lag'), ('W_timewarp', 'W_r'), ('S_saturation', 'S_q')):
        sel = at == t
        for k in sorted(np.unique(sev[sel])):
            m[f'recall_{pre}{k:g}'] = float(flag_syn[sel & np.isclose(sev, k)].mean())
    return m


# ------------------------------------------------------------------ M3: compact dense AE
def score_sets_v2(method, ctx, N, seed):
    if not (method.startswith('dae') or method.startswith('cae')):
        return _ORIG_SCORE_SETS(method, ctx, N, seed)
    kind, spec = method[:3], method[3:]
    lr = 1e-3
    if '_' in spec:
        spec, tag = spec.split('_', 1)
        lr = LR_TAGS[tag]
    z = int(spec)
    Xtr = ctx['train_pool'][:N]; vs = max(1, int(len(Xtr) * 0.2))
    M.set_seeds(seed)
    t0 = time.time()
    ae = (MV.DenseAE if kind == 'dae' else MV.ConvCompactAE)(Xtr.shape[1], z, lr=lr)
    info = ae.train(Xtr[:-vs], Xtr[-vs:], epochs=M.AE_EPOCHS, batch_size=M.BATCH)
    ft = time.time() - t0
    sc = {'train': ae.score(Xtr), 'holdout': ae.score(ctx['holdout']), 'syn': ae.score(ctx['syn']),
          'analysis': ae.score(ctx['analysis'])}
    # selection-battery recall at the same holdout P99.5 threshold (oracle input)
    thr = np.percentile(sc['holdout'], 99.5)
    s_sel = ae.score(ctx['sel']) >= thr
    at, sv = ctx['sel_meta']
    pos = np.isin(at, M.POS_TYPES)
    rec = dict(group=_CUR.get('group'), item=_CUR.get('item'), method=method, kind=kind, z=z, lr=lr, N=N, seed=seed, **info,
               sel_recall_main=float(s_sel[pos & (sv >= 2)].mean()),
               sel_recall_k1=float(s_sel[pos & (sv == 1)].mean()),
               sel_clean_flag=float(s_sel[at == 'clean'].mean()))
    with open(SEL_LOG, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec) + '\n')
    t1 = time.time(); _ = ae.score(ctx['analysis']); ip = (time.time() - t1) / max(len(ctx['analysis']), 1)
    M.tf.keras.backend.clear_session()
    return sc, ft, ip


def install():
    R.set_clip(False)
    MV.linear_outputs(True)
    M.render_feats = render_feats_v2
    M.metrics_at = metrics_v2
    M.score_sets = score_sets_v2
    M.COMPUTE_METHODS = list(M.COMPUTE_METHODS) + DAE
    M.STOCHASTIC = set(M.STOCHASTIC) | set(DAE)
    M.METHODS_OUT = list(ALL_METHODS)


def self_test():
    """Fail fast: verify every override is active before hours of computation."""
    ae = M.AutoencoderModel(input_length=len(M.RAM))
    assert MV.output_activation(ae.model) == 'linear', 'AE output is not linear'
    cnn = M.CNN1DFeatureExtractor(input_length=len(M.RAM), feature_dim=64)
    assert MV.output_activation(cnn._autoencoder) == 'linear', '1D-CNN output is not linear'
    z = M.apply_norm(np.array([[-1.0, 2.0]]), 0.0, 1.0)
    assert z.min() < 0 and z.max() > 1, 'normalization still clipped'
    assert M.render_feats is render_feats_v2 and M.metrics_at is metrics_v2 and M.score_sets is score_sets_v2
    M.tf.keras.backend.clear_session()
    print('self-test OK: linear AE/CNN outputs, unclipped norm, v2 render/metrics/score_sets', flush=True)


def main():
    global SEL_LOG
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--tag', default=None, help='write to Results/20260917_Rev_v2Benchmark_<tag> with its own cache')
    ap.add_argument('--items', nargs='+', default=None)
    ap.add_argument('--groups', nargs='+', default=list(R.GROUPS))
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS)
    a = ap.parse_args()
    install(); self_test()
    suffix = '_pilot' if a.pilot else (f'_{a.tag}' if a.tag else '')
    out = R.results_dir('20260917_Rev_v2Benchmark' + suffix)
    for gname in a.groups:
        g = R.use_group(gname, out, 'v2' + (a.tag or ''))
        items = [it for it in g['items'] if a.items is None or it[0] in a.items]
        if not items:
            continue
        M.load_item = make_loader(gname, g['loader'])
        os.makedirs(M.REPORTS, exist_ok=True)
        SEL_LOG = os.path.join(M.REPORTS, 'dae_selection.jsonl')
        print(f'=== v2 group={gname} items={[i[0] for i in items]} methods={a.methods} out={M.OUT}', flush=True)
        t0 = time.time()
        M.run(items, a.methods, do_list=False)
        print(f'=== group {gname} done in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
