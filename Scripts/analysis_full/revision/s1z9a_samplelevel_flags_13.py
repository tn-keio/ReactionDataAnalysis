# Stage 1z9a: per-sample flags for the 13-method main analysis.
# The revised Table 1 keeps the submitted 1D-AE and adds TWO compact AEs to Table 1: the validation-selected
# dense AE (dae_elbow; z per item from dae_z_selection_fixed.csv, N = 500) and the convolutional cae8.
# s2g_sample_level_tests.py wrote flags for cae8 and dae8 but not for dae_elbow (z differs by item), and
# never for the two ensembles (AE + 2D-CNN, AND cascade and score-level max).  This script writes, per
# item and seed, the extra columns  dae_elbow, ae2d_and, ae2d_ens  on the same rows (same battery, same
# loader, same holdout P99.5 rule) so that s1z9b can run McNemar / clustered bootstrap over 13 methods.
#   * dae_elbow: score_sets_v2('dae<z>') for z = z_elbow(item, N=500)
#   * ae2d_and / ae2d_ens: same construction as run_paper_method_comparison.run() (ae + 2D score from
#     ONE seed; AND of the two P99.5 thresholds / max of holdout-z-normalised scores, single P99.5)
#   The ae and 2D scores are refitted here, so they are not bit-identical to the ae / twod_only columns of
#   the s2g flags (TensorFlow nondeterminism); the ensembles are internally consistent per seed.
# Existing scripts are imported unchanged; only module globals are redirected.
# Output: Results/20261015_Rev_M3_Stats13/samplelevel_extra/flags_<item>_s<seed>.parquet,
#         dae_selection_s1z9a.jsonl (score_sets_v2 log)
# TF venv.  Usage: python s1z9a_samplelevel_flags_13.py [--items ...]   (one item per process is safest)

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2b_benchmark_v2 as S2B      # noqa: E402  (imports rev_common -> pipeline + TF)
R = S2B.R
M = R.M

N = 500
SEEDS = [0, 1, 2, 3]
OUT = os.path.join(R.results_dir('20261015_Rev_M3_Stats13'), 'samplelevel_extra')
ZSEL = os.path.join(R.results_dir('20261013_Rev_M3_CompactSelFix'), 'dae_z_selection_fixed.csv')
S2G = os.path.join(R.results_dir('20260917_Rev_v2Benchmark'), 'samplelevel')


def feats_readonly(proc, nm, X):
    """Read the v2 2D-feature cache WITHOUT ever writing it (M.feats_cached re-renders and overwrites on a
    length mismatch).  A longer cache is accepted only as a prefix (rendering is deterministic: the v2 and
    v2prozone CRE caches are bit-identical, checked 2026-10-15)."""
    f = np.load(os.path.join(M.CACHE, f'{proc}_{nm}.npy'))
    assert len(f) >= len(X), f'cache {proc}_{nm} shorter than data ({len(f)} < {len(X)})'
    return f[:len(X)]


def run_item(proc, synth, loader, z):
    ctx = loader(proc, synth)
    # The extended battery later gained saturation levels q = 0.5 / 0.2 (prozone run), appended AFTER the
    # rows that s2g scored.  Drop exactly those trailing rows before scoring so the scope equals s2g's.
    n_ref = len(pd.read_parquet(os.path.join(S2G, f'flags_{proc}_s0.parquet'), columns=['base_id']))
    if len(ctx['syn']) > n_ref:
        tail = ctx['meta'].iloc[n_ref:]
        old_q = set(ctx['meta'].iloc[:n_ref].query("anomaly_type == 'S_saturation'")['severity'].round(6))
        assert (tail.anomaly_type == 'S_saturation').all() and not set(tail.severity.round(6)) & old_q, \
            f'unexpected trailing rows {proc}'
        for k in ('syn', 'syn_ram', 'syn_ras'):
            ctx[k] = ctx[k][:n_ref]
        ctx['meta'] = ctx['meta'].iloc[:n_ref].reset_index(drop=True)
    for nm, arr in (('train', ctx['train_pool']), ('holdout', ctx['holdout']),
                    ('analysis', ctx['analysis']), ('syn', ctx['syn'])):
        ctx[f'feat_{nm}'] = feats_readonly(proc, nm, arr)
    meta = ctx['meta']
    for seed in SEEDS:
        p = os.path.join(OUT, f'flags_{proc}_s{seed}.parquet')
        if os.path.exists(p):
            print(f'  {proc} s{seed}: exists, skip', flush=True); continue
        out = pd.DataFrame({'item': proc, 'seed': seed, 'anomaly_type': meta['anomaly_type'].values,
                            'severity': meta['severity'].values.astype(float), 'base_id': meta['base_id'].values})
        sc, _, _ = M.score_sets(f'dae{z}', ctx, N, seed)
        out['dae_elbow'] = (sc['syn'] >= np.percentile(sc['holdout'], 99.5)).astype(np.int8)
        del sc; M.tf.keras.backend.clear_session()
        sc_ae, _, _ = M.score_sets('ae', ctx, N, seed)
        M.tf.keras.backend.clear_session()
        sc_2d, _, _ = M.score_sets('ae2d', ctx, N, seed)
        thr_ae = np.percentile(sc_ae['holdout'], 99.5); thr_2d = np.percentile(sc_2d['holdout'], 99.5)
        out['ae2d_and'] = ((sc_ae['syn'] >= thr_ae) & (sc_2d['syn'] >= thr_2d)).astype(np.int8)
        z_ho = np.maximum(M._z(sc_ae, 'holdout'), M._z(sc_2d, 'holdout'))
        z_syn = np.maximum(M._z(sc_ae, 'syn'), M._z(sc_2d, 'syn'))
        out['ae2d_ens'] = (z_syn >= np.percentile(z_ho, 99.5)).astype(np.int8)
        del sc_ae, sc_2d; M.tf.keras.backend.clear_session()
        # row alignment with the s2g flags of the same item / seed
        ref = pd.read_parquet(os.path.join(S2G, f'flags_{proc}_s{seed}.parquet'),
                              columns=['anomaly_type', 'severity', 'base_id'])
        # The extended battery later gained saturation levels q = 0.5 / 0.2 (prozone run), appended AFTER the
        # rows that s2g scored.  Drop exactly those trailing rows so the 13-method tests use the s2g scope.
        if len(out) > len(ref):
            tail = out.iloc[len(ref):]
            old_q = set(ref.loc[ref.anomaly_type == 'S_saturation', 'severity'].round(6))
            assert (tail.anomaly_type == 'S_saturation').all() and not set(tail.severity.round(6)) & old_q, \
                f'unexpected trailing rows {proc} s{seed}'
            out = out.iloc[:len(ref)].reset_index(drop=True)
        ok = len(ref) == len(out) and (ref['anomaly_type'].values == out['anomaly_type'].values).all() \
            and np.allclose(ref['severity'].values, out['severity'].values) \
            and (ref['base_id'].values == out['base_id'].values).all()
        if not ok:
            out.to_parquet(os.path.join(OUT, f'MISMATCH_{proc}_s{seed}.parquet'), index=False)
            print(f'  row mismatch {proc} s{seed}: len {len(ref)} vs {len(out)}; types ref '
                  f'{ref.anomaly_type.value_counts().to_dict()} new {out.anomaly_type.value_counts().to_dict()}', flush=True)
            raise SystemExit(2)
        out.to_parquet(p, index=False)
        main = np.isin(out.anomaly_type, M.POS_TYPES) & (out.severity >= 2)
        print(f'  {proc} s{seed} z={z}: recall(k>=2) dae_elbow={out.dae_elbow[main].mean():.3f} '
              f'ae2d_and={out.ae2d_and[main].mean():.3f} ae2d_ens={out.ae2d_ens[main].mean():.3f}', flush=True)
        del out; gc.collect()
    del ctx; gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    zsel = pd.read_csv(ZSEL)
    zmap = dict(zip(zsel[zsel.N == N]['item'], zsel[zsel.N == N]['z_elbow'].astype(int)))
    S2B.install(); S2B.self_test()
    for gname, g in R.GROUPS.items():
        items = [it for it in g['items'] if a.items is None or it[0] in a.items]
        if not items:
            continue
        R.use_group(gname, R.results_dir('20260917_Rev_v2Benchmark'), 'v2')   # reads the v2 2D-feature cache
        loader = S2B.make_loader(gname, g['loader'])
        S2B.SEL_LOG = os.path.join(OUT, 'dae_selection_s1z9a.jsonl')
        for proc, synth in items:
            print(f'== {gname} {proc} (z_elbow={zmap[proc]})', flush=True)
            run_item(proc, synth, loader, zmap[proc])


if __name__ == '__main__':
    main()
