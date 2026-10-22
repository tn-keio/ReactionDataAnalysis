# Stage 2g: sample-level tests for the v2 benchmark (manuscript section 3.6 / S6.7), N = 500 only.
#
# The v2 run wrote aggregate metrics only (do_list=False), so the per-sample flags needed for McNemar
# and the base-curve-clustered bootstrap do not exist.  This script recomputes the N=500 scores for the
# methods of interest (seed 0 for every method, so it is deterministic and cheap: the 2D ResNet features
# are already cached), stores the per-sample flags, and runs:
#   * McNemar (exact binomial on discordant pairs), maha vs each method, pooled over items
#   * base-curve-clustered bootstrap (resampling the 200 base curves per item with replacement)
# Scope: the anomaly rows of the SUBMITTED battery definition (A/B/C/E at k >= 2) and, separately, the
# full v2 battery (including sub-1-sigma, W_timewarp, S_saturation).
# Output: Results/20260917_Rev_v2Benchmark/samplelevel/{flags_<item>.parquet, mcnemar.csv, bootstrap.csv}
# TF venv (imports the pipeline).  Usage: python s2g_sample_level_tests.py [--items ...] [--tests-only]

import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2b_benchmark_v2 as S2B      # noqa: E402  (also imports rev_common -> pipeline + TF)
R = S2B.R
M = R.M

OUT = os.path.join(R.results_dir('20260917_Rev_v2Benchmark'), 'samplelevel')
N = 500
SEEDS = [0, 1, 2, 3]        # stochastic methods are refitted per seed; deterministic ones are scored once
METHODS = ['maha', 'maha_robust', 'pca', 'lof', 'iforest', 'ocsvm', 'ae', 'cnn1d', 'twod_only', 'cae8', 'dae8']
PAIRED = [m for m in METHODS if m != 'maha']
POS = None      # set after the pipeline import (M.POS_TYPES)


def flags_for_item(group, proc, synth, loader, seeds):
    """Load the item ONCE and write one flag file per seed (re-loading per seed exhausted memory)."""
    ctx = loader(proc, synth)
    if any(m in M.USES_2D for m in METHODS):        # ResNet features come from the v2 run's cache
        for nm, arr in (('train', ctx['train_pool']), ('holdout', ctx['holdout']),
                        ('analysis', ctx['analysis']), ('syn', ctx['syn'])):
            ctx[f'feat_{nm}'] = M.feats_cached(proc, nm, arr)
    meta = ctx['meta'].copy()
    for seed in seeds:
        out = pd.DataFrame({'anomaly_type': meta['anomaly_type'].values,
                            'severity': meta['severity'].values.astype(float),
                            'base_id': meta['base_id'].values})
        for m in METHODS:
            if seed != SEEDS[0] and m not in M.STOCHASTIC and m not in ('cae8',):
                continue          # deterministic: identical across seeds, scored once
            sc, _, _ = M.score_sets(m, ctx, N, seed)
            thr = float(np.percentile(sc['holdout'], 99.5))
            out[m] = (sc['syn'] >= thr).astype(np.int8)
            del sc
            M.tf.keras.backend.clear_session()
        out.insert(0, 'seed', seed); out.insert(0, 'item', proc)
        out.to_parquet(os.path.join(OUT, f'flags_{proc}_s{seed}.parquet'), index=False)
        print(f'  {proc} seed={seed}: flags written ({len(out)} rows, {len(out.columns) - 5} methods)', flush=True)
        del out
    del ctx
    import gc; gc.collect()


def mcnemar(df, mask, tag):
    from scipy.stats import binomtest
    rows = []
    a = df.loc[mask, 'maha'].values
    for m in PAIRED:
        b = df.loc[mask, m].values
        n01 = int(((a == 1) & (b == 0)).sum())      # maha only
        n10 = int(((a == 0) & (b == 1)).sum())      # other only
        n = n01 + n10
        p = binomtest(n01, n, 0.5, alternative='greater').pvalue if n else 1.0
        rows.append(dict(scope=tag, comparison=f'maha>{m}', maha_only=n01, other_only=n10, p_binom=p))
    return rows


def bootstrap(df, mask, tag, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    sub = df.loc[mask, ['item', 'base_id', 'maha'] + PAIRED]
    clusters = sub.groupby(['item', 'base_id']).indices          # base curve = cluster
    keys = list(clusters)
    idx = [clusters[k] for k in keys]
    rows = []
    means = {m: [] for m in ['maha'] + PAIRED}
    for _ in range(n_boot):
        pick = rng.randint(0, len(keys), len(keys))
        sel = np.concatenate([idx[i] for i in pick])
        for m in means:
            means[m].append(sub[m].values[sel].mean())
    for m in PAIRED:
        d = np.array(means['maha']) - np.array(means[m])
        rows.append(dict(scope=tag, comparison=f'maha-{m}', mean_diff=float(d.mean()),
                         ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)),
                         frac_resamples_favouring_maha=float((d > 0).mean())))
    return rows


def main():
    global POS
    POS = list(M.POS_TYPES)
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=None)
    ap.add_argument('--tests-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.tests_only:
        S2B.install(); S2B.self_test()
        for gname, g in R.GROUPS.items():
            items = [it for it in g['items'] if a.items is None or it[0] in a.items]
            if not items:
                continue
            R.use_group(gname, R.results_dir('20260917_Rev_v2Benchmark'), 'v2')
            loader = S2B.make_loader(gname, g['loader'])
            S2B.SEL_LOG = os.path.join(OUT, 'dae_selection_samplelevel.jsonl')
            for proc, synth in items:
                todo = [s for s in SEEDS if not os.path.exists(os.path.join(OUT, f'flags_{proc}_s{s}.parquet'))]
                if not todo:
                    print(f'  {proc}: exists, skip', flush=True); continue
                flags_for_item(gname, proc, synth, loader, todo)
    files = sorted(f for f in os.listdir(OUT) if f.startswith('flags_') and f.endswith('.parquet') and '_s' in f)
    if not files:
        print('no flag files yet'); return
    parts = [pd.read_parquet(os.path.join(OUT, f)) for f in files]
    # deterministic methods are stored only for the first seed -> fill them forward per item
    base = pd.concat([p for p in parts if p['seed'].iloc[0] == SEEDS[0]], ignore_index=True)
    frames = []
    for p in parts:
        s = int(p['seed'].iloc[0]); it = p['item'].iloc[0]
        if s == SEEDS[0]:
            frames.append(p); continue
        b = base[base.item == it].reset_index(drop=True)
        miss = [c for c in b.columns if c not in p.columns]
        q = pd.concat([p.reset_index(drop=True), b[miss]], axis=1)
        frames.append(q[b.columns.tolist()].assign(seed=s))
    df = pd.concat(frames, ignore_index=True)
    print(f'seeds={sorted(df.seed.unique())}, rows={len(df)}')
    sub_mask = df.anomaly_type.isin(POS) & (df.severity >= 2)                 # submitted definition
    all_mask = df.anomaly_type.isin(POS + ['W_timewarp', 'S_saturation', 'D_kinetic'])
    rows_mc, rows_bs = [], []
    for s in sorted(df.seed.unique()):
        m = df.seed == s
        rows_mc += [dict(seed=s, **r) for r in mcnemar(df, sub_mask & m, 'submitted_battery(k>=2)')]
        rows_mc += [dict(seed=s, **r) for r in mcnemar(df, all_mask & m, 'v2_battery_all')]
        rows_bs += [dict(seed=s, **r) for r in bootstrap(df, sub_mask & m, 'submitted_battery(k>=2)')]
    mc = pd.DataFrame(rows_mc)
    bs = pd.DataFrame(rows_bs)
    mc.to_csv(os.path.join(OUT, 'mcnemar.csv'), index=False)
    bs.to_csv(os.path.join(OUT, 'bootstrap.csv'), index=False)
    with pd.option_context('display.width', 250):
        print('\n== McNemar ==\n', mc.round(4).to_string(index=False))
        print('\n== clustered bootstrap ==\n', bs.round(4).to_string(index=False))
    print(f'\nitems={df.item.nunique()} rows={len(df)}')


if __name__ == '__main__':
    main()
