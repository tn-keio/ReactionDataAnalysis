# Stage 1z9b: §3.6 / S6 statistics for the 13-method main family
#   = the 11 submitted detectors + dae_elbow (validation-selected dense compact AE, primary) + cae8.
# Replaces the 12-method family (11 + cae8) of s2f_stats_v2.py --family prespecified.
#   1. assay-blocked rank tests (Friedman, mean ranks, Nemenyi, one-sided Wilcoxon maha > other with Holm):
#      s2f_stats_v2.main() imported unchanged; BASE_METHODS gets dae_elbow, the loader is s1z5.f_load
#      (v2 JSONL + the CORRECTED compact-AE selection, dae_z_selection_fixed.csv).
#   2. Table S6-3: item-block bootstrap of mean and worst-item recall (k >= 2), B = 2000, seed 20260726,
#      same construction as ForPaper/Rev03_Uncertainty_and_Ranks/make_uncertainty.worstitem_ci.
#   3. Table S6-2 / §3.6: sample-level McNemar discordance (maha vs each of 12 competitors) on the submitted
#      battery definition (A/B/C/E, k >= 2) and on the full v2 battery, with the base-curve cluster
#      bootstrap of b/c and b - c (make_uncertainty.clustered_mcnemar construction, B = 2000) and the
#      recall-difference bootstrap of s2g_sample_level_tests.bootstrap.  Flags: s2g samplelevel files
#      + s1z9a samplelevel_extra (dae_elbow, ae2d_and, ae2d_ens).  Per seed; seed 0 is the headline
#      (as in v2 §5b), seeds 1-3 give the range.
# Output: Results/20261015_Rev_M3_Stats13/{stats13_*.csv, worstitem_ci13.csv, mcnemar13.csv,
#         mcnemar13_cluster.csv, bootstrap13.csv}
# py3.14 (no TF).  Usage: py -3.14 s1z9b_stats_13methods.py [--skip-samplelevel]

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2f_stats_v2 as F                    # noqa: E402
import s1z5_downstream_compact_fix as Z5    # noqa: E402

RES = os.path.abspath(os.path.join(F.V2, '..'))
OUT = os.path.join(RES, '20261015_Rev_M3_Stats13')
SEL_FIXED = os.path.join(RES, '20261013_Rev_M3_CompactSelFix', 'dae_z_selection_fixed.csv')
S2G = os.path.join(F.V2, 'samplelevel')
EXTRA = os.path.join(OUT, 'samplelevel_extra')
POS = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']
METHODS13 = F.BASE_METHODS + ['dae_elbow', 'cae8']
PAIRED = [m for m in METHODS13 if m != 'maha']


def rank_tests():
    os.makedirs(OUT, exist_ok=True)
    shutil.copyfile(SEL_FIXED, os.path.join(OUT, 'dae_z_selection.csv'))
    F.V2 = OUT                                   # outputs + the selection table read by Z5.f_load
    F.load = Z5.f_load
    F.BASE_METHODS = F.BASE_METHODS + ['dae_elbow']   # 'prespecified' then appends cae8 -> 13 methods
    sys.argv = ['s2f_stats_v2.py', '--family', 'prespecified']
    F.main()
    for f in os.listdir(OUT):                    # s2f writes stats_v2_*.csv; tag them as the 13-method family
        if f.startswith('stats_v2_') and f.endswith('.csv'):
            os.replace(os.path.join(OUT, f), os.path.join(OUT, f.replace('stats_v2_', 'stats13_')))
    mr = pd.read_csv(os.path.join(OUT, 'stats13_meanranks.csv'))
    assert mr.method.nunique() == 13 and set(mr.method) == set(METHODS13), mr.method.unique()


def worstitem_ci(B=2000, seed=20260726):
    recs = Z5.f_load()
    X, items = F.matrix(recs, METHODS13, 'recall_main')
    items = [it for it in sorted({k[0] for k in recs}) if all((it, m, F.N) in recs for m in METHODS13)]
    assert X.shape == (21, 13), X.shape
    rng = np.random.default_rng(seed)
    rows = []
    for j, m in enumerate(METHODS13):
        v = X[:, j]
        idx = rng.integers(0, len(v), size=(B, len(v)))
        bmin, bmean = v[idx].min(axis=1), v[idx].mean(axis=1)
        rows.append(dict(method=m, mean=v.mean(), mean_lo=np.percentile(bmean, 2.5), mean_hi=np.percentile(bmean, 97.5),
                         worst=v.min(), worst_item=items[int(v.argmin())],
                         worst_lo=np.percentile(bmin, 2.5), worst_hi=np.percentile(bmin, 97.5)))
    df = pd.DataFrame(rows).sort_values('mean', ascending=False)
    df.to_csv(os.path.join(OUT, 'worstitem_ci13.csv'), index=False)
    print('\n== Table S6-3 (item-block bootstrap, B=2000) ==\n', df.round(3).to_string(index=False))


def load_flags():
    files = sorted(f for f in os.listdir(S2G) if f.startswith('flags_') and f.endswith('.parquet'))
    base = {}
    for f in files:
        p = pd.read_parquet(os.path.join(S2G, f))
        base[(p['item'].iloc[0], int(p['seed'].iloc[0]))] = p
    frames = []
    for (it, s), p in base.items():
        if s != 0:          # deterministic methods are stored only for seed 0 (s2g) -> fill forward
            b = base[(it, 0)]
            miss = [c for c in b.columns if c not in p.columns]
            p = pd.concat([p.reset_index(drop=True), b[miss].reset_index(drop=True)], axis=1)
        ex = pd.read_parquet(os.path.join(EXTRA, f'flags_{it}_s{s}.parquet'))
        assert len(ex) == len(p) and (ex.base_id.values == p.base_id.values).all() \
            and (ex.anomaly_type.values == p.anomaly_type.values).all(), (it, s)
        p = p.reset_index(drop=True)
        for c in ('dae_elbow', 'ae2d_and', 'ae2d_ens'):
            p[c] = ex[c].values
        frames.append(p)
    df = pd.concat(frames, ignore_index=True)
    assert df.item.nunique() == 21 and sorted(df.seed.unique()) == [0, 1, 2, 3], (df.item.nunique(), df.seed.unique())
    return df


def mcnemar_counts(sub):
    a = sub['maha'].values
    out = {}
    for m in PAIRED:
        b = sub[m].values
        out[m] = (int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()))
    return out


def samplelevel(B=2000):
    from scipy.stats import binomtest
    df = load_flags()
    sub_mask = df.anomaly_type.isin(POS) & (df.severity >= 2)
    all_mask = df.anomaly_type.isin(POS + ['W_timewarp', 'S_saturation', 'D_kinetic'])
    rows_mc, rows_cl, rows_bs = [], [], []
    for s in range(4):
        for tag, mask in (('submitted_battery(k>=2)', sub_mask), ('v2_battery_all', all_mask)):
            sub = df[mask & (df.seed == s)]
            for m, (n01, n10) in mcnemar_counts(sub).items():
                p = binomtest(n01, n01 + n10, 0.5, alternative='greater').pvalue if n01 + n10 else 1.0
                rows_mc.append(dict(seed=s, scope=tag, comparison=f'maha>{m}', maha_only=n01, other_only=n10,
                                    recall_maha=sub['maha'].mean(), recall_other=sub[m].mean(), p_binom=p))
        # cluster bootstrap of b/c and b-c (make_uncertainty construction), submitted battery only
        sub = df[sub_mask & (df.seed == s)].reset_index(drop=True)
        rng = np.random.default_rng(20260726)
        offs = []
        for it, g in sub.groupby('item'):
            offs.append([g.index.values[ix] for ix in g.groupby('base_id').indices.values()])
        A = sub['maha'].values.astype(bool)
        Fm = {m: sub[m].values.astype(bool) for m in PAIRED}
        ratios = {m: [] for m in PAIRED}; diffs = {m: [] for m in PAIRED}
        for _ in range(B):
            take = np.concatenate([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]) for g in offs])
            a = A[take]
            for m in PAIRED:
                o = Fm[m][take]
                b = int((a & ~o).sum()); c = int((~a & o).sum())
                ratios[m].append(b / c if c else np.inf); diffs[m].append(b - c)
        obs = mcnemar_counts(sub)
        for m in PAIRED:
            rr = np.array(ratios[m]); dd = np.array(diffs[m], float); fin = rr[np.isfinite(rr)]
            b, c = obs[m]
            rows_cl.append(dict(seed=s, comparison=f'maha>{m}', b=b, c=c, ratio=b / c if c else np.inf,
                                ratio_lo=np.percentile(fin, 2.5) if fin.size else np.inf,
                                ratio_hi=np.percentile(fin, 97.5) if fin.size else np.inf,
                                diff_lo=np.percentile(dd, 2.5), diff_hi=np.percentile(dd, 97.5),
                                frac_favouring_maha=(dd > 0).mean()))
        # recall-difference bootstrap (s2g construction: clusters = (item, base_id) pooled, RandomState(0))
        rs = np.random.RandomState(0)
        keys = [ix for g in offs for ix in g]
        means = {m: [] for m in ['maha'] + PAIRED}
        cols = {m: sub[m].values for m in means}
        for _ in range(B):
            sel = np.concatenate([keys[i] for i in rs.randint(0, len(keys), len(keys))])
            for m in means:
                means[m].append(cols[m][sel].mean())
        for m in PAIRED:
            d = np.array(means['maha']) - np.array(means[m])
            rows_bs.append(dict(seed=s, comparison=f'maha-{m}', mean_diff=d.mean(), ci_lo=np.percentile(d, 2.5),
                                ci_hi=np.percentile(d, 97.5), frac_resamples_favouring_maha=(d > 0).mean()))
        print(f'  seed {s}: sample-level done', flush=True)
    mc, cl, bs = pd.DataFrame(rows_mc), pd.DataFrame(rows_cl), pd.DataFrame(rows_bs)
    mc.to_csv(os.path.join(OUT, 'mcnemar13.csv'), index=False)
    cl.to_csv(os.path.join(OUT, 'mcnemar13_cluster.csv'), index=False)
    bs.to_csv(os.path.join(OUT, 'bootstrap13.csv'), index=False)
    with pd.option_context('display.width', 250):
        print('\n== McNemar (seed 0, submitted battery) ==\n',
              cl[cl.seed == 0].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-samplelevel', action='store_true')
    a = ap.parse_args()
    rank_tests()
    worstitem_ci()
    if not a.skip_samplelevel:
        samplelevel()


if __name__ == '__main__':
    main()
