# Stage 2d-3: spike recall at the clinical limit over the curves
# whose reported value the spike actually changes ("hit" curves), instead of over all 200 spike positions.
#   s2d/s2d2 choose the spike's k* from the median |bias| over hit curves (s2a2: |dA change| at k = 1 >= 5% of the
#   item maximum), but read recall_A_spike_k{k*} from the v2 benchmark JSONL, which averages over all 200 curves
#   (66-83% of them leave the reported value unchanged).  This script makes the recall use the same hit curves.
#   (1) score: re-runs the v2 benchmark scoring (s2b_benchmark_v2: unclipped inputs, linear AE outputs, N = 500,
#       holdout P99.5 threshold, seeds as in the benchmark) for the supplement table's methods, and stores the per-curve flag of
#       every A_spike row with the hit mask.  The all-curve recall recomputed from these flags is checked against the
#       JSONL value (exact for the deterministic methods; compact AE: TensorFlow, reported as a difference).
#   (2) summarize: per yardstick, hit-curve recall at s2d2's k* (react_only rows of clinical_severity*.csv), the
#       all-curve recall at the same k*, n_hit per item, and the mean over the items that reach the limit.
# The hit mask is recomputed here from the battery's RAM channel with s2a2.delta_a and checked against s2a2's
# frac_curves_changed.  Nothing in the earlier result folders is modified.
# Output: Results/20261021_Rev_M2_SpikeHit/{flags_<item>.parquet, recall_check.csv, spike_hit_by_item.csv,
#         spike_hit_summary.csv, dae_selection_rescore.jsonl}
# Usage:  <TF venv python> s2d3_m2_spike_hit_recall.py score --items TP      (one process per item)
#         py -3.14 s2d3_m2_spike_hit_recall.py summarize
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(BASE, 'Results')
OUT = os.path.join(RES, '20261021_Rev_M2_SpikeHit')
V2 = os.path.join(RES, '20260917_Rev_v2Benchmark')
YS = os.path.join(RES, '20261020_Rev_M2_Yardsticks')
BIAS = os.path.join(RES, '20261020_Rev_M2_BiasPhotometric', 'bias_summary_photometric.csv')
N_MAIN = 500
BUDGET = 99.5
METHODS = ['maha', 'maha_robust', 'pca', 'lof', 'dae_elbow']      # the supplement table's methods
YARDSTICKS = (('EFLM desirable bias', 'desirable_bias_pct', 'clinical_severity_desirablebias.csv'),
              ('EFLM desirable TEa', 'TEa_pct', 'clinical_severity.csv'),
              ('CLIA (reference)', 'CLIA_pct', 'clinical_severity_desirablebias.csv'))


def z_elbow():
    z = pd.read_csv(os.path.join(YS, 'dae_z_selection.csv'))
    return {r['item']: int(r['z_elbow']) for _, r in z[z['N'] == N_MAIN].iterrows()}


def hit_mask(ctx, sp):
    """base_id -> hit, from the k = 1 spike rows (s2a2's definition, recomputed on the benchmark's own rows)."""
    import s2a2_bias_photometric as B2
    meta = ctx['meta']
    at, sv = meta['anomaly_type'].values, meta['severity'].values
    cl = np.flatnonzero(at == 'clean'); k1 = np.flatnonzero((at == 'A_spike') & np.isclose(sv, 1.0))
    cl = cl[np.argsort(meta['base_id'].values[cl])]; k1 = k1[np.argsort(meta['base_id'].values[k1])]
    assert (meta['base_id'].values[cl] == meta['base_id'].values[k1]).all()
    dd1 = B2.delta_a(ctx['syn_ram'][k1] - ctx['syn_ram'][cl], sp)
    a = np.abs(dd1)
    hit = a >= 0.05 * a.max() if a.max() > 0 else np.zeros(len(a), bool)
    return pd.DataFrame({'base_id': meta['base_id'].values[k1], 'hit': hit, 'dd1_rel': a / a.max() if a.max() > 0 else a})


def score(items):
    import rev_common as R
    import s2b_benchmark_v2 as S
    import s2a2_bias_photometric as B2
    M = R.M
    S.install(); S.self_test()
    os.makedirs(OUT, exist_ok=True)
    S.SEL_LOG = os.path.join(OUT, 'dae_selection_rescore.jsonl')   # keep the benchmark's own log untouched
    sp_all = B2.settings()[0]
    ze = z_elbow()
    recs = {}
    for g in R.GROUPS:
        for l in open(os.path.join(V2, g, 'Reports', 'comparison.jsonl'), encoding='utf-8'):
            if l.strip():
                d = json.loads(l)
                recs[(d['item'], d['method'], d['N'])] = d
    for gname, g in R.GROUPS.items():
        for proc, synth in g['items']:
            if items and proc not in items:
                continue
            dst = os.path.join(OUT, f'flags_{proc}.parquet')
            if os.path.exists(dst):
                print(f'{proc}: done, skipping', flush=True)
                continue
            R.use_group(gname, OUT, 'spikehit')
            M.load_item = S.make_loader(gname, g['loader'])
            ctx = M.load_item(proc, synth)
            meta = ctx['meta']
            hm = hit_mask(ctx, sp_all[proc])
            bs = pd.read_csv(BIAS)
            chg = bs[(bs['item'] == sp_all[proc]['key']) & (bs['anomaly_type'] == 'A_spike') & np.isclose(bs['severity'], 1.0)]
            assert len(chg) == 1 and np.isclose(hm['hit'].mean(), chg['frac_curves_changed'].iloc[0]), proc
            at, sv = meta['anomaly_type'].values, meta['severity'].values
            sel = np.flatnonzero(at == 'A_spike')
            rows, chk = [], []
            for m in METHODS:
                mm = f'dae{ze[proc]}' if m == 'dae_elbow' else m
                seeds = M.SEEDS if mm in M.STOCHASTIC else [M.SEEDS[0]]
                for sd in seeds:
                    sc, _, _ = M.score_sets(mm, ctx, N_MAIN, sd)
                    thr = np.percentile(sc['holdout'], BUDGET)
                    flag = sc['syn'] >= thr
                    rows.append(pd.DataFrame({'item': proc, 'method': m, 'method_run': mm, 'seed': sd,
                                              'base_id': meta['base_id'].values[sel], 'severity': sv[sel],
                                              'flag': flag[sel]}))
                    if mm.startswith('dae'):
                        M.tf.keras.backend.clear_session()
                f = pd.concat([r for r in rows if r['method'].iloc[0] == m])
                rec = recs.get((proc, mm, N_MAIN))
                for k, gk in f.groupby('severity'):
                    jl = rec['budgets'][str(BUDGET)].get(f'recall_A_spike_k{k:g}', {}).get('mean', np.nan) if rec else np.nan
                    chk.append(dict(item=proc, method=m, method_run=mm, severity=k, n_seeds=len(seeds),
                                    recall_all_rescored=float(gk['flag'].mean()), recall_all_jsonl=jl))
                print(f'  {proc} {mm}: scored ({len(seeds)} seed(s))', flush=True)
            out = pd.concat(rows, ignore_index=True).merge(hm, on='base_id', how='left', validate='many_to_one')
            assert out['hit'].notna().all()
            ck = pd.DataFrame(chk)
            ck['abs_diff'] = (ck['recall_all_rescored'] - ck['recall_all_jsonl']).abs()
            assert ck['abs_diff'].notna().all(), f'{proc}: missing JSONL reference'
            assert ck.loc[ck['method'] != 'dae_elbow', 'abs_diff'].max() < 1e-12, f'{proc}: deterministic mismatch'
            ckp = os.path.join(OUT, 'recall_check.csv')
            if os.path.exists(ckp):                   # re-running an item replaces its rows
                old = pd.read_csv(ckp)
                ck = pd.concat([old[old['item'] != proc], ck], ignore_index=True)
            ck.to_csv(ckp, index=False)
            out.to_parquet(dst + '.tmp', index=False)
            os.replace(dst + '.tmp', dst)             # last: its presence marks the item as complete
            print(f'{proc}: n_hit={int(hm["hit"].sum())}, max |rescored - JSONL| = '
                  + ', '.join(f'{m} {d:.4f}' for m, d in ck.groupby('method')['abs_diff'].max().items()), flush=True)
            del ctx


def summarize():
    fl = pd.concat([pd.read_parquet(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))
                    if f.startswith('flags_') and f.endswith('.parquet')], ignore_index=True)
    # per (item, method, severity): recall over seeds then curves (= mean over all rows; seeds have equal n)
    keys = ['item', 'method', 'severity']
    per = pd.DataFrame({'recall_hit': fl[fl['hit']].groupby(keys)['flag'].mean(),
                        'recall_all': fl.groupby(keys)['flag'].mean(),
                        'recall_miss': fl[~fl['hit']].groupby(keys)['flag'].mean(),
                        'n_hit': fl[fl['seed'] == fl['seed'].min()].groupby(keys)['hit'].sum()}).reset_index()
    scored = set(fl['item'])
    rows, summ = [], []
    for ys, sub, fn in YARDSTICKS:
        cl = pd.read_csv(os.path.join(YS, sub, fn))
        assert (cl['yardstick'] == sub).all(), f'unexpected yardstick in {sub}/{fn}'
        cl = cl[(cl['anomaly_type'] == 'A_spike') & (cl['bias_def'] == 'react_only')]
        for _, r in cl.iterrows():
            assert r['item'] in scored, f"missing flags_{r['item']}.parquet"      # no silent partial run
            k = r['severity_at_TEa']
            for m in METHODS:
                p = per[(per['item'] == r['item']) & (per['method'] == m) & np.isclose(per['severity'], k)] \
                    if pd.notna(k) else per.iloc[0:0]
                assert pd.isna(k) or len(p) == 1, (r['item'], m, k)
                rows.append(dict(yardstick=ys, item=r['item'], limit_pct=r['limit_pct'], k_star=k,
                                 bias_at_k_star=r['bias_at_severity'], method=m,
                                 n_hit=int(p['n_hit'].iloc[0]) if len(p) else np.nan,
                                 recall_hit=float(p['recall_hit'].iloc[0]) if len(p) else np.nan,
                                 recall_miss=float(p['recall_miss'].iloc[0]) if len(p) else np.nan,
                                 recall_all=float(p['recall_all'].iloc[0]) if len(p) else np.nan,
                                 recall_all_s2d2=r[m] if pd.notna(k) else np.nan))
    by = pd.DataFrame(rows)
    by.to_csv(os.path.join(OUT, 'spike_hit_by_item.csv'), index=False)
    reached = by.dropna(subset=['k_star'])
    # deterministic methods must reproduce s2d2 exactly; the compact AE is retrained (TensorFlow is not bit-reproducible
    # across runs), so its hit and all-curve recalls are both taken from this run and the difference is reported
    assert reached['recall_all'].notna().all() and reached['recall_hit'].notna().all()
    det = reached[reached['method'] != 'dae_elbow']
    assert (det['recall_all'] - det['recall_all_s2d2']).abs().max() < 1e-9, 'all-curve recall does not reproduce s2d2'
    by['diff_all_vs_s2d2'] = by['recall_all'] - by['recall_all_s2d2']
    by.to_csv(os.path.join(OUT, 'spike_hit_by_item.csv'), index=False)
    for (ys, m), g in reached.groupby(['yardstick', 'method'], sort=False):
        summ.append(dict(yardstick=ys, method=m, n_items_evaluated=int(by[(by.yardstick == ys) & (by.method == m)].shape[0]),
                         n_items_reached=len(g), recall_hit_mean=g['recall_hit'].mean(), recall_hit_min=g['recall_hit'].min(),
                         recall_all_mean=g['recall_all'].mean(), recall_all_s2d2_mean=g['recall_all_s2d2'].mean(),
                         recall_miss_mean=g['recall_miss'].mean(), n_hit_total=int(g['n_hit'].sum()),
                         n_hit_min=int(g['n_hit'].min()), n_hit_max=int(g['n_hit'].max())))
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(OUT, 'spike_hit_summary.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(s.round(3).to_string(index=False))
        print(by[by.yardstick == 'EFLM desirable bias'].pivot(index='item', columns='method', values='recall_hit').round(3))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['score', 'summarize'])
    ap.add_argument('--items', nargs='+', default=None)
    a = ap.parse_args()
    score(a.items) if a.stage == 'score' else summarize()
