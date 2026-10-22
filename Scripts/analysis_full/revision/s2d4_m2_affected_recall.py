# Stage 2d-4: recall at the clinical limit over the curves whose
# reported value the anomaly actually changes ("affected" curves), for every anomaly type whose effect varies by curve.
#   Policy: anomalies that leave the reported value unchanged may still be abnormal reactions and must be
#   detected (shown by the main benchmark); this analysis relates each method's sensitivity to the effect on the reported value,
#   so the clinical-limit analysis counts recall only on curves whose reported value was affected.
#   - A_spike     : affected = s2a2's hit mask, fixed at k = 1 (|dA change| >= 5% of the item maximum), every k.
#   - S_saturation, W_timewarp : affected = s2a2's 'substantive change' mask at each severity
#                   (|dA change| > 1% of the median |dA| of the clean curves).
#   - B_drift, C_hump, E_baseline change dA by the same amount on every curve (s2a2), so all curves are affected and
#     their recall stays the benchmark's all-curve value (not re-scored here).
#   In every case k* (q*, r*) is s2d2's react_only severity, i.e. the median |bias| over the same affected curves.
# s2d3 (spike only) is superseded by this script; its folder is kept as the record of 2026-10-21 morning.
#   (1) score: re-runs the v2 benchmark scoring (s2b_benchmark_v2: unclipped inputs, linear AE outputs, N = 500,
#       holdout P99.5 threshold, the benchmark's seeds) for the supplement table's methods and stores the per-curve flag of
#       every A_spike, S_saturation and W_timewarp row with its affected mask.  The all-curve recall from these flags
#       is checked against the benchmark JSONL (main run; q = 0.5 / 0.2 from the prozone run): exact for the
#       deterministic methods.  The compact AE is retrained and TensorFlow training is not bit-reproducible here
#       (enable_op_determinism did not make two runs identical, 2026-10-21), so its affected-curve and all-curve values
#       are both taken from this one run.
#   (2) summarize: per yardstick and type, affected-curve recall at k*, the all-curve recall at the same k*, n_affected
#       per item, and the mean over the items that reach the limit.
# The masks are recomputed from the benchmark's own rows with s2a2.delta_a and checked against s2a2's
# frac_curves_changed.  Nothing in the earlier result folders is modified.
# Output: Results/20261021_Rev_M2_AffectedRecall/{flags_<item>.parquet, recall_check.csv, affected_by_item.csv,
#         affected_summary.csv, dae_selection_rescore.jsonl}
# Usage:  <TF venv python> s2d4_m2_affected_recall.py score --items TP      (one process per item; run_s2d4_affected.ps1)
#         py -3.14 s2d4_m2_affected_recall.py summarize
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
OUT = os.path.join(RES, '20261021_Rev_M2_AffectedRecall')
V2 = os.path.join(RES, '20260917_Rev_v2Benchmark')
PROZONE = os.path.join(RES, '20260917_Rev_v2Benchmark_prozone', 'main', 'Reports', 'comparison.jsonl')
YS = os.path.join(RES, '20261020_Rev_M2_Yardsticks')
BIAS = os.path.join(RES, '20261020_Rev_M2_BiasPhotometric', 'bias_summary_photometric.csv')
N_MAIN = 500
BUDGET = 99.5
METHODS = ['maha', 'maha_robust', 'pca', 'lof', 'dae_elbow']      # the supplement table's methods
TYPES = ['A_spike', 'S_saturation', 'W_timewarp']
KEY = {'A_spike': 'recall_A_spike_k{:g}', 'S_saturation': 'recall_S_q{:g}', 'W_timewarp': 'recall_W_r{:g}'}
YARDSTICKS = (('EFLM desirable bias', 'desirable_bias_pct', 'clinical_severity_desirablebias.csv'),
              ('EFLM desirable TEa', 'TEa_pct', 'clinical_severity.csv'),
              ('CLIA (reference)', 'CLIA_pct', 'clinical_severity_desirablebias.csv'))


def z_elbow():
    z = pd.read_csv(os.path.join(YS, 'dae_z_selection.csv'))
    return {r['item']: int(r['z_elbow']) for _, r in z[z['N'] == N_MAIN].iterrows()}


def affected_masks(ctx, sp):
    """(anomaly_type, severity, base_id) -> affected, recomputed on the benchmark's own rows with s2a2's definitions."""
    import s2a2_bias_photometric as B2
    meta = ctx['meta']
    at, sv, bid = meta['anomaly_type'].values, meta['severity'].values, meta['base_id'].values
    cl = np.flatnonzero(at == 'clean')
    assert len(np.unique(bid[cl])) == len(cl)
    pos = pd.Series(cl, index=bid[cl])
    calc = ctx['syn_ram'] - ctx['syn_ras']
    d0 = pd.Series(B2.delta_a(calc[cl], sp), index=bid[cl])
    out = []
    # spike: fixed at k = 1 on the RAM channel (RAS is unchanged by the additive types), the same for every k
    k1 = np.flatnonzero((at == 'A_spike') & np.isclose(sv, 1.0))
    dd1 = B2.delta_a(ctx['syn_ram'][k1] - ctx['syn_ram'][pos.loc[bid[k1]].values], sp)
    a = np.abs(dd1)
    hit = pd.Series(a >= 0.05 * a.max() if a.max() > 0 else np.zeros(len(a), bool), index=bid[k1])
    sel = np.flatnonzero(at == 'A_spike')
    out.append(pd.DataFrame({'anomaly_type': 'A_spike', 'severity': sv[sel], 'base_id': bid[sel],
                             'affected': hit.loc[bid[sel]].values}))
    # saturation / time warp: per severity, |dA change| > 1% of the median clean |dA|
    tol = 1e-2 * np.median(np.abs(d0.values))
    for t in ('S_saturation', 'W_timewarp'):
        sel = np.flatnonzero(at == t)
        dd = B2.delta_a(calc[sel], sp) - d0.loc[bid[sel]].values
        out.append(pd.DataFrame({'anomaly_type': t, 'severity': sv[sel], 'base_id': bid[sel], 'affected': np.abs(dd) > tol}))
    return pd.concat(out, ignore_index=True)


def check_masks(mk, key):
    """The recomputed masks must reproduce s2a2's frac_curves_changed for EVERY (type, severity) cell."""
    bs = pd.read_csv(BIAS)
    bs = bs[(bs['item'] == key) & bs['anomaly_type'].isin(TYPES)]
    fr = mk.groupby(['anomaly_type', 'severity'])['affected'].mean()
    assert len(fr) == len(bs), (key, len(fr), len(bs))
    for (t, k), f in fr.items():
        r = bs[(bs['anomaly_type'] == t) & np.isclose(bs['severity'], k)]
        assert len(r) == 1 and np.isclose(f, r['frac_curves_changed'].iloc[0]), (key, t, k, f)
    return len(fr)


def jsonl_recs():
    recs = {}
    for g in ('main', 'ca', 'm2'):
        for l in open(os.path.join(V2, g, 'Reports', 'comparison.jsonl'), encoding='utf-8'):
            if l.strip():
                d = json.loads(l)
                recs[(d['item'], d['method'], d['N'])] = d
    for l in open(PROZONE, encoding='utf-8'):        # q = 0.5 / 0.2 (six items), as s2d2.jsonl_recs
        if l.strip():
            d = json.loads(l)
            r = recs.get((d['item'], d['method'], d['N']))
            if r is not None:
                for q in (0.5, 0.2):
                    k = f'recall_S_q{q:g}'
                    if k in d['budgets'][str(BUDGET)]:
                        r['budgets'][str(BUDGET)][k] = d['budgets'][str(BUDGET)][k]
    return recs


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
    recs = jsonl_recs()
    for gname, g in R.GROUPS.items():
        for proc, synth in g['items']:
            if items and proc not in items:
                continue
            dst = os.path.join(OUT, f'flags_{proc}.parquet')
            if os.path.exists(dst):
                print(f'{proc}: done, skipping', flush=True)
                continue
            R.use_group(gname, OUT, 'affected')
            M.load_item = S.make_loader(gname, g['loader'])
            ctx = M.load_item(proc, synth)
            meta = ctx['meta']
            mk = affected_masks(ctx, sp_all[proc])
            n_chk = check_masks(mk, sp_all[proc]['key'])
            sel = np.flatnonzero(meta['anomaly_type'].isin(TYPES).values)
            base = pd.DataFrame({'anomaly_type': meta['anomaly_type'].values[sel], 'severity': meta['severity'].values[sel],
                                 'base_id': meta['base_id'].values[sel]})
            rows, chk = [], []
            for m in METHODS:
                mm = f'dae{ze[proc]}' if m == 'dae_elbow' else m
                seeds = M.SEEDS if mm in M.STOCHASTIC else [M.SEEDS[0]]
                for sd in seeds:
                    sc, _, _ = M.score_sets(mm, ctx, N_MAIN, sd)
                    thr = np.percentile(sc['holdout'], BUDGET)
                    rows.append(base.assign(item=proc, method=m, method_run=mm, seed=sd, flag=(sc['syn'] >= thr)[sel]))
                    if mm.startswith('dae'):
                        M.tf.keras.backend.clear_session()
                f = pd.concat([r for r in rows if r['method'].iloc[0] == m])
                rec = recs.get((proc, mm, N_MAIN))
                for (t, k), gk in f.groupby(['anomaly_type', 'severity']):
                    jl = rec['budgets'][str(BUDGET)].get(KEY[t].format(k), {}).get('mean', np.nan) if rec else np.nan
                    chk.append(dict(item=proc, method=m, method_run=mm, anomaly_type=t, severity=k, n_seeds=len(seeds),
                                    recall_all_rescored=float(gk['flag'].mean()), recall_all_jsonl=jl))
                print(f'  {proc} {mm}: scored ({len(seeds)} seed(s))', flush=True)
            out = pd.concat(rows, ignore_index=True)
            n0 = len(out)
            out = out.merge(mk, on=['anomaly_type', 'severity', 'base_id'], how='left', validate='many_to_one')
            assert len(out) == n0 and out['affected'].notna().all()
            ck = pd.DataFrame(chk)
            ck['abs_diff'] = (ck['recall_all_rescored'] - ck['recall_all_jsonl']).abs()
            assert ck['abs_diff'].notna().all(), f'{proc}: missing JSONL reference ' + str(ck[ck.abs_diff.isna()].head())
            assert ck.loc[ck['method'] != 'dae_elbow', 'abs_diff'].max() < 1e-12, f'{proc}: deterministic mismatch'
            ckp = os.path.join(OUT, 'recall_check.csv')
            if os.path.exists(ckp):                   # re-running an item replaces its rows
                old = pd.read_csv(ckp)
                ck = pd.concat([old[old['item'] != proc], ck], ignore_index=True)
            ck.to_csv(ckp, index=False)
            out.to_parquet(dst + '.tmp', index=False)
            os.replace(dst + '.tmp', dst)             # last: its presence marks the item as complete
            print(f'{proc}: masks checked on {n_chk} (type, severity) cells; max |rescored - JSONL| = '
                  + ', '.join(f'{m} {d:.4f}' for m, d in ck[ck.item == proc].groupby('method')['abs_diff'].max().items()),
                  flush=True)
            del ctx


def summarize():
    fl = pd.concat([pd.read_parquet(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))
                    if f.startswith('flags_') and f.endswith('.parquet')], ignore_index=True)
    keys = ['item', 'method', 'anomaly_type', 'severity']
    one = fl[fl['seed'] == fl.groupby(['item', 'method'])['seed'].transform('min')]
    per = pd.DataFrame({'recall_aff': fl[fl['affected']].groupby(keys)['flag'].mean(),
                        'recall_all': fl.groupby(keys)['flag'].mean(),
                        'recall_unaff': fl[~fl['affected']].groupby(keys)['flag'].mean(),
                        'n_affected': one.groupby(keys)['affected'].sum()}).reset_index()
    scored = set(fl['item'])
    assert len(scored) == 21, f'expected 21 scored items, got {len(scored)}: {sorted(scored)}'
    rows = []
    for ys, sub, fn in YARDSTICKS:
        cl = pd.read_csv(os.path.join(YS, sub, fn))
        assert (cl['yardstick'] == sub).all(), f'unexpected yardstick in {sub}/{fn}'
        cl = cl[cl['anomaly_type'].isin(TYPES) & (cl['bias_def'] == 'react_only')]
        for _, r in cl.iterrows():
            assert r['item'] in scored, f"missing flags_{r['item']}.parquet"      # no silent partial run
            k = r['severity_at_TEa']
            for m in METHODS:
                p = per[(per['item'] == r['item']) & (per['method'] == m) & (per['anomaly_type'] == r['anomaly_type'])
                        & np.isclose(per['severity'], k)] if pd.notna(k) else per.iloc[0:0]
                assert pd.isna(k) or len(p) == 1, (r['item'], r['anomaly_type'], m, k)
                g = lambda c: float(p[c].iloc[0]) if len(p) else np.nan
                rows.append(dict(yardstick=ys, item=r['item'], anomaly_type=r['anomaly_type'], limit_pct=r['limit_pct'],
                                 k_star=k, bias_at_k_star=r['bias_at_severity'], method=m,
                                 n_affected=int(p['n_affected'].iloc[0]) if len(p) else np.nan,
                                 recall_aff=g('recall_aff'), recall_unaff=g('recall_unaff'), recall_all=g('recall_all'),
                                 recall_all_s2d2=r[m] if pd.notna(k) else np.nan))
    by = pd.DataFrame(rows)
    reached = by.dropna(subset=['k_star'])
    assert reached[['recall_all', 'recall_aff', 'recall_all_s2d2']].notna().all().all()
    det = reached[reached['method'] != 'dae_elbow']
    assert (det['recall_all'] - det['recall_all_s2d2']).abs().max() < 1e-9, 'all-curve recall does not reproduce s2d2'
    by['diff_all_vs_s2d2'] = by['recall_all'] - by['recall_all_s2d2']
    by.to_csv(os.path.join(OUT, 'affected_by_item.csv'), index=False)
    summ = []
    for (ys, t, m), g in reached.groupby(['yardstick', 'anomaly_type', 'method'], sort=False):
        ev = by[(by.yardstick == ys) & (by.anomaly_type == t) & (by.method == m)]
        summ.append(dict(yardstick=ys, anomaly_type=t, method=m, n_items_evaluated=len(ev), n_items_reached=len(g),
                         recall_aff_mean=g['recall_aff'].mean(),
                         # curve-weighted alternative: items with few affected curves weigh less
                         recall_aff_pooled=float((g['recall_aff'] * g['n_affected']).sum() / g['n_affected'].sum()),
                         recall_aff_min=g['recall_aff'].min(),
                         recall_all_mean=g['recall_all'].mean(), recall_all_s2d2_mean=g['recall_all_s2d2'].mean(),
                         n_affected_total=int(g['n_affected'].sum()), n_affected_min=int(g['n_affected'].min()),
                         n_affected_max=int(g['n_affected'].max())))
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(OUT, 'affected_summary.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(s.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['score', 'summarize'])
    ap.add_argument('--items', nargs='+', default=None)
    a = ap.parse_args()
    score(a.items) if a.stage == 'score' else summarize()
