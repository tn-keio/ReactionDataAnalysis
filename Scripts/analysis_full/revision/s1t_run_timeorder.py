# Stage 1t (run): re-run the main analyses on the time-ordered copy of the processed data (s1t_build_timeordered).
# Every script is used unchanged; only its data path (and output folder) is redirected, exactly as the
# unclipped re-runs were done (s1b, s1d, s1e, s1f, s1g, s1m).  All runs are unclipped.
#   classical : Table 1 non-deep detectors (maha, maha_robust, pca, lof), N 500/5,000/10,000, 21 assays  [TF env]
#   dl        : minimal deep-learning check: v2 benchmark, N = 500, CRE and TP          [TF env]
#   r2        : R2 rolling evaluation (s1d)
#   burden    : Table S3 workload (rev06), fixed-reference IgM/IgG (rev11), per-analyzer lead times (rev13)
#   cu        : Cu real case (s1e)
#   s4        : transferability / pooling (s1f --which inter)
#   s1m       : per-analyzer fixed reference (--items ...; then --summarize)
# Output: Results/20260925_Rev_TimeOrder/<analysis>/  (dl: Results/20260917_Rev_v2Benchmark_timeorder)

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, '..'))):
    if p not in sys.path:
        sys.path.insert(0, p)

DST = os.environ.get('TIMEORDER_DIR') or os.path.abspath(os.path.join(HERE, '..', '..', '..', 'Data', 'BigData', 'processed_timeordered'))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(BASE, 'Results', '20260925_Rev_TimeOrder')


def run_classical(items):
    import rev_common as R
    M = R.M
    M.PROC = DST
    R.set_clip(False)
    methods = ['maha', 'maha_robust', 'pca', 'lof']
    M.METHODS_OUT = list(methods)
    out = os.path.join(OUT, 'classical')
    for gname in R.GROUPS:
        g = R.use_group(gname, out, 'timeorder_classical')
        its = [it for it in g['items'] if items is None or it[0] in items]
        if its:
            print(f'=== classical group={gname} items={[i[0] for i in its]}', flush=True)
            M.run(its, methods, do_list=False)


def run_dl(items):
    import s2b_benchmark_v2 as V
    V.M.PROC = DST
    V.M.N_GRID = [500]
    sys.argv = ['s2b', '--tag', 'timeorder', '--groups', 'main', '--items'] + (items or ['CRE', 'TP'])
    V.main()


def run_r2(items):
    import s1d_rolling_contemporaneous as D
    D.PROC = DST
    D.OUT = os.path.join(OUT, 'r2')
    sys.argv = ['s1d'] + (['--items'] + items if items else [])
    D.main()


def run_burden(which):
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    import s1g_burden_fixedref_noclip as G
    G.B.PROC = DST
    G.OUT = os.path.join(OUT, 'burden')
    G.B.apply_norm = G.apply_norm_noclip
    G.P.out_dir = G.out_dir_redirect
    for key in which:
        for folder, fname, argv in G.SCRIPTS[key]:
            mod = G.load_by_path(folder, fname)
            assert mod.B is G.B and mod.P is G.P
            if hasattr(mod, 'load_curves'):          # Rev13 builds its data path itself
                def load_curves(a_name, analyzer, _mod=mod):
                    d = ds.dataset(DST, partitioning='hive')
                    tb = d.to_table(columns=['S_Date', 'SystemName'] + G.B.RAM + G.B.RAS,
                                    filter=pc.field('A_Name') == a_name).to_pandas()
                    tb['sys'] = tb['SystemName'].astype(str).str.replace('LABOSPECT 008 ALPHA ', '', regex=False).str.strip()
                    tb = tb[tb['sys'] == analyzer]
                    if tb.empty:
                        return None, None
                    tb['S_Date'] = pd.to_datetime(tb['S_Date'])
                    tb = tb.sort_values('S_Date').reset_index(drop=True)
                    calc = tb[G.B.RAM].values.astype('float64') - tb[G.B.RAS].values.astype('float64')
                    return pd.to_datetime(tb['S_Date'].values), calc
                mod.load_curves = load_curves
            print(f'===== {key}: {folder}/{fname} (time order, noclip) =====', flush=True)
            sys.argv = [fname] + argv
            mod.main()


def run_cu():
    import s1e_cu_case_noclip as E
    E.C.PROC = DST
    E.OUT = os.path.join(OUT, 'cu')
    E.main()


def run_s4():
    import s1f_event_interanalyzer_noclip as F
    F.IA.PROC = DST
    F.OUT = os.path.join(OUT, 's4')
    sys.argv = ['s1f', '--which', 'inter']
    F.main()


def run_s1m(items, summarize):
    import s1m_fixed_reference_per_analyzer as S
    S.B.PROC = DST
    S.OUT = os.path.join(OUT, 's1m')
    os.makedirs(S.OUT, exist_ok=True)
    if summarize:
        S.summarize()
        return
    for it in items:
        if not os.path.exists(os.path.join(S.OUT, f'monthly_{it}.csv')):
            S.run_item(it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['classical', 'dl', 'r2', 'burden', 'cu', 's4', 's1m'])
    ap.add_argument('--items', nargs='+', default=None)
    ap.add_argument('--which', nargs='+', default=['rev06', 'rev11', 'rev13'])
    ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {'classical': lambda: run_classical(a.items), 'dl': lambda: run_dl(a.items), 'r2': lambda: run_r2(a.items),
     'burden': lambda: run_burden(a.which), 'cu': run_cu, 's4': run_s4,
     's1m': lambda: run_s1m(a.items or [], a.summarize)}[a.what]()


if __name__ == '__main__':
    main()
