# Re-run the method comparison for Ca ONLY, on a CLEAN CONTEMPORANEOUS window (analyzer #2,
# 2021-09-01..2022-04-30) that avoids the Aug-2021 optical event and the multi-year drift that
# contaminated the original Ca 'clean' specificity (0.65).  Training + synthetic base curves are both
# from this window, so specificity reflects true method performance (not train/test period mismatch).
# Points 5-38, 14->1 item (Ca), 11 methods, N in {500,5000,10000}.  TF venv required.
import os, sys, tempfile
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import run_paper_method_comparison as M   # noqa: E402

P0 = 5
M.RAM = [f'RAM{i}' for i in range(P0, 39)]
M.RAS = [f'RAS{i}' for i in range(P0, 39)]
M.XPTS = np.arange(P0, 39)
M.ITEMS = [('Ca', 'CA')]
M.SYN = os.path.join(M.BASE_DIR, 'Data', 'BigData', 'synthetic_anomalies',
                     'endpoint_synthetic_anomalies_p5_38_Ca_clean.parquet')
M.OUT = os.path.join(M.BASE_DIR, 'Results', '20260718_MethodComparison_Ca_clean')
M.REPORTS = os.path.join(M.OUT, 'Reports')
M.CACHE = os.path.join(os.environ.get('LOCALAPPDATA', tempfile.gettempdir()), 'mc_ca_clean_cache')
M.JSONL = os.path.join(M.REPORTS, 'comparison.jsonl')
M.SELF_CAL_BUDGET = 99.5

WIN0, WIN1 = pd.Timestamp('2021-09-01'), pd.Timestamp('2022-04-30')


def clean_load_item(proc_name, synth_name):
    """Same as M.load_item but restricted to analyzer #2 in the clean contemporaneous window."""
    df = pd.read_parquet(M.PROC, filters=[('A_Name', '==', proc_name)],
                         columns=['S_Date', 'S_ID', 'SystemName'] + M.RAM + M.RAS)
    d = pd.to_datetime(df['S_Date'])
    sysv = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    df = df[((sysv == '#2') & (d >= WIN0) & (d <= WIN1)).values]
    df = df.sort_values('S_Date').reset_index(drop=True).iloc[:M.N_TRAINPOOL + M.N_HOLDOUT + M.N_ANALYSIS]
    ram = df[M.RAM].values.astype('float64'); ras = df[M.RAS].values.astype('float64')
    calc = ram - ras
    lo, hi = M.norm_fit(calc[:min(M.N_TRAINPOOL, len(calc))])
    Xn = M.apply_norm(calc, lo, hi)
    sl_tr = slice(0, M.N_TRAINPOOL); sl_ho = slice(M.N_TRAINPOOL, M.N_TRAINPOOL + M.N_HOLDOUT)
    sl_an = slice(M.N_TRAINPOOL + M.N_HOLDOUT, M.N_TRAINPOOL + M.N_HOLDOUT + M.N_ANALYSIS)
    s = pd.read_parquet(M.SYN, filters=[('item', '==', synth_name)])
    syn_ram = s[M.RAM].values.astype('float64'); syn_ras = s[M.RAS].values.astype('float64')
    syn_calc = syn_ram - syn_ras
    return dict(
        train_pool=Xn[sl_tr], holdout=Xn[sl_ho], analysis=Xn[sl_an],
        syn=M.apply_norm(syn_calc, lo, hi),
        meta=s[['anomaly_type', 'severity', 'base_id']].reset_index(drop=True),
        an_ram=ram[sl_an], an_ras=ras[sl_an], an_sid=df['S_ID'].values[sl_an].astype(str),
        syn_ram=syn_ram, syn_ras=syn_ras,
        tr_calc_mean=calc[sl_tr].mean(0), tr_calc_std=calc[sl_tr].std(0), norm=(lo, hi))


M.load_item = clean_load_item

if __name__ == '__main__':
    assert os.path.exists(M.SYN), f'Missing Ca-clean synthetic: {M.SYN} (run generate_ca_clean_synthetic.py)'
    print(f'Ca CLEAN re-run @ points {P0}-38 | window #2 {WIN0.date()}..{WIN1.date()} | out={M.OUT}', flush=True)
    M.run(M.ITEMS, M.METHODS_OUT, do_list=True)
