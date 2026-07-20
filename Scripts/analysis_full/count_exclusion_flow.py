# CONSORT-style exclusion counts for the dataset (§2/Suppl).  Mirrors clean_and_merge_bigdata.py EXACTLY
# (same order & conditions) but only COUNTS rows at each stage — no parquet is written.
# Stages (per file, in pipeline order):
#   raw -> [1] item exclusion (ISE/serum) -> [2] QC (C1 in 20001-30000) -> [3] alert (D_Alarm empty)
#       -> [4] missing absorbance (RAS/RAM) -> [5] valid date -> [6] period (<2026-01-01) = FINAL
# Per-file counts are checkpointed (append) so the run is resumable across kills.  py3.14 (default py).
import os, csv, sys, time
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RAWDATA = os.path.join(BASE, 'Data', 'BigData', 'rawdata')
OUT = os.path.join(BASE, 'Results', '20260718_ExclusionFlow')
CKPT = os.path.join(OUT, 'per_file_counts.csv')

EXCLUDE_ITEMS = {'Na', 'K', 'Cl', 'L', 'H', 'I'}
META = ['SystemName', 'S_ID', 'S_Date', 'A_Name', 'Value']
RAS = [f'RAS{i}' for i in range(1, 39)]; RAM = [f'RAM{i}' for i in range(1, 39)]
FILT = ['C1', 'D_Alarm']
READ = META + RAS + RAM + FILT
STAGES = ['n_raw', 'n_item', 'n_qc', 'n_alert', 'n_missing', 'n_datevalid', 'n_final']


def count_file(path):
    if os.path.getsize(path) < 100:
        return None
    df = pd.read_csv(path, encoding='cp932', usecols=lambda c: c in READ, on_bad_lines='skip', low_memory=False)
    if [c for c in READ if c not in df.columns]:
        return None                                        # crucial cols missing -> file skipped (as in pipeline)
    n_raw = len(df)
    df = df[~df['A_Name'].isin(EXCLUDE_ITEMS)]; n_item = len(df)
    df['C1'] = pd.to_numeric(df['C1'], errors='coerce')
    df = df[(df['C1'] >= 20001) & (df['C1'] <= 30000)]; n_qc = len(df)
    df['D_Alarm'] = df['D_Alarm'].astype(str).str.strip().replace({'nan': '', 'None': ''})
    df = df[(df['D_Alarm'].isna()) | (df['D_Alarm'] == '')]; n_alert = len(df)
    for col in ['Value'] + RAS + RAM:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=RAS + RAM); n_missing = len(df)
    df['S_Date'] = pd.to_datetime(df['S_Date'], errors='coerce')
    df = df.dropna(subset=['S_Date']); n_datevalid = len(df)
    df = df[df['S_Date'] < '2026-01-01']; n_final = len(df)
    return dict(zip(STAGES, [n_raw, n_item, n_qc, n_alert, n_missing, n_datevalid, n_final]))


def main():
    os.makedirs(OUT, exist_ok=True)
    files = []
    for root, _, fs in os.walk(RAWDATA):
        for f in fs:
            if f.lower().endswith('.csv'):
                files.append(os.path.join(root, f))
    files.sort()
    done = set()
    if os.path.exists(CKPT):
        for r in csv.DictReader(open(CKPT, encoding='utf-8')):
            done.add(r['file'])
    todo = [f for f in files if os.path.relpath(f, RAWDATA) not in done]
    print(f'{len(files)} csv files, {len(done)} done, {len(todo)} to go', flush=True)
    new = not os.path.exists(CKPT)
    fh = open(CKPT, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(fh, fieldnames=['file', 'skipped'] + STAGES)
    if new:
        w.writeheader()
    t0 = time.time()
    for k, path in enumerate(todo):
        rel = os.path.relpath(path, RAWDATA)
        try:
            c = count_file(path)
        except Exception as e:
            c = None
            print(f'  ERR {rel}: {type(e).__name__}', flush=True)
        if c is None:
            w.writerow({'file': rel, 'skipped': 1, **{s: 0 for s in STAGES}})
        else:
            w.writerow({'file': rel, 'skipped': 0, **c})
        if (k + 1) % 25 == 0:
            fh.flush()
            print(f'  {k+1}/{len(todo)}  ({(time.time()-t0)/60:.1f} min)', flush=True)
    fh.flush(); fh.close()

    # aggregate
    tot = {s: 0 for s in STAGES}; nsk = 0; nfiles = 0
    for r in csv.DictReader(open(CKPT, encoding='utf-8')):
        nfiles += 1; nsk += int(r['skipped'])
        for s in STAGES:
            tot[s] += int(r[s])
    print('\n=== EXCLUSION FLOW (aggregate over raw CSVs) ===', flush=True)
    print(f'files: {nfiles} (skipped/no-cols: {nsk})')
    prev = tot['n_raw']
    labels = {'n_raw': 'raw rows', 'n_item': 'after [1] item exclusion (ISE/serum)',
              'n_qc': 'after [2] QC exclusion (C1 20001-30000)', 'n_alert': 'after [3] alert exclusion (D_Alarm)',
              'n_missing': 'after [4] missing-absorbance drop', 'n_datevalid': 'after [5] valid date',
              'n_final': 'after [6] period exclusion (<2026) = FINAL'}
    for s in STAGES:
        removed = (tot['n_raw'] if s == 'n_raw' else prev) - tot[s]
        print(f'  {labels[s]:48} {tot[s]:>12,}  (removed {removed:,})' if s != 'n_raw'
              else f'  {labels[s]:48} {tot[s]:>12,}')
        prev = tot[s]
    print(f'\nprocessed parquet total (expected) = 22,781,420  |  computed final = {tot["n_final"]:,}')
    with open(os.path.join(OUT, 'exclusion_flow_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f); wr.writerow(['stage', 'label', 'n_rows'])
        for s in STAGES:
            wr.writerow([s, labels[s], tot[s]])
    print('saved exclusion_flow_summary.csv')


if __name__ == '__main__':
    main()
