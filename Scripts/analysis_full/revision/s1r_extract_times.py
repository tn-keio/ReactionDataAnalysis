# Stage 1r: one pass over the raw instrument CSVs to extract
#   (A) every raw row of the Cu real-anomaly sample (all assays, all dates, no filters: repeats,
#       dilutions and alarm-flagged rows included) -> the original and repeat Cu results;
#   (B) a per-measurement time table for the 21 panel assays (same row filters as the pipeline) with the
#       per-test measurement time M_Date, Value and the reagent lot -> exact lot-change onsets for S9.
# The sample identifier is never printed or written in clear: outputs carry a salted hash.
# Output (local only): Results/20260924_Rev_TimeFromRaw/{cu_case_rows.csv, times_rows.parquet}
# py3.14.

import os
import sys
import time
import hashlib
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'ForPaper'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
from _common import panel as P            # noqa: E402
import real_anomaly_cu_validation as CU   # noqa: E402  (ANOM_SID; never printed)

RAW = os.path.join(BASE, 'Data', 'BigData', 'rawdata')
OUT = os.path.join(BASE, 'Results', '20260924_Rev_TimeFromRaw')
PANEL = set(P.proc_names(21))
TARGET = str(CU.ANOM_SID).strip()
SALT = 'rev2026-cu'
CU_COLS = ['SystemName', 'S_ID', 'S_Date', 'F_Date', 'A_Date', 'M_Date', 'A_Name', 'A_Unit', 'Value', 'Pre_Dil',
           'S_Type', 'Cup', 'R_Type1', 'R_Type2', 'C1', 'D_Alarm', 'R1_Lot', 'Order', 'R_Orgn', 'R_Exist']
T_COLS = ['SystemName', 'S_ID', 'S_Date', 'M_Date', 'A_Name', 'Value', 'C1', 'D_Alarm', 'R1_Lot']
SCHEMA = pa.schema([('analyzer', pa.string()), ('S_ID', pa.string()), ('S_Date', pa.timestamp('ns')),
                    ('M_Date', pa.timestamp('ns')), ('A_Name', pa.string()), ('Value', pa.string()),
                    ('R1_Lot', pa.string())])


def h(s):
    return hashlib.sha256((SALT + str(s)).encode()).hexdigest()[:10]


def read_one(path):
    try:
        if os.path.getsize(path) < 100:
            return None, None, None
        df = pd.read_csv(path, encoding='cp932', usecols=lambda c: c in set(CU_COLS) | set(T_COLS), dtype=str,
                         on_bad_lines='skip', low_memory=False)
        if any(c not in df.columns for c in T_COLS):
            return None, None, None
        df['S_ID'] = df['S_ID'].astype(str).str.strip()
        cu = df[df['S_ID'] == TARGET].copy()
        if len(cu):
            cu['S_ID'] = cu['S_ID'].map(h)
            cu['src'] = os.path.basename(path)
        t = df.copy()
        t['A_Name'] = t['A_Name'].str.strip()
        t = t[t['A_Name'].isin(PANEL)]
        c1 = pd.to_numeric(t['C1'], errors='coerce')
        t = t[(c1 >= 20001) & (c1 <= 30000) & (t['D_Alarm'].fillna('').str.strip() == '')]
        t = t.assign(S_Date=pd.to_datetime(t['S_Date'], errors='coerce'),
                     M_Date=pd.to_datetime(t['M_Date'], errors='coerce')).dropna(subset=['S_Date'])
        t = t[t['S_Date'] < '2026-01-01']
        out = pd.DataFrame({
            'analyzer': t['SystemName'].str.strip().str.replace('LABOSPECT 008 ALPHA ', '', regex=False),
            'S_ID': t['S_ID'], 'S_Date': t['S_Date'].astype('datetime64[ns]'),
            'M_Date': t['M_Date'].astype('datetime64[ns]'), 'A_Name': t['A_Name'],
            'Value': t['Value'].fillna('').str.strip(), 'R1_Lot': t['R1_Lot'].fillna('').str.strip()})
        return out, (cu if len(cu) else None), None
    except Exception as exc:  # noqa: BLE001
        return None, None, f'{type(exc).__name__}: {exc}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(RAW) for f in fs if f.lower().endswith('.csv'))
    print(f'{len(files)} files', flush=True)
    t0 = time.time()
    w = pq.ParquetWriter(os.path.join(OUT, 'times_rows.parquet'), SCHEMA)
    cus, errs, n = [], [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(read_one, f): f for f in files}
        for i, fu in enumerate(as_completed(futs), 1):
            t, cu, err = fu.result()
            if err:
                errs.append((futs[fu], err))
            if t is not None and len(t):
                w.write_table(pa.Table.from_pandas(t, schema=SCHEMA, preserve_index=False))
                n += len(t)
            if cu is not None:
                cus.append(cu)
            if i % 500 == 0:
                print(f'  {i}/{len(files)} files, {n:,} rows, {time.time() - t0:.0f}s', flush=True)
    w.close()
    cu = pd.concat(cus, ignore_index=True) if cus else pd.DataFrame(columns=CU_COLS)
    cu.to_csv(os.path.join(OUT, 'cu_case_rows.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(errs, columns=['path', 'error']).to_csv(os.path.join(OUT, 'read_errors.csv'), index=False)
    print(f'DONE {len(files)} files, {n:,} panel rows, {len(cu)} rows of the Cu-case sample, {len(errs)} errors, '
          f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
