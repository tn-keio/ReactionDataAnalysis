# Stage 1o: extract reagent-lot numbers (R1_Lot, R3_Lot) from the raw instrument CSVs.
#
# The processed parquet (Scripts/clean_and_merge_bigdata.py) keeps only SystemName, S_ID, S_Date, A_Name,
# Value and the absorbances, so the lot columns were dropped at read time.  The raw CSVs carry the lot of
# every measurement (R1_Lot, R2_Lot, R3_Lot), which is more reliable than the hand-entered reagent log.
# This script re-reads only the key and lot columns, applies the same row filters as the pipeline
# (exclude QC by C1 in [20001, 30000], empty D_Alarm, S_Date < 2026-01-01), keeps the 21 panel assays,
# and writes one row per measurement.  The absorbance-completeness filter of the pipeline cannot be applied
# without reading the absorbances; rows are therefore matched to the processed data by key downstream.
# Output: Results/20260924_Rev_LotFromRaw/lots_rows.parquet, read_errors.csv
# py3.14.  Raw data stay local (sample identifiers are not sent anywhere).

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RAW = os.path.join(BASE, 'Data', 'BigData', 'rawdata')
OUT = os.path.join(BASE, 'Results', '20260924_Rev_LotFromRaw')
sys.path.insert(0, os.path.join(BASE, 'ForPaper'))
from _common import panel as P   # noqa: E402

PANEL = set(P.proc_names(21))
COLS = ['SystemName', 'S_ID', 'S_Date', 'A_Name', 'C1', 'D_Alarm', 'R1_Lot', 'R2_Lot', 'R3_Lot']
SCHEMA = pa.schema([('analyzer', pa.string()), ('S_ID', pa.string()), ('S_Date', pa.timestamp('ns')),
                    ('A_Name', pa.string()), ('R1_Lot', pa.string()), ('R2_Lot', pa.string()),
                    ('R3_Lot', pa.string()), ('src', pa.string())])


def read_one(path):
    try:
        if os.path.getsize(path) < 100:
            return None, None
        df = pd.read_csv(path, encoding='cp932', usecols=lambda c: c in COLS, dtype=str,
                         on_bad_lines='skip', low_memory=False)
        if any(c not in df.columns for c in COLS):
            return None, None
        df['A_Name'] = df['A_Name'].str.strip()
        df = df[df['A_Name'].isin(PANEL)]
        c1 = pd.to_numeric(df['C1'], errors='coerce')
        df = df[(c1 >= 20001) & (c1 <= 30000)]
        da = df['D_Alarm'].fillna('').str.strip()
        df = df[da == '']
        df = df.assign(S_Date=pd.to_datetime(df['S_Date'], errors='coerce')).dropna(subset=['S_Date'])
        df = df[df['S_Date'] < '2026-01-01']
        if df.empty:
            return None, None
        out = pd.DataFrame({
            'analyzer': df['SystemName'].str.strip().str.replace('LABOSPECT 008 ALPHA ', '', regex=False),
            'S_ID': df['S_ID'].astype(str).str.strip(),
            'S_Date': df['S_Date'].astype('datetime64[ns]'),
            'A_Name': df['A_Name'],
            'R1_Lot': df['R1_Lot'].fillna('').str.strip(),
            'R2_Lot': df['R2_Lot'].fillna('').str.strip(),
            'R3_Lot': df['R3_Lot'].fillna('').str.strip(),
            'src': os.path.relpath(path, RAW),
        })
        return out, None
    except Exception as exc:  # noqa: BLE001
        return None, f'{type(exc).__name__}: {exc}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=None, help='first N files only (trial)')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(RAW) for f in fs if f.lower().endswith('.csv'))
    if a.limit:
        files = files[:a.limit]
    print(f'{len(files)} files', flush=True)
    t0 = time.time()
    dst = os.path.join(OUT, 'lots_rows.parquet' if not a.limit else 'lots_rows_trial.parquet')
    writer = pq.ParquetWriter(dst, SCHEMA)
    errors, n_rows, done = [], 0, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(read_one, f): f for f in files}
        for fu in as_completed(futs):
            df, err = fu.result()
            done += 1
            if err:
                errors.append((futs[fu], err))
            if df is not None and len(df):
                writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
                n_rows += len(df)
            if done % 200 == 0:
                print(f'  {done}/{len(files)} files, {n_rows:,} rows, {time.time() - t0:.0f}s', flush=True)
    writer.close()
    pd.DataFrame(errors, columns=['path', 'error']).to_csv(os.path.join(OUT, 'read_errors.csv'), index=False)
    print(f'DONE {done} files, {n_rows:,} rows, {len(errors)} errors, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
