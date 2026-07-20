# -*- coding: utf-8 -*-
"""
Convert the shipped Data/*.csv sample into the Hive-partitioned Parquet store that
the analysis scripts in Scripts/analysis_full/ expect
(<BASE>/Data/BigData/processed/A_Name=<item>/...).

This bridges the format gap so the analysis code can be run against the shipped
sample without the full dataset. It maps the de-identified `sample_id` column to
the `S_ID` column the scripts read.

IMPORTANT — what this does and does NOT enable:
  * It DOES let the core scripts (method comparison, estimator selection, synthetic
    validity, point-wise variability) run end-to-end on the sample, producing
    *sample-scale* results (smaller and noisier than the paper's 15,000-curve/item
    numbers).
  * It does NOT reproduce results that need the full dataset, namely:
      - the known Cu anomaly (real_anomaly_cu_validation.py): the specific curve is
        not in the 2,000-curve random sample, and IDs are surrogate keys;
      - maintenance / reagent-lot robustness and cross-analyzer / temporal analyses,
        which need the full multi-year volume and the instrument logs.
    Run those against the full dataset (available on request; see README).

Usage:
    python Scripts/csv_to_parquet.py
Requires: pandas, pyarrow.
"""
import os
import glob
import shutil
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = os.path.join(REPO, 'Data')
PROC = os.path.join(DATA, 'BigData', 'processed')

SKIP = {'items_info.csv', 'manifest.csv'}


def main():
    csvs = [f for f in glob.glob(os.path.join(DATA, '*.csv'))
            if os.path.basename(f) not in SKIP]
    if not csvs:
        raise SystemExit(f'No item CSVs found in {DATA}')

    if os.path.isdir(PROC):
        shutil.rmtree(PROC)
    os.makedirs(PROC, exist_ok=True)

    total = 0
    for f in sorted(csvs):
        item = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        if 'sample_id' in df.columns and 'S_ID' not in df.columns:
            df = df.rename(columns={'sample_id': 'S_ID'})
        df['S_ID'] = df['S_ID'].astype(str)
        # write one Hive partition per item: A_Name=<item>/data.parquet
        part = os.path.join(PROC, f'A_Name={item}')
        os.makedirs(part, exist_ok=True)
        df.drop(columns=['A_Name'], errors='ignore').to_parquet(
            os.path.join(part, 'data.parquet'), engine='pyarrow', index=False)
        total += len(df)
        print(f'  {item:8} {len(df):>6} rows')

    print(f'\nWrote {total} rows across {len(csvs)} items to:\n  {PROC}')
    print('The analysis scripts read this via '
          "pd.read_parquet(PROC, filters=[('A_Name','==', <item>)], ...).")


if __name__ == '__main__':
    main()
