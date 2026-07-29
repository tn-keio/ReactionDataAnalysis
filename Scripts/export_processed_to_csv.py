"""Export a GitHub-shareable CSV sample of the reaction-curve dataset.

Reads the internal processed Parquet store (Hive-partitioned by A_Name; columns
SystemName, S_ID, S_Date, Value, RAS1..RAS38, RAM1..RAM38 -- ALL 38 photometric points) and
writes one CSV per ForPaper item into Data/, keeping the processed column layout.

Design choices (see Data/README in this repo):
  * Scope   : the 21 ForPaper=1 items (parameter_mapped.csv). All are 2-point endpoint assays.
  * Sample  : up to SAMPLE_N rows per item, drawn at random over the full period (seed=SEED),
              then sorted by S_Date. Full data (~13M rows) is too large for GitHub.
  * Points  : all 38 points are kept (RAS1..38, RAM1..38) for full reproducibility; the analysis
              window (points 5..38) is a downstream selection, not baked into the shared data.
  * Privacy : the raw sample barcode S_ID is DE-IDENTIFIED to a surrogate key `sample_id`
              (deterministic, consistent across items so cross-item linkage is preserved, but the
              barcode itself is removed). Re-run with ANONYMIZE_SID=False only if your ethics
              approval permits publishing raw sample IDs.

This script documents provenance; external users cannot run it (they do not have the processed
store). It is shipped for transparency.
"""
import os
import glob
import hashlib
import pandas as pd
import pyarrow.dataset as ds

# ---- config ----
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))                 # repository root
MAIN = os.path.abspath(os.path.join(REPO, '..'))                 # main project root
PROC = os.path.join(MAIN, 'Data', 'BigData', 'processed')
PARAM = os.path.join(MAIN, 'Data', 'parameter_mapped.csv')
OUT = os.path.join(REPO, 'Data')

SAMPLE_N = 2000
SEED = 42
ANONYMIZE_SID = True

RAS = [f'RAS{i}' for i in range(1, 39)]
RAM = [f'RAM{i}' for i in range(1, 39)]
ABS_COLS = RAS + RAM
COLS = ['SystemName', 'S_ID', 'S_Date', 'Value'] + ABS_COLS


def resolve_partitions():
    """Map ForPaper item names (parameter_mapped) to processed A_Name partition dirs (case-insensitive)."""
    parts = {os.path.basename(p).replace('A_Name=', ''): p
             for p in glob.glob(os.path.join(PROC, 'A_Name=*'))}
    low = {k.lower(): k for k in parts}
    pm = pd.read_csv(PARAM)
    fp = pm[pm['ForPaper'] == 1]['ItemName'].tolist()
    mapping = {}
    for name in fp:
        key = low.get(name.lower())
        if key is None:
            raise SystemExit(f'ForPaper item {name!r} has no processed partition')
        mapping[key] = parts[key]                                # keyed by canonical processed A_Name
    return mapping


def surrogate(sid: str) -> str:
    """Deterministic anonymized id for a raw sample barcode (stable across items)."""
    h = hashlib.sha256(str(sid).encode('utf-8')).hexdigest()[:12]
    return f'S{h}'


def main():
    os.makedirs(OUT, exist_ok=True)
    mapping = resolve_partitions()
    manifest = []
    for aname, pdir in sorted(mapping.items()):
        tb = ds.dataset(pdir).to_table(columns=COLS).to_pandas()
        n_total = len(tb)
        if n_total > SAMPLE_N:
            tb = tb.sample(n=SAMPLE_N, random_state=SEED)
        tb = tb.sort_values('S_Date').reset_index(drop=True)
        # absorbance counts are integers -> write clean integers
        tb[ABS_COLS] = tb[ABS_COLS].round().astype('int64')
        tb.insert(0, 'A_Name', aname)
        if ANONYMIZE_SID:
            tb['S_ID'] = tb['S_ID'].map(surrogate)
        tb = tb.rename(columns={'S_ID': 'sample_id'})
        tb['S_Date'] = pd.to_datetime(tb['S_Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        out_cols = ['A_Name', 'SystemName', 'sample_id', 'S_Date', 'Value'] + ABS_COLS
        fn = os.path.join(OUT, f'{aname}.csv')
        tb[out_cols].to_csv(fn, index=False, encoding='utf-8')
        manifest.append({'item': aname, 'file': f'{aname}.csv',
                         'n_rows_exported': len(tb), 'n_rows_total': n_total,
                         'date_min': tb['S_Date'].min(), 'date_max': tb['S_Date'].max()})
        print(f'  {aname:8s} {len(tb):5d}/{n_total:>8d} -> {aname}.csv', flush=True)

    man = pd.DataFrame(manifest)
    man.to_csv(os.path.join(OUT, 'manifest.csv'), index=False)
    print(f'\n{len(man)} items, {man.n_rows_exported.sum()} rows exported (of {man.n_rows_total.sum()} total).')
    print('-> ', OUT)


if __name__ == '__main__':
    main()
