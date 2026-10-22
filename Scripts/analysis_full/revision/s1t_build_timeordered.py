# Stage 1t (build): a copy of the processed data for the 21 panel assays in which the within-day order of the
# curves is chronological.  S_Date (date only) is replaced by the sample date plus the time of day of the
# per-test measurement time M_Date (from s1r), clipped to the same calendar day, so that no curve changes its
# date, month or date-based split membership; only the order inside a day changes.  The original processed
# folder is not touched; the copy is written to local disk (C:), outside Google Drive.
# Validation per assay: same row count, identical calendar dates, identical multiset of absorbances.
# Output: <TIMEORDER_DIR>/A_Name=<item>/part-0.parquet, build_report.csv (in the parent folder);
#         TIMEORDER_DIR defaults to Data/BigData/processed_timeordered
# py3.14.

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import s1s_s9_exact_time as S               # noqa: E402  (attach_time)
from _common import panel as P              # noqa: E402  (path set up by s1s imports)

BASE = S.EV.BASE
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
DST = os.environ.get('TIMEORDER_DIR') or os.path.abspath(os.path.join(HERE, '..', '..', '..', 'Data', 'BigData', 'processed_timeordered'))
DAY = pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def build(item):
    src = ds.dataset(PROC, partitioning='hive')
    tb = src.to_table(filter=pc.field('A_Name') == item)
    cols = [c for c in tb.column_names if c != 'A_Name']
    schema = pa.schema([tb.schema.field(c) for c in cols])
    df = tb.select(cols).to_pandas()
    df['d'] = pd.to_datetime(df['S_Date'])
    df['sys'] = df['SystemName'].astype(str).str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    m, rep = S.attach_time(item, df)          # returns rows in the original order with M_Date
    day = m['d'].dt.normalize()
    tod = (m['M_Date'] - day).clip(lower=pd.Timedelta(0), upper=DAY)
    new = day + tod.fillna(pd.Timedelta(0))
    out = df[cols].copy()
    out['S_Date'] = new.values.astype(out['S_Date'].dtype) if out['S_Date'].dtype.kind == 'M' else new.values
    lag = (m['M_Date'].dt.normalize() - day).dt.days
    # validation
    ok_n = len(out) == len(df)
    ok_day = bool((pd.to_datetime(out['S_Date']).dt.normalize().values == day.values).all())
    ram = [c for c in cols if c.startswith(('RAM', 'RAS'))]
    ok_abs = bool(np.array_equal(out[ram].to_numpy(), df[ram].to_numpy(), equal_nan=True))
    d = os.path.join(DST, f'A_Name={item}')
    os.makedirs(d, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False), os.path.join(d, 'part-0.parquet'))
    rep.update(ok_rows=ok_n, ok_dates=ok_day, ok_absorbance=ok_abs, time_missing=float(m['M_Date'].isna().mean()),
               measured_next_day=float((lag > 0).mean()), measured_before=float((lag < 0).mean()))
    print(f"  {item}: n={len(out):,} matched {rep['matched']:.4f} rows ok={ok_n} dates ok={ok_day} abs ok={ok_abs}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=P.proc_names(21))
    a = ap.parse_args()
    os.makedirs(DST, exist_ok=True)
    reps = [build(it) for it in a.items]
    pth = os.path.join(os.path.dirname(DST), 'build_report.csv')   # outside the dataset folder
    old = pd.read_csv(pth) if os.path.exists(pth) else pd.DataFrame()
    new = pd.DataFrame(reps)
    pd.concat([old[~old.item.isin(new.item)] if len(old) else old, new]).to_csv(pth, index=False)
    print(new.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
