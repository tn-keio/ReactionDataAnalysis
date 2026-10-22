# Stage 1d-2 (R2 follow-up): where do the two abnormal curves of the Cu case of Section 3.11
# (initial -5 on #3, 3-fold repeat 93 on #2) fall in the rolling evaluation of s1d, and with what
# ratio to the threshold are they scored there?  The 09-25 log quotes 6,047x and 309x without a stored output.
# Same set-up as s1d (pooled analyzers, monthly anchor, train = 500 curves before it, test = next 500,
# scalar min-max from the training window, no clipping, empirical Mahalanobis, self-calibrated P99.5);
# s1d's own functions are imported, nothing is re-implemented except locating the two rows.
# Output: Results/20260917_Rev_R2_RollingContemporaneous/cu_case_in_rolling.csv
import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

import s1d_rolling_contemporaneous as D

# the case's sample date is read from the (unpublished) case file written when the case was located
CASE_ROWS = os.path.join(D.BASE, 'Results', '20260925_Rev_CuCase2', 'raw_rows.csv')
VALUES = {-5.0: 'initial', 93.0: '3-fold repeat', 99.0: '5-fold repeat'}


def main():
    d = ds.dataset(D.PROC, partitioning='hive')
    cols = [c for c in ['S_Date', 'M_Date', 'Value', 'SystemName'] if c in d.schema.names]
    tb = d.to_table(columns=cols + D.RAM + D.RAS, filter=pc.field('A_Name') == 'Cu').to_pandas()
    tb = tb.sort_values('S_Date', kind='mergesort').reset_index(drop=True)      # the order s1d uses
    dates = pd.to_datetime(tb['S_Date']).values.astype('datetime64[ns]')
    calc = tb[D.RAM].values.astype('float64') - tb[D.RAS].values.astype('float64')
    val = pd.to_numeric(tb['Value'], errors='coerce')
    DAY = pd.Timestamp(pd.read_csv(CASE_ROWS, dtype=str)['S_Date'].min()).strftime('%Y-%m-%d')
    day = pd.to_datetime(tb['S_Date']).dt.strftime('%Y-%m-%d') == DAY
    hits = tb.index[day & val.isin(list(VALUES))].tolist()
    print('rows on', DAY, 'with case values:', [(i, val[i], tb.get('SystemName', pd.Series(['?'] * len(tb)))[i]) for i in hits])
    rows = []
    m0 = pd.Timestamp(DAY).to_period('M')
    months = pd.period_range(m0 - 1, m0 + 1, freq='M')
    for mo in months:
        cut = int(np.searchsorted(dates, np.datetime64(mo.start_time, 'ns'), side='left'))
        if pd.Timestamp(dates[cut]).to_period('M') != mo:
            continue
        W = calc[cut - D.N:cut]; lo, hi = D.TS.norm_fit(W)
        f = D.fit(D.norm(W, lo, hi, False), 'E')
        thr = float(np.percentile(f(D.norm(W, lo, hi, False)), D.BUDGET))
        S = slice(cut, cut + D.M_TEST)
        flags = f(D.norm(calc[S], lo, hi, False)) >= thr
        for i in hits:
            if cut <= i < cut + D.M_TEST:
                s = float(f(D.norm(calc[i:i + 1], lo, hi, False))[0])
                rows.append(dict(block=str(mo), block_start=str(dates[cut])[:10], block_end=str(dates[cut + D.M_TEST - 1])[:10],
                                 row=i, value=val[i], measurement=VALUES[val[i]], d2=round(s, 1), thr=round(thr, 2),
                                 ratio=round(s / thr, 1), flagged=s >= thr,
                                 block_flags=int(flags.sum()), block_flag_rate=float(flags.mean())))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(D.OUT, 'cu_case_in_rolling.csv'), index=False)


if __name__ == '__main__':
    main()
