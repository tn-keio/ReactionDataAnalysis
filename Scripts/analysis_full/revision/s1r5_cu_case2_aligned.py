# Stage 1r-5 (Reviewer #2 D1, option B): the second Cu case (M-protein, initial -5, 3x-diluted repeat 93,
# 5x-diluted repeat 99 reported) recomputed with exactly the protocol of the paper's Cu case (S10):
#   real_anomaly_cu_validation.run() with s1e's unclipped maha_fit (and the original clipped one, for the M1
#   response); train on the 500 most recent curves before the
#   case date (single analyzer = the analyzer that measured the curve, the other analyzer, and pooled #2+#3);
#   windows 5-38 and 1-38; test set = same analyzer, case date .. +70 days, case rows excluded.
# All three curves are taken from the raw files (repeat 2 is not in processed: C1/S_Date blank); the first two are
# checked against processed.  Adds the distance-scale multiple, points outside the training range (would-be
# clipped), and per-point |z| summaries per phase (for M2).
# Output: Results/20261019_Rev_CuCase2_Aligned/  (local only; contains result values, no sample IDs)
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, '..'))):
    if p not in sys.path:
        sys.path.insert(0, p)
import real_anomaly_cu_validation as C     # noqa: E402
import s1e_cu_case_noclip as E             # noqa: E402

ORIG_FIT = C.maha_fit
RAW = os.path.join(C.BASE, 'Data', 'BigData', 'rawdata')
CASE_ROWS = os.path.join(C.BASE, 'Results', '20260925_Rev_CuCase2', 'raw_rows.csv')
OUT = os.path.join(C.BASE, 'Results', '20261019_Rev_CuCase2_Aligned')


def case_day():
    """Sample date of the case, read from the (unpublished) case file."""
    return pd.Timestamp(pd.read_csv(CASE_ROWS, dtype=str)['S_Date'].min()).normalize()


LABEL = {-5.0: 'initial', 93.0: 'repeat1_3x', 99.0: 'repeat2_5x_reported'}


def raw_case_rows(sid):
    tag = case_day().strftime('%y%m%d')      # raw export files are named by measurement day (YYMMDD)
    rows = []
    for r, _, fs in os.walk(RAW):
        for f in fs:
            if tag not in f or not f.lower().endswith('.csv'):
                continue
            d = pd.read_csv(os.path.join(r, f), encoding='cp932', dtype=str, on_bad_lines='skip')
            d = d[(d['A_Name'].str.strip() == 'Cu') & (d['S_ID'].astype(str).str.strip() == sid)]
            if len(d):
                rows.append(d.assign(file=f))
    return pd.concat(rows, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    sid = str(pd.read_csv(CASE_ROWS, dtype=str)['S_ID'].iloc[0]).strip()
    raw = raw_case_rows(sid)
    raw['sys'] = raw['SystemName'].str[-2:]
    raw['v'] = pd.to_numeric(raw['Value'], errors='coerce')
    raw = raw.sort_values('M_Date').reset_index(drop=True)
    print(raw[['sys', 'M_Date', 'Value', 'Pre_Dil', 'D_Alarm', 'file']].to_string(index=False))
    rawc = C.calc38(raw)

    df = pd.read_parquet(C.PROC, columns=['SystemName', 'S_ID', 'S_Date', 'Value'] + C.RAS + C.RAM,
                         filters=[('A_Name', '==', 'Cu')])
    df['d'] = pd.to_datetime(df['S_Date'])
    df['sys'] = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    is_case = df['S_ID'].astype(str).str.strip() == sid
    # processed check for the curves that are in the analysed dataset
    pc = C.calc38(df[is_case])
    match = [bool(any(np.allclose(rawc[i], q) for q in pc)) for i in range(len(raw))]
    print('in processed (curve identical):', match, '| processed case rows:', int(is_case.sum()))

    rest = df[~is_case]
    DAY = case_day()
    pre = rest[rest['d'] < DAY]
    out = []
    for norm, fn in (('noclip', E.maha_fit_noclip), ('clip', ORIG_FIT)):
        C.maha_fit = fn
        for i, r in raw.iterrows():
            an = r['sys']; other = '#2' if an == '#3' else '#3'
            test = rest[(rest['sys'] == an) & (rest['d'] >= DAY) & (rest['d'] <= DAY + pd.Timedelta(days=70))]
            variants = [(f'single_{an}(own)', pre[pre['sys'] == an]), (f'single_{other}(other)', pre[pre['sys'] == other]),
                        ('pool_#2+#3', pre[pre['sys'].isin(['#2', '#3'])])]
            for lab, tr in variants:
                tc = C.calc38(tr.sort_values('d').iloc[-C.N_TRAIN:])
                for (p0, p1) in [(5, 38), (1, 38)]:
                    res, m = C.run(df, rawc[i], tc, C.calc38(test), lab, p0, p1)
                    cols = C.window_cols(p0, p1)
                    t = tc[:, cols]; a = rawc[i][cols]
                    z = np.abs(a - t.mean(0)) / (t.std(0) + 1e-9)
                    onset = int(np.argmax(np.abs(np.diff(t.mean(0)))))
                    b = np.arange(len(cols)) <= onset
                    res.update(norm=norm, measurement=LABEL.get(float(r['v']), str(r['Value'])), analyzer=an, value=float(r['v']),
                               in_processed=match[i], dist_scale_multiple=round(float(np.sqrt(res['d_anomaly'] / res['threshold_P995'])), 2),
                               points_outside_train_range=int(((a < t.min(0)) | (a > t.max(0))).sum()), n_points=len(cols),
                               z_baseline_median=round(float(np.median(z[b])), 2), z_baseline_max=round(float(z[b].max()), 2),
                               z_reaction_median=round(float(np.median(z[~b])), 2), z_reaction_max=round(float(z[~b].max()), 2),
                               onset_point=p0 + onset)
                    out.append(res)
    pd.DataFrame(out).to_csv(os.path.join(OUT, 'case2_aligned.csv'), index=False, encoding='utf-8-sig')
    json.dump(out, open(os.path.join(OUT, 'case2_aligned.json'), 'w', encoding='utf-8'), indent=1, default=str)
    print('DONE ->', OUT)


if __name__ == '__main__':
    main()
