# Real-anomaly validation: does the method flag the one KNOWN, adjudicated real anomaly?
#   Cu abnormal reaction (specimen containing a monoclonal protein), analyzer #3.
# Deployment-realistic: train Mahalanobis(E) on the recent 500 Cu samples BEFORE the anomaly date
# (self-calibrated P99.5), on the confirmed window 5-38 (bubble points 1-4 excluded), and test whether the
# held-out anomaly is flagged.  Two training variants: single-analyzer #3, and pooled #2+#3.  Window 1-38
# is also computed for context (does 5-38 alone catch it, without the excluded early spike?).
# Maha only -> py3.14, no TF.
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
OUT = os.path.join(BASE, 'Results', '20260718_RealAnomaly_Cu')
FIGDIR = os.path.join(BASE, 'ForPaper', 'Suppl09_RealAnomalyValidation', 'Figures')
RAS = [f'RAS{i}' for i in range(1, 39)]; RAM = [f'RAM{i}' for i in range(1, 39)]
# The specimen identifier of the internal dataset is not published. 900000001 is a placeholder;
# set CU_CASE_ID to the identifier in your own copy of the processed data to run the analysis.
ANOM_SID = os.environ.get('CU_CASE_ID', '900000001')      # S_ID is stored as a string in processed
N_TRAIN = 500
BUDGET = 99.5


def maha_fit(train):
    lo, hi = train.min(0), train.max(0)
    nz = lambda X: (X - lo) / (hi - lo + 1e-8)      # affine, no clipping
    ntr = nz(train); mu = ntr.mean(0); P = np.linalg.pinv(np.cov(ntr.T))
    dtr = np.einsum('ij,jk,ik->i', ntr - mu, P, ntr - mu)
    thr = np.percentile(dtr, BUDGET)
    return dict(nz=nz, mu=mu, P=P, thr=thr, dtr=dtr, tr_mean=train.mean(0), tr_sd=train.std(0))


def dist(m, X):
    nx = m['nz'](X)
    return np.einsum('ij,jk,ik->i', nx - m['mu'], m['P'], nx - m['mu'])


def window_cols(p0, p1):
    idx = [i - 1 for i in range(p0, p1 + 1)]      # 0-based indices into 1..38 arrays
    return idx


def run(df, anom_calc38, train_calc38, test_calc38, label, p0, p1):
    """Fit on training (window p0-p1), test the anomaly + a contemporaneous test set."""
    cols = window_cols(p0, p1)
    m = maha_fit(train_calc38[:, cols])
    d_anom = float(dist(m, anom_calc38[None, cols])[0])
    thr = m['thr']
    flagged = bool(d_anom >= thr)
    pct = float((m['dtr'] < d_anom).mean() * 100)            # percentile among training distances
    d_test = dist(m, test_calc38[:, cols]) if len(test_calc38) else np.array([])
    rank = int((d_test > d_anom).sum() + 1) if len(d_test) else -1   # 1 = most extreme
    # localization (window points), onset from training mean of the window
    rmu, rsd = m['tr_mean'], m['tr_sd'] + 1e-9
    onset_local = int(np.argmax(np.abs(np.diff(rmu))))
    dev = np.abs(anom_calc38[cols] - rmu) / rsd
    bmask = np.arange(len(cols)) <= onset_local
    res = dict(variant=label, window=f'{p0}-{p1}', n_train=int(train_calc38.shape[0]),
               d_anomaly=round(d_anom, 1), threshold_P995=round(float(thr), 1),
               ratio_d_over_thr=round(d_anom / thr, 2), flagged=flagged,
               train_percentile=round(pct, 3), n_test=int(len(d_test)),
               rank_in_test=rank, baseline_dev_sigma=round(float(dev[bmask].mean()), 2),
               reaction_dev_sigma=round(float(dev[~bmask].mean()), 2))
    print(f'  [{label} | pts {p0}-{p1}] d={d_anom:.1f} thr={thr:.1f} ratio={d_anom/thr:.2f} '
          f'flagged={flagged} pctile={pct:.3f} rank={rank}/{len(d_test)}', flush=True)
    return res, m


def calc38(df):
    return df[RAM].values.astype('float64') - df[RAS].values.astype('float64')


def save_fig(fig, name, gray):
    os.makedirs(FIGDIR, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    for d in (FIGDIR, OUT):
        p = os.path.join(d, f'{name}_{"gray" if gray else "color"}.png')
        fig.savefig(p, dpi=600)
    if gray:
        try:
            from PIL import Image
            for d in (FIGDIR, OUT):
                p = os.path.join(d, f'{name}_gray.png'); Image.open(p).convert('L').save(p, dpi=(600, 600))
        except Exception:
            pass
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(PROC, columns=['SystemName', 'S_ID', 'S_Date', 'Value'] + RAS + RAM,
                         filters=[('A_Name', '==', 'Cu')])
    df['d'] = pd.to_datetime(df['S_Date'])
    df['sys'] = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    # anomaly row (unique)
    anom = df[df['S_ID'] == ANOM_SID]
    assert len(anom) == 1, f'expected 1 anomaly row, got {len(anom)}'
    anom_calc = calc38(anom)[0]
    print(f'anomaly located: sys={anom.iloc[0]["sys"]} '
          f'Value={anom.iloc[0]["Value"]}  Calc1={anom_calc[0]:.0f} Calc5={anom_calc[4]:.0f}', flush=True)

    anom_date = anom.iloc[0]['d'].normalize()      # sample date of the case: the training window ends before it
    pre = df[df['d'] < anom_date]
    train_single = pre[pre['sys'] == '#3'].sort_values('d').iloc[-N_TRAIN:]
    train_pool = pre[pre['sys'].isin(['#2', '#3'])].sort_values('d').iloc[-N_TRAIN:]
    # contemporaneous test set: Cu #3 in [anomaly date, +70d], excluding the anomaly itself
    test = df[(df['sys'] == '#3') & (df['d'] >= anom_date) & (df['d'] <= anom_date + pd.Timedelta(days=70))
              & (df['S_ID'] != ANOM_SID)]
    print(f'train_single(#3)={len(train_single)}  train_pool(#2+#3)={len(train_pool)}  test(#3 +70d)={len(test)}',
          flush=True)

    results = []
    models = {}
    for label, tr in [('single_#3', train_single), ('pool_#2+#3', train_pool)]:
        tc = calc38(tr); testc = calc38(test)
        for (p0, p1) in [(5, 38), (1, 38)]:
            res, m = run(df, anom_calc, tc, testc, label, p0, p1)
            results.append(res)
            if p0 == 5:
                models[label] = (m, tc)

    with open(os.path.join(OUT, 'detection_result.json'), 'w', encoding='utf-8') as f:
        json.dump({'anomaly': {'analyzer': '#3',
                               'value': float(anom.iloc[0]['Value'])}, 'results': results}, f, indent=2)

    # ---- figA: anomaly curve vs training normal band ----
    m_s, tc_s = models['single_#3']
    tr_mu, tr_sd = tc_s.mean(0), tc_s.std(0)
    x = np.arange(1, 39)
    for gray in (True, False):
        c_anom = '0.0' if gray else '#d62728'; c_band = '0.8' if gray else '#9ecae1'
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        for ax, (lo, hi, ttl) in [(a1, (1, 38, 'Full curve (pts 1–38): early spike at excluded pts 1–4')),
                                  (a2, (5, 38, 'Analysis window 5–38: anomaly deviates without the spike'))]:
            sl = slice(lo - 1, hi)
            ax.fill_between(x[sl], (tr_mu - 2 * tr_sd)[sl], (tr_mu + 2 * tr_sd)[sl], color=c_band, alpha=0.6, label='normal ±2SD (train #3)')
            ax.plot(x[sl], tr_mu[sl], '0.5', ls='--', lw=0.8, label='normal mean')
            ax.plot(x[sl], anom_calc[sl], color=c_anom, lw=1.8, marker='o', ms=3, label='anomaly')
            ax.set_xlabel('photometric point'); ax.set_ylabel('Calc (RAM−RAS)'); ax.set_title(ttl, fontsize=9.5)
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        if gray:
            a1.axvspan(1, 4, color='0.9', zorder=0)
        fig.suptitle('Cu real anomaly (analyzer #3) vs training normal band', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); save_fig(fig, 'figA_curve', gray)

    # ---- figB: training distance histogram + threshold + anomaly ----
    for gray in (True, False):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for ax, (label, (m, tc)) in zip(axes, models.items()):
            d_anom = float(dist(m, anom_calc[None, window_cols(5, 38)])[0])
            ax.hist(np.log10(m['dtr'] + 1), bins=40, color='0.7' if gray else '#9ecae1', label='training d² (log)')
            ax.axvline(np.log10(m['thr'] + 1), color='0.3' if gray else '#2ca02c', ls='--', lw=1.3, label='P99.5 threshold')
            ax.axvline(np.log10(d_anom + 1), color='0.0' if gray else '#d62728', ls='-', lw=2.0, label=f'anomaly d²={d_anom:.0f}')
            ax.set_xlabel('log10(Mahalanobis d² + 1)'); ax.set_ylabel('count')
            ax.set_title(f'{label} (pts 5–38)', fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle('Anomaly Mahalanobis distance vs training distribution / P99.5 threshold', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94]); save_fig(fig, 'figB_distance', gray)

    print('\nDONE ->', OUT, flush=True)


if __name__ == '__main__':
    main()
