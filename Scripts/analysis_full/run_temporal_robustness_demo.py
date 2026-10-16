# Temporal-robustness DEMO (CRE only): rolling "most-recent 500" Mahalanobis detector over
# real calendar time (2021-2025), measuring month-to-month decision CONSISTENCY.
# Spec: ForPaper/05_TemporalRobustness/experiment_plan.md.
#
# For each monthly anchor t: train window = the 500 real CRE curves immediately BEFORE t.
#   normalize (global min-max from the window, as in 04_MethodComparison code),
#   fit Empirical maha / Robust maha / PCA, threshold = training-window P99.5 (self-calibration).
# Score a FIXED probe (CRE synthetic anomalies) with each month's model -> recall/specificity(t)
#   and the 0/1 flag vector.  Consistency = agreement / Cohen's kappa / Spearman between
#   consecutive months (last month vs this month) -> directly answers the reviewer concern.
# No TensorFlow: fast, CPU-only.

import os, sys, json
import numpy as np, pandas as pd
import pyarrow.dataset as ds, pyarrow.compute as pc
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
SYN = os.path.join(BASE, 'Data', 'BigData', 'synthetic_anomalies', 'endpoint_synthetic_anomalies.parquet')
OUT = os.path.join(BASE, 'Results', '20260706_TemporalRobustness_CRE_demo')
ITEM = 'CRE'
N = 500
BUDGET = 99.5
RAM = [f'RAM{i}' for i in range(11, 39)]
RAS = [f'RAS{i}' for i in range(11, 39)]
POS = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']


def norm_fit(a):
    return float(a.min()), float(a.max())


def apply_norm(a, lo, hi):
    return (a - lo) / (hi - lo + 1e-8)      # affine, no clipping


def mcd_like(D, keep=0.95, iters=5):
    loc = D.mean(0); cov = np.cov(D.T); n = int(len(D) * keep)
    for _ in range(iters):
        d = np.sum((D - loc) @ np.linalg.pinv(cov) * (D - loc), axis=1)
        core = D[np.argsort(d)[:n]]; loc, cov = core.mean(0), np.cov(core.T)
    return loc, cov


def maha(loc, cov, X):
    P = np.linalg.pinv(cov)
    return np.sum((X - loc) @ P * (X - loc), axis=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    print('loading CRE real curves...', flush=True)
    d = ds.dataset(PROC, partitioning='hive')
    tb = d.to_table(columns=['S_Date'] + RAM + RAS, filter=pc.field('A_Name') == ITEM).to_pandas()
    tb = tb.sort_values('S_Date').reset_index(drop=True)
    dates = pd.to_datetime(tb['S_Date'])
    calc_all = (tb[RAM].values.astype('float64') - tb[RAS].values.astype('float64'))
    print(f'  {len(tb):,} rows, {dates.min()} -> {dates.max()}', flush=True)

    # fixed probe = synthetic anomalies for CRE
    s = pd.read_parquet(SYN, filters=[('item', '==', ITEM)])
    probe = s[RAM].values.astype('float64') - s[RAS].values.astype('float64')
    at = s['anomaly_type'].values; sev = s['severity'].values
    clean = at == 'clean'; benign = at == 'benign_offset'
    main_pos = np.isin(at, POS) & (sev >= 2)
    k1_pos = np.isin(at, POS) & (sev == 1)
    # borderline decision region where consistency actually matters (near the threshold)
    borderline = benign | k1_pos

    months = pd.period_range(dates.min().to_period('M') + 1, dates.max().to_period('M'), freq='M')
    methods = ['E', 'R', 'P']  # Empirical maha, Robust maha, PCA
    rows = []
    flags = {m: {} for m in methods}
    scores = {m: {} for m in methods}

    for mo in months:
        anchor = mo.start_time
        idx = np.where(dates.values < np.datetime64(anchor))[0]
        if len(idx) < N:
            continue
        win = calc_all[idx[-N:]]
        lo, hi = norm_fit(win)
        Wn = apply_norm(win, lo, hi); Pn = apply_norm(probe, lo, hi)
        mkey = str(mo)
        # --- Empirical maha ---
        e = EmpiricalCovariance().fit(Wn)
        sE_tr = maha(e.location_, e.covariance_, Wn); sE = maha(e.location_, e.covariance_, Pn)
        thrE = np.percentile(sE_tr, BUDGET)
        # --- Robust maha ---
        lR, cR = mcd_like(Wn)
        sR_tr = maha(lR, cR, Wn); sR = maha(lR, cR, Pn); thrR = np.percentile(sR_tr, BUDGET)
        # --- PCA reconstruction ---
        pca = PCA(n_components=0.95).fit(Wn)
        rec = lambda X: np.mean((X - pca.inverse_transform(pca.transform(X))) ** 2, axis=1)
        sP_tr = rec(Wn); sP = rec(Pn); thrP = np.percentile(sP_tr, BUDGET)
        smap = {'E': (sE, thrE), 'R': (sR, thrR), 'P': (sP, thrP)}
        row = {'month': mkey, 'n_hist': int(len(idx)),
               'win_start': str(dates.values[idx[-N:]][0])[:10], 'win_end': str(dates.values[idx[-1]])[:10]}
        for m in methods:
            sc, thr = smap[m]
            f = sc >= thr
            flags[m][mkey] = f; scores[m][mkey] = sc
            row[f'{m}_recall_main'] = float(f[main_pos].mean())
            row[f'{m}_recall_k1'] = float(f[k1_pos].mean())
            row[f'{m}_spec_clean'] = float(1 - f[clean].mean())
            row[f'{m}_fp_benign'] = float(f[benign].mean())
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'monthly_metrics.csv'), index=False)
    mks = list(df['month'])

    # ---- consecutive-month consistency (last month vs this month) ----
    cons = {m: {'agree_all': [], 'kappa_all': [], 'agree_bord': [], 'kappa_bord': [], 'spearman': []} for m in methods}
    for m in methods:
        for a, b in zip(mks[:-1], mks[1:]):
            fa, fb = flags[m][a], flags[m][b]
            cons[m]['agree_all'].append(float((fa == fb).mean()))
            cons[m]['kappa_all'].append(float(cohen_kappa_score(fa, fb)) if fa.sum() not in (0, len(fa)) or fb.sum() not in (0, len(fb)) else 1.0)
            cons[m]['agree_bord'].append(float((fa[borderline] == fb[borderline]).mean()))
            kb = cohen_kappa_score(fa[borderline], fb[borderline])
            cons[m]['kappa_bord'].append(float(kb) if not np.isnan(kb) else 1.0)
            rho = spearmanr(scores[m][a], scores[m][b]).correlation
            cons[m]['spearman'].append(float(rho))

    summary = {'item': ITEM, 'N': N, 'budget': BUDGET, 'n_months': len(mks),
               'month_range': [mks[0], mks[-1]], 'metrics_time_stability': {}, 'consecutive_consistency': {}}
    label = {'E': 'Empirical_maha', 'R': 'Robust_maha', 'P': 'PCA'}
    for m in methods:
        summary['metrics_time_stability'][label[m]] = {
            k: {'mean': float(df[f'{m}_{k}'].mean()), 'sd': float(df[f'{m}_{k}'].std()),
                'min': float(df[f'{m}_{k}'].min()), 'max': float(df[f'{m}_{k}'].max())}
            for k in ['recall_main', 'recall_k1', 'spec_clean', 'fp_benign']}
        summary['consecutive_consistency'][label[m]] = {
            k: {'mean': float(np.mean(v)), 'min': float(np.min(v))} for k, v in cons[m].items()}
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- figure ----
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    x = np.arange(len(mks))
    col = {'E': '#d62728', 'R': '#e377c2', 'P': '#2ca02c'}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for m in methods:
        ax[0, 0].plot(x, df[f'{m}_recall_main'], color=col[m], label=label[m], lw=1.6)
        ax[0, 1].plot(x, df[f'{m}_spec_clean'], color=col[m], label=label[m], lw=1.6)
        ax[1, 0].plot(x, df[f'{m}_recall_k1'], color=col[m], label=label[m], lw=1.6)
        ax[1, 1].plot(x[1:], cons[m]['kappa_bord'], color=col[m], label=label[m], lw=1.6)
    tk = np.linspace(0, len(mks) - 1, 8).astype(int)
    for a in ax.flat:
        a.set_xticks(tk); a.set_xticklabels([mks[i] for i in tk], rotation=40, ha='right', fontsize=8)
        a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[0, 0].set_title('recall_main (k>=2) over time'); ax[0, 0].set_ylim(0, 1.03)
    ax[0, 1].set_title('specificity on clean over time'); ax[0, 1].set_ylim(0, 1.03)
    ax[1, 0].set_title('recall of SUBTLE (k=1) over time'); ax[1, 0].set_ylim(0, 1.03)
    ax[1, 1].set_title("month-to-month Cohen's kappa (borderline: benign + k=1)"); ax[1, 1].set_ylim(0, 1.03)
    fig.suptitle(f'{ITEM}: rolling trailing-{N} Mahalanobis over 2021-2025 (n={len(mks)} monthly models)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, 'temporal_robustness_CRE.png'), dpi=200); plt.close(fig)

    # ---- console summary ----
    print('\n===== CRE temporal-robustness demo =====')
    print(f'monthly models: {len(mks)}  ({mks[0]} .. {mks[-1]})')
    for m in methods:
        s = summary['metrics_time_stability'][label[m]]; c = summary['consecutive_consistency'][label[m]]
        print(f'\n[{label[m]}]')
        print(f'  recall_main  mean={s["recall_main"]["mean"]:.3f} sd={s["recall_main"]["sd"]:.3f} min={s["recall_main"]["min"]:.3f}')
        print(f'  recall_k1    mean={s["recall_k1"]["mean"]:.3f} sd={s["recall_k1"]["sd"]:.3f} min={s["recall_k1"]["min"]:.3f}')
        print(f'  spec_clean   mean={s["spec_clean"]["mean"]:.3f} sd={s["spec_clean"]["sd"]:.3f} min={s["spec_clean"]["min"]:.3f}')
        print(f'  consec agree(all)={c["agree_all"]["mean"]:.4f}(min {c["agree_all"]["min"]:.4f})  '
              f'kappa(all)={c["kappa_all"]["mean"]:.3f}(min {c["kappa_all"]["min"]:.3f})')
        print(f'  consec agree(borderline)={c["agree_bord"]["mean"]:.4f}(min {c["agree_bord"]["min"]:.4f})  '
              f'kappa(borderline)={c["kappa_bord"]["mean"]:.3f}(min {c["kappa_bord"]["min"]:.3f})  '
              f'spearman={c["spearman"]["mean"]:.3f}(min {c["spearman"]["min"]:.3f})')
    print('\nOUT ->', OUT)


if __name__ == '__main__':
    main()
