# Temporal-robustness for ALL items with synthetic ground truth: rolling "most-recent N"
# Mahalanobis detector over real calendar time (2021-2025), measuring per-item detection-power
# stability AND month-to-month decision CONSISTENCY.  Purpose: scan every item and surface ANY
# item where Mahalanobis degrades, for cause analysis.
# Spec: ForPaper/05_TemporalRobustness/experiment_plan.md.  Demo (CRE): run_temporal_robustness_demo.py.
#
# Scope: only items that have BOTH real data (Data/BigData/processed) AND synthetic anomalies
#   (Data/BigData/synthetic_anomalies).  The synthetic set defines the ground-truth probe, so items
#   without synthetic anomalies (AST/ALT/GGT/ChE/AMY/LIP/CK/Mg/CH50/RF/UN/... ) are OUT OF SCOPE here.
# Per item, monthly anchor t -> train window = the N real curves immediately BEFORE t; normalize
#   (global min-max from the window, as in 04_MethodComparison code); fit Empirical maha / Robust
#   maha / PCA; threshold = training-window P{BUDGET} (self-calibration).  A FIXED probe (the item's
#   synthetic anomalies) is scored by every month's model -> recall/specificity(t) and 0/1 flags.
#   Consistency = agreement / Cohen's kappa / Spearman between consecutive months.
# No TensorFlow: CPU-only, light (deep methods intentionally excluded to keep the all-item sweep cheap;
#   the goal is whether MAHALANOBIS is temporally robust on every item, with PCA as a classical foil).

import os, sys, gc, json, time, argparse
import numpy as np, pandas as pd
import pyarrow.dataset as ds, pyarrow.compute as pc
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
SYN = os.path.join(BASE, 'Data', 'BigData', 'synthetic_anomalies', 'endpoint_synthetic_anomalies.parquet')
OUT = os.path.join(BASE, 'Results', '20260706_TemporalRobustness_AllItems')

# (synth item -> processed A_Name).  35 items verified present in both sources (2026-07-06).
# NOTE: synthetic 'RPR-2' has NO matching processed partition (processed has RPR / RPR-S only) -> excluded
#   pending confirmation of the intended mapping (see experiment_plan.md).  Add here once confirmed.
ITEMS = {
    'ALB': 'ALB', 'B2M': 'B2M', 'C3': 'C3', 'C4': 'C4', 'CA': 'Ca', 'CRE': 'CRE', 'CRP': 'CRP',
    'CU': 'Cu', 'DB': 'DB', 'FE': 'Fe', 'FER': 'FER', 'GA': 'GA', 'GAALB': 'GAALB', 'GLU': 'GLU',
    'HDL-C': 'HDL-C', 'IGA': 'IgA', 'IGG': 'IgG', 'IGM': 'IgM', 'IP': 'IP', 'IgG4': 'IgG4',
    'KL-6': 'KL-6', 'LDL-C': 'LDL-C', 'LRG': 'LRG', 'Li': 'Li', 'MMP-3': 'MMP-3', 'TB': 'TB',
    'TC': 'TC', 'TG': 'TG', 'TP': 'TP', 'UA': 'UA', 'UIBC': 'UIBC', 'VALP': 'VALP', 'VCM': 'VCM',
    'ZN': 'Zn', 'mTP': 'mTP',
}

N = 500                         # rolling training-window size (most-recent N)
BUDGET = 99.5                   # operating point: training-window percentile (self-calibration)
MIN_ANCHORS = 3                 # need at least this many monthly models to report an item
RAM = [f'RAM{i}' for i in range(11, 39)]
RAS = [f'RAS{i}' for i in range(11, 39)]
POS = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']
METHODS = ['E', 'R', 'P']                                   # Empirical maha, Robust maha, PCA
LABEL = {'E': 'Empirical_maha', 'R': 'Robust_maha', 'P': 'PCA'}
COL = {'E': '#d62728', 'R': '#e377c2', 'P': '#2ca02c'}
# "breakage" thresholds used to flag items for cause analysis (Mahalanobis is expected to pass all)
OK_RECALL_MAIN_MIN = 0.90
OK_KAPPA_BORD_MIN = 0.80
OK_SPEC_MIN = 0.90


# ---- core (identical formulation to 04_MethodComparison / the CRE demo) ----
def norm_fit(a):
    return float(a.min()), float(a.max())


def apply_norm(a, lo, hi):
    return np.clip((a - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def mcd_like(D, keep=0.95, iters=5):
    loc = D.mean(0); cov = np.cov(D.T); n = int(len(D) * keep)
    for _ in range(iters):
        d = np.sum((D - loc) @ np.linalg.pinv(cov) * (D - loc), axis=1)
        core = D[np.argsort(d)[:n]]; loc, cov = core.mean(0), np.cov(core.T)
    return loc, cov


def maha(loc, cov, X):
    P = np.linalg.pinv(cov)
    return np.sum((X - loc) @ P * (X - loc), axis=1)


def _kappa(a, b):
    # cohen_kappa is undefined (nan) when both label vectors are constant & identical -> treat as 1.0
    if a.sum() in (0, len(a)) and b.sum() in (0, len(b)) and a.sum() == b.sum():
        return 1.0
    k = cohen_kappa_score(a, b)
    return float(k) if not np.isnan(k) else 1.0


def load_real(proc_name):
    d = ds.dataset(PROC, partitioning='hive')
    tb = d.to_table(columns=['S_Date'] + RAM + RAS, filter=pc.field('A_Name') == proc_name).to_pandas()
    tb = tb.sort_values('S_Date').reset_index(drop=True)
    dates = pd.to_datetime(tb['S_Date'])
    calc = tb[RAM].values.astype('float64') - tb[RAS].values.astype('float64')
    del tb; gc.collect()
    return dates, calc


def load_probe(synth_item):
    s = pd.read_parquet(SYN, filters=[('item', '==', synth_item)])
    probe = s[RAM].values.astype('float64') - s[RAS].values.astype('float64')
    return probe, s['anomaly_type'].values, s['severity'].values


# ---- one item ----
def run_item(synth_item, proc_name, outdir_item):
    dates, calc = load_real(proc_name)
    probe, at, sev = load_probe(synth_item)
    clean = at == 'clean'; benign = at == 'benign_offset'
    main_pos = np.isin(at, POS) & (sev >= 2)
    k1_pos = np.isin(at, POS) & (sev == 1)
    borderline = benign | k1_pos                      # decisions near the threshold that can flip

    d_ns = dates.values.astype('datetime64[ns]')
    months = pd.period_range(dates.min().to_period('M') + 1, dates.max().to_period('M'), freq='M')
    rows, flags, scores, cuts = [], {m: {} for m in METHODS}, {m: {} for m in METHODS}, []
    for mo in months:
        anchor = np.datetime64(mo.start_time, 'ns')
        cut = int(np.searchsorted(d_ns, anchor, side='left'))     # #samples strictly before month start
        if cut < N:
            continue
        cuts.append(cut)
        win = calc[cut - N:cut]
        lo, hi = norm_fit(win)
        Wn = apply_norm(win, lo, hi); Pn = apply_norm(probe, lo, hi)
        e = EmpiricalCovariance().fit(Wn)
        sE = maha(e.location_, e.covariance_, Pn); thrE = np.percentile(maha(e.location_, e.covariance_, Wn), BUDGET)
        lR, cR = mcd_like(Wn)
        sR = maha(lR, cR, Pn); thrR = np.percentile(maha(lR, cR, Wn), BUDGET)
        pca = PCA(n_components=0.95).fit(Wn)
        rec = lambda X: np.mean((X - pca.inverse_transform(pca.transform(X))) ** 2, axis=1)
        sP = rec(Pn); thrP = np.percentile(rec(Wn), BUDGET)
        smap = {'E': (sE, thrE), 'R': (sR, thrR), 'P': (sP, thrP)}
        mkey = str(mo)
        row = {'month': mkey, 'n_hist': cut, 'win_start': str(d_ns[cut - N])[:10], 'win_end': str(d_ns[cut - 1])[:10]}
        for m in METHODS:
            sc, thr = smap[m]; f = sc >= thr
            flags[m][mkey] = f; scores[m][mkey] = sc
            row[f'{m}_recall_main'] = float(f[main_pos].mean()) if main_pos.any() else np.nan
            row[f'{m}_recall_k1'] = float(f[k1_pos].mean()) if k1_pos.any() else np.nan
            row[f'{m}_spec_clean'] = float(1 - f[clean].mean()) if clean.any() else np.nan
            row[f'{m}_fp_benign'] = float(f[benign].mean()) if benign.any() else np.nan
        rows.append(row)

    if len(rows) < MIN_ANCHORS:
        return {'item': synth_item, 'proc': proc_name, 'status': 'insufficient',
                'n_months': len(rows), 'n_real': int(len(calc))}, None

    df = pd.DataFrame(rows)
    os.makedirs(outdir_item, exist_ok=True)
    df.to_csv(os.path.join(outdir_item, 'monthly_metrics.csv'), index=False)
    mks = list(df['month'])
    # mean fraction of the training window shared by consecutive months (rare items -> high overlap)
    overlaps = [max(0.0, 1.0 - (cuts[i + 1] - cuts[i]) / N) for i in range(len(cuts) - 1)]
    win_overlap = float(np.mean(overlaps)) if overlaps else np.nan

    cons = {m: {'agree_all': [], 'kappa_all': [], 'agree_bord': [], 'kappa_bord': [], 'spearman': []} for m in METHODS}
    for m in METHODS:
        for a, b in zip(mks[:-1], mks[1:]):
            fa, fb = flags[m][a], flags[m][b]
            cons[m]['agree_all'].append(float((fa == fb).mean()))
            cons[m]['kappa_all'].append(_kappa(fa, fb))
            cons[m]['agree_bord'].append(float((fa[borderline] == fb[borderline]).mean()) if borderline.any() else np.nan)
            cons[m]['kappa_bord'].append(_kappa(fa[borderline], fb[borderline]) if borderline.any() else np.nan)
            rho = spearmanr(scores[m][a], scores[m][b]).correlation
            cons[m]['spearman'].append(float(rho) if rho is not None and not np.isnan(rho) else np.nan)

    rec_item = {'item': synth_item, 'proc': proc_name, 'status': 'ok', 'n_months': len(mks),
                'n_real': int(len(calc)), 'month_range': [mks[0], mks[-1]], 'win_overlap_mean': win_overlap}
    for m in METHODS:
        rec_item[LABEL[m]] = {
            'recall_main': {'mean': float(df[f'{m}_recall_main'].mean()), 'sd': float(df[f'{m}_recall_main'].std()),
                            'min': float(df[f'{m}_recall_main'].min())},
            'recall_k1': {'mean': float(df[f'{m}_recall_k1'].mean()), 'sd': float(df[f'{m}_recall_k1'].std()),
                          'min': float(df[f'{m}_recall_k1'].min())},
            'spec_clean': {'mean': float(df[f'{m}_spec_clean'].mean()), 'min': float(df[f'{m}_spec_clean'].min())},
            'consec_agree_bord': {'mean': float(np.nanmean(cons[m]['agree_bord'])), 'min': float(np.nanmin(cons[m]['agree_bord']))},
            'consec_kappa_bord': {'mean': float(np.nanmean(cons[m]['kappa_bord'])), 'min': float(np.nanmin(cons[m]['kappa_bord']))},
            'consec_spearman': {'mean': float(np.nanmean(cons[m]['spearman'])), 'min': float(np.nanmin(cons[m]['spearman']))}}

    _plot_item(synth_item, df, cons, mks, win_overlap, outdir_item)
    del calc, probe; gc.collect()
    return rec_item, df


# ---- per-item 4-panel figure (same layout as the CRE demo) ----
def _plot_item(item, df, cons, mks, win_overlap, outdir_item):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    x = np.arange(len(mks))
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for m in METHODS:
        ax[0, 0].plot(x, df[f'{m}_recall_main'], color=COL[m], label=LABEL[m], lw=1.6)
        ax[0, 1].plot(x, df[f'{m}_spec_clean'], color=COL[m], label=LABEL[m], lw=1.6)
        ax[1, 0].plot(x, df[f'{m}_recall_k1'], color=COL[m], label=LABEL[m], lw=1.6)
        ax[1, 1].plot(x[1:], cons[m]['kappa_bord'], color=COL[m], label=LABEL[m], lw=1.6)
    tk = np.linspace(0, len(mks) - 1, min(8, len(mks))).astype(int)
    for a in ax.flat:
        a.set_xticks(tk); a.set_xticklabels([mks[i] for i in tk], rotation=40, ha='right', fontsize=8)
        a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[0, 0].set_title('recall_main (k>=2) over time'); ax[0, 0].set_ylim(0, 1.03)
    ax[0, 1].set_title('specificity on clean over time'); ax[0, 1].set_ylim(0, 1.03)
    ax[1, 0].set_title('recall of SUBTLE (k=1) over time'); ax[1, 0].set_ylim(0, 1.03)
    ax[1, 1].set_title("month-to-month Cohen's kappa (borderline: benign + k=1)"); ax[1, 1].set_ylim(0, 1.03)
    fig.suptitle(f'{item}: rolling trailing-{N} Mahalanobis, {len(mks)} monthly models '
                 f'(mean window overlap {win_overlap:.0%})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir_item, f'temporal_{item}.png'), dpi=150); plt.close(fig)


# ---- cross-item overview figure: sorted worst-first, so any breaking item is obvious ----
def plot_overview(recs):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ok = [r for r in recs if r['status'] == 'ok']
    ok.sort(key=lambda r: r['Robust_maha']['recall_main']['min'])   # worst recall first
    items = [r['item'] for r in ok]; y = np.arange(len(items))
    fig, ax = plt.subplots(1, 3, figsize=(17, max(5, 0.34 * len(items))))
    def panel(a, getter, title, okline):
        for m, off in [('E', -0.2), ('R', 0.2)]:
            a.barh(y + off, [getter(r, m) for r in ok], height=0.38, color=COL[m], label=LABEL[m])
        a.axvline(okline, color='k', ls='--', lw=1, label=f'ok>= {okline}')
        a.set_yticks(y); a.set_yticklabels(items, fontsize=8); a.invert_yaxis()
        a.set_xlim(0, 1.02); a.set_title(title); a.grid(alpha=0.3, axis='x'); a.legend(fontsize=8, loc='lower right')
    panel(ax[0], lambda r, m: r[LABEL[m]]['recall_main']['min'], 'min recall_main(k>=2) over months', OK_RECALL_MAIN_MIN)
    panel(ax[1], lambda r, m: r[LABEL[m]]['consec_kappa_bord']['min'], "min month-to-month kappa (borderline)", OK_KAPPA_BORD_MIN)
    panel(ax[2], lambda r, m: r[LABEL[m]]['spec_clean']['min'], 'min specificity(clean) over months', OK_SPEC_MIN)
    fig.suptitle(f'Temporal robustness across {len(items)} items — worst-month value per item '
                 f'(Empirical vs Robust Mahalanobis). Bars left of the dashed line = flagged for review.', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, 'overview_worstmonth.png'), dpi=200); plt.close(fig)


def flags_for_review(recs):
    out = []
    for r in recs:
        if r['status'] != 'ok':
            out.append((r['item'], f"status={r['status']} (n_months={r.get('n_months')}, n_real={r.get('n_real')})")); continue
        for m in ('Empirical_maha', 'Robust_maha'):
            probs = []
            if r[m]['recall_main']['min'] < OK_RECALL_MAIN_MIN: probs.append(f"recall_main.min={r[m]['recall_main']['min']:.2f}")
            if r[m]['consec_kappa_bord']['min'] < OK_KAPPA_BORD_MIN: probs.append(f"kappa_bord.min={r[m]['consec_kappa_bord']['min']:.2f}")
            if r[m]['spec_clean']['min'] < OK_SPEC_MIN: probs.append(f"spec.min={r[m]['spec_clean']['min']:.2f}")
            if probs:
                out.append((r['item'], f"{m}: " + ', '.join(probs)))
    return out


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=list(ITEMS.keys()))
    ap.add_argument('--outdir', default=OUT)
    args = ap.parse_args()
    OUT = args.outdir
    os.makedirs(OUT, exist_ok=True)
    todo = [(s, ITEMS[s]) for s in args.items if s in ITEMS]
    t0 = time.time(); recs = []
    for i, (s, p) in enumerate(todo, 1):
        print(f'[{i}/{len(todo)}] {s} (proc={p}) ...', flush=True)
        try:
            rec, _ = run_item(s, p, os.path.join(OUT, s))
        except Exception as ex:
            rec = {'item': s, 'proc': p, 'status': f'error: {type(ex).__name__}: {ex}'}
            print('   ERROR', ex, flush=True)
        recs.append(rec)
        if rec.get('status') == 'ok':
            print(f'   n_months={rec["n_months"]} overlap={rec["win_overlap_mean"]:.0%}  '
                  f'maha_R recall_main.min={rec["Robust_maha"]["recall_main"]["min"]:.3f} '
                  f'kappa_bord.min={rec["Robust_maha"]["consec_kappa_bord"]["min"]:.3f} '
                  f'spec.min={rec["Robust_maha"]["spec_clean"]["min"]:.3f}', flush=True)
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'N': N, 'budget': BUDGET, 'items': recs, 'elapsed_s': time.time() - t0}, f, indent=2, ensure_ascii=False)
    # flat CSV
    flat = []
    for r in recs:
        if r.get('status') != 'ok':
            flat.append({'item': r['item'], 'proc': r.get('proc'), 'status': r.get('status')}); continue
        d = {'item': r['item'], 'proc': r['proc'], 'status': 'ok', 'n_months': r['n_months'],
             'n_real': r['n_real'], 'win_overlap_mean': r['win_overlap_mean']}
        for m in ('Empirical_maha', 'Robust_maha', 'PCA'):
            d[f'{m}_recall_main_min'] = r[m]['recall_main']['min']
            d[f'{m}_recall_k1_min'] = r[m]['recall_k1']['min']
            d[f'{m}_spec_clean_min'] = r[m]['spec_clean']['min']
            d[f'{m}_kappa_bord_min'] = r[m]['consec_kappa_bord']['min']
            d[f'{m}_kappa_bord_mean'] = r[m]['consec_kappa_bord']['mean']
        flat.append(d)
    pd.DataFrame(flat).to_csv(os.path.join(OUT, 'all_items_summary.csv'), index=False)
    if any(r.get('status') == 'ok' for r in recs):
        plot_overview(recs)
    review = flags_for_review(recs)
    print('\n===== FLAGGED FOR REVIEW (Mahalanobis expected to pass all) =====')
    print('\n'.join(f'  {it}: {msg}' for it, msg in review) if review else '  none — Mahalanobis robust on all items')
    n_ok = sum(1 for r in recs if r.get('status') == 'ok')
    print(f'\nDONE  {n_ok}/{len(recs)} ok  ->  {OUT}', flush=True)


if __name__ == '__main__':
    main()
