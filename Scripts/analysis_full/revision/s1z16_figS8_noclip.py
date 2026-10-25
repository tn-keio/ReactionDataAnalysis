# Stage 1z16 (Fig. S8): the operating-point and optical-event panels recomputed without clipping.
# The paper is computed without np.clip throughout. Of the five panels of Fig. S8, (c) was already unclipped
# (s1v reproduced event_wavelength.csv exactly), (a) and (b) were drawn from the clipped R0 benchmark list
# (Results/20260717_MethodComparison_p5_38/Ca|CRE/N500/list.csv), and the scripts behind (d) and (e) were lost.
# This script recomputes (a), (b), (d) and (e) unclipped and redraws them with the drawing code of
# ForPaper/Suppl09b_OperatingPoint_Ca (copied, because those scripts run at import); only the values printed in the
# panels are taken from the data instead of being typed in (11.7%, 89% of flags on 3 days, 0.95%, 90.8%).
#   bench    : the N = 500 empirical-Mahalanobis arm of run_paper_method_comparison.py for Ca and CRE on the
#              pre-re-base data (same loader and synthetic battery as the R0 list, points 5-38), unclipped.
#              Checked first against the unclipped classical re-run (s1b --clip off): every N = 500 maha metric
#              at P99.5 of the 18 main-group items must agree.
#   (a)      : real vs synthetic distances. The R0 loader counted every synthetic curve twice (legend n = 1600 /
#              6400 for 800 / 3200 curves); here each curve is counted once. The frozen-holdout line is drawn at the
#              holdout P99.5 threshold itself (R0: at the matching percentile of the real curves).
#   (b)      : daily flag rate over Ca's 3,000-curve analysis window. The R0 panel plotted CRE's own analysis
#              window (2021-05-29..06-08) at the positions of the Ca dates; here CRE's N = 500 model and holdout
#              threshold are applied to the CRE curves of the same calendar days.
#   (d)      : TP, one fixed model per analyzer on its first 8,000 curves, self-calibrated P99.5, weekly flag rate,
#              cell/lamp replacements from the equipment log. Reproduces every value printed in the R0 panel
#              (weeks 179/246/185, spike weeks 0/3/0 at 69/84/95%, base median 0.63/0.63/0.59%).
#   (e)      : Ca, fixed model on the first 10,000 curves (2021-04-24..07-02), self-calibrated P99.5, weekly flag
#              rate over the first 120,000 curves (the R0 CSV's weekly counts are reproduced exactly).
# Without clipping the per-point min-max map is affine, so the Mahalanobis distance does not depend on it.
# Output: Results/20261025_Rev_FigS8_noclip/{bench,csv,Figures}/
# py3.14 (no TensorFlow).
import os
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
from sklearn.covariance import EmpiricalCovariance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
PROC = os.path.join(BASE, 'Data', 'BigData', 'processed')
SYN = os.path.join(BASE, 'Data', 'BigData', 'synthetic_anomalies', 'endpoint_synthetic_anomalies_p5_38.parquet')
EQUIP = os.path.join(BASE, 'Data', '20210301-20251231_equipment-comments.csv')
NOCLIP_JSONL = os.path.join(BASE, 'Results', '20260917_Rev_M1_Classical_noclip', 'main', 'Reports', 'comparison.jsonl')
OUT = os.path.join(BASE, 'Results', '20261025_Rev_FigS8_noclip')
FIGS = os.path.join(OUT, 'Figures')

RAM = [f'RAM{i}' for i in range(5, 39)]
RAS = [f'RAS{i}' for i in range(5, 39)]
N_TRAINPOOL, N_HOLDOUT, N_ANALYSIS = 10000, 2000, 3000
N = 500
BUDGET = 99.5
POS_TYPES = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']
DAILY = 435.7          # Ca tests/day, as in the R0 panel
MAIN18 = ['TP', 'TB', 'DB', 'CRE', 'UA', 'CRP', 'Fe', 'Cu', 'Zn', 'GLU', 'TG', 'LDL-C', 'HDL-C', 'TC',
          'IgG', 'KL-6', 'IgA', 'IgM']
THR_HOLDOUT = {}       # item -> holdout P99.5 threshold of the N = 500 model (set by bench)
SYN_NAME = {'Ca': 'CA', 'Fe': 'FE', 'Cu': 'CU', 'Zn': 'ZN', 'IgG': 'IGG', 'IgA': 'IGA', 'IgM': 'IGM'}


def maha(loc, cov, X):
    P = np.linalg.pinv(cov)
    return np.sum((X - loc) @ P * (X - loc), axis=1)


# ---------------- bench: the N = 500 maha arm, unclipped ----------------
def load_item(proc_name):
    """run_paper_method_comparison.load_item without the clip (and without the raw curves it keeps for plotting)."""
    df = pd.read_parquet(PROC, filters=[('A_Name', '==', proc_name)], columns=['S_Date'] + RAM + RAS)
    df = df.sort_values('S_Date').reset_index(drop=True).iloc[:N_TRAINPOOL + N_HOLDOUT + N_ANALYSIS]
    calc = df[RAM].values.astype('float64') - df[RAS].values.astype('float64')
    lo, hi = float(calc[:N_TRAINPOOL].min()), float(calc[:N_TRAINPOOL].max())
    norm = lambda a: (a - lo) / (hi - lo + 1e-8)
    Xn = norm(calc)
    s = pd.read_parquet(SYN, filters=[('item', '==', SYN_NAME.get(proc_name, proc_name))])
    syn = norm(s[RAM].values.astype('float64') - s[RAS].values.astype('float64'))
    return dict(train=Xn[:N], holdout=Xn[N_TRAINPOOL:N_TRAINPOOL + N_HOLDOUT],
                analysis=Xn[N_TRAINPOOL + N_HOLDOUT:], syn=syn,
                meta=s[['anomaly_type', 'severity']].reset_index(drop=True),
                an_date=pd.to_datetime(df['S_Date'].values[N_TRAINPOOL + N_HOLDOUT:]))


def score(ctx):
    e = EmpiricalCovariance().fit(ctx['train'])
    f = lambda X: maha(e.location_, e.covariance_, X)
    return {k: f(ctx[k]) for k in ('train', 'holdout', 'syn', 'analysis')}


def metrics(sc, meta):
    thr = np.percentile(sc['holdout'], BUDGET)
    fl = sc['syn'] >= thr
    at, sev = meta['anomaly_type'].values, meta['severity'].values
    m = {'specificity_clean': 1 - fl[at == 'clean'].mean(), 'fp_benign': fl[at == 'benign_offset'].mean(),
         'fp_holdout': (sc['holdout'] >= thr).mean(),
         'recall_main': fl[np.isin(at, POS_TYPES) & (sev >= 2)].mean()}
    for t in POS_TYPES:
        for k in sorted(np.unique(sev[at == t])):
            m[f'recall_{t}_k{int(k)}'] = fl[(at == t) & (sev == k)].mean()
    return m


def check_against_s1b():
    ref = {}
    for line in open(NOCLIP_JSONL, encoding='utf-8'):
        d = json.loads(line)
        if d['method'] == 'maha' and d['N'] == N:
            ref[d['item']] = d['budgets'][str(BUDGET)]
    worst = 0.0
    for it in MAIN18:
        ctx = load_item(it)
        m = metrics(score(ctx), ctx['meta'])
        diff = max(abs(m[k] - ref[it][k]['mean']) for k in m)
        worst = max(worst, diff)
        print(f'  check {it:6s} max |diff| vs s1b noclip = {diff:.2e}', flush=True)
    assert worst < 1e-9, f'standalone maha arm does not reproduce s1b --clip off (max diff {worst})'
    print('bench check passed: the unclipped N = 500 maha arm reproduces s1b --clip off on 18 items', flush=True)


def control_on_dates(item, lo, hi):
    """Daily flag rate of `item` on the calendar days lo..hi, from its own N = 500 model and holdout P99.5.
    The R0 panel drew CRE's own analysis window (2021-05-29..06-08) at the positions of the Ca dates."""
    df = pd.read_parquet(PROC, filters=[('A_Name', '==', item)], columns=['S_Date'] + RAM + RAS)
    df = df.sort_values('S_Date').reset_index(drop=True)
    X = df[RAM].values.astype('float64') - df[RAS].values.astype('float64')
    e = EmpiricalCovariance().fit(X[:N])
    thr = np.percentile(maha(e.location_, e.covariance_, X[N_TRAINPOOL:N_TRAINPOOL + N_HOLDOUT]), BUDGET)
    day = pd.to_datetime(df['S_Date']).dt.date
    sel = ((day >= lo) & (day <= hi)).values
    assert sel.any() and np.flatnonzero(sel).min() >= N_TRAINPOOL + N_HOLDOUT, 'window overlaps train/holdout'
    g = pd.DataFrame({'date': day[sel].values, 'maha': maha(e.location_, e.covariance_, X[sel]) >= thr}) \
        .groupby('date')['maha'].agg(n='size', fl='sum')
    g['rate'] = g['fl'] / g['n'] * 100
    return g


def bench(item):
    ctx = load_item(item)
    sc = score(ctx)
    thr = np.percentile(sc['holdout'], BUDGET)
    THR_HOLDOUT[item] = thr
    nsyn, nan = len(ctx['meta']), len(sc['analysis'])
    lst = pd.DataFrame({
        'source': ['synthetic'] * nsyn + ['real'] * nan,
        'ground_truth': list(ctx['meta']['anomaly_type']) + ['real_unknown'] * nan,
        'severity_k': list(ctx['meta']['severity']) + [np.nan] * nan,
        'date': [pd.NaT] * nsyn + list(ctx['an_date'].date),
        'maha': np.concatenate([sc['syn'] >= thr, sc['analysis'] >= thr]).astype(int),
        'score_maha': np.round(np.concatenate([sc['syn'], sc['analysis']]), 4)})
    os.makedirs(os.path.join(OUT, 'bench'), exist_ok=True)
    lst.to_csv(os.path.join(OUT, 'bench', f'{item}_N{N}_list.csv'), index=False)
    m = metrics(sc, ctx['meta'])
    print(f'  {item}: real flag rate at holdout P99.5 = {100 * (sc["analysis"] >= thr).mean():.2f}%  '
          f'recall_main = {m["recall_main"]:.3f}  specificity_clean = {m["specificity_clean"]:.3f}', flush=True)
    return lst


# ---------------- (e) weekly Ca, (d) TP per analyzer ----------------
def weekly_ca():
    df = pd.read_parquet(PROC, filters=[('A_Name', '==', 'Ca')], columns=['S_Date'] + RAM + RAS)
    df = df.sort_values('S_Date', kind='mergesort').reset_index(drop=True).iloc[:120000]
    X = df[RAM].values.astype('float64') - df[RAS].values.astype('float64')
    e = EmpiricalCovariance().fit(X[:N_TRAINPOOL])
    d = maha(e.location_, e.covariance_, X)
    thr = np.percentile(d[:N_TRAINPOOL], BUDGET)
    wk = pd.to_datetime(df['S_Date']).dt.to_period('W').dt.start_time
    g = pd.DataFrame({'week': wk, 'fl': d >= thr}).groupby('week')['fl'].agg(n='size', fl='sum').reset_index()
    g['rate'] = 100 * g['fl'] / g['n']
    os.makedirs(os.path.join(OUT, 'csv'), exist_ok=True)
    g.to_csv(os.path.join(OUT, 'csv', 'ca_weekly_flagrate_noclip.csv'), index=False)
    return g


def horizontal_tp():
    t = ds.dataset(PROC, partitioning='hive').to_table(
        columns=['S_Date', 'SystemName'] + RAM + RAS, filter=pc.field('A_Name') == 'TP').to_pandas()
    t['analyzer'] = t['SystemName'].astype(str).str.replace('LABOSPECT 008 ALPHA ', '', regex=False).str.strip()
    t['S_Date'] = pd.to_datetime(t['S_Date'])
    rows = []
    for az in ['#1', '#2', '#3']:
        s = t[t['analyzer'] == az].sort_values('S_Date', kind='mergesort').reset_index(drop=True)
        X = s[RAM].values.astype('float64') - s[RAS].values.astype('float64')
        e = EmpiricalCovariance().fit(X[:8000])
        d = maha(e.location_, e.covariance_, X)
        thr = np.percentile(d[:8000], BUDGET)
        g = pd.DataFrame({'week': s['S_Date'].dt.to_period('W').dt.start_time, 'fl': d >= thr}) \
            .groupby('week')['fl'].agg(n='size', fl='sum').reset_index()
        g['rate'] = 100 * g['fl'] / g['n']
        g.insert(0, 'analyzer', az)
        rows.append(g)
    w = pd.concat(rows, ignore_index=True)
    w.to_csv(os.path.join(OUT, 'csv', 'tp_weekly_by_analyzer_noclip.csv'), index=False)
    eq = pd.read_csv(EQUIP, encoding='utf-8-sig')
    eq.columns = ['date', 'system', 'case', 'cell', 'lamp', 'mech']
    eq['analyzer'] = eq['system'].astype(str).str.replace('LABOSPECT 008 ALPHA ', '', regex=False).str.strip()
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq[(eq['cell'] == 1) | (eq['lamp'] == 1)]
    return w, eq


# ---------------- drawing (code of ForPaper/Suppl09b_OperatingPoint_Ca, values from the data) ----------------
def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, f'{name}_color.png')
    fig.savefig(p, dpi=600); plt.close(fig)
    print('  wrote', os.path.relpath(p, OUT), flush=True)


def fig_a(lst):
    syn = lst[lst['source'] == 'synthetic']
    real = lst.loc[lst['source'] == 'real', 'score_maha'].values
    pos = syn[syn['ground_truth'].isin(POS_TYPES)]
    k1 = pos.loc[pos['severity_k'] == 1, 'score_maha'].values
    k2 = pos.loc[pos['severity_k'] >= 2, 'score_maha'].values
    frozen = 100 * lst.loc[lst['source'] == 'real', 'maha'].mean()      # real flag rate at the holdout P99.5
    thr = {f'{frozen:.1f}% (frozen holdout P99.5)': THR_HOLDOUT['Ca'],
           '0.5% (rate-calibrated)': np.percentile(real, 99.5), '0.1%': np.percentile(real, 99.9)}
    gap = (real.max(), min(k1.min(), k2.min()))
    print(f'  (a) real max = {gap[0]:.1f}, weakest anomaly = {gap[1]:.1f}, frozen-holdout rate = {frozen:.2f}%')
    c_real, c_k2, c_k1 = '#1f77b4', '#d62728', '#ff7f0e'
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.2))
    bins = np.logspace(0, np.log10(max(k2.max(), k1.max(), real.max())) + 0.3, 60)
    axA.hist(real, bins=bins, histtype='step', lw=2.4, ls='-', color=c_real,
             label=f'Real operational (unlabeled, n={len(real)})')
    axA.hist(k1, bins=bins, histtype='step', lw=1.8, ls='--', color=c_k1, label=f'Synthetic anomaly k=1 (n={len(k1)})')
    axA.hist(k2, bins=bins, histtype='step', lw=1.8, ls=':', color=c_k2, label=f'Synthetic anomaly k≥2 (n={len(k2)})')
    if gap[0] < gap[1]:
        axA.axvspan(gap[0], gap[1], color='#ffe08a', alpha=0.6, zorder=0)
        axA.text(np.sqrt(gap[0] * gap[1]), axA.get_ylim()[1] * 0.5, 'empty gap\n(no samples)\n%.0f–%.0f'
                 % gap, ha='center', va='center', fontsize=7.5, rotation=90)
    styles = [(':', 1.4), ('-.', 1.8), ((0, (1, 1)), 1.4)]
    for (lab, tv), (ls, lw) in zip(thr.items(), styles):
        axA.axvline(tv, color='#2ca02c', ls=ls, lw=lw)
        axA.text(tv, axA.get_ylim()[1] * (0.92 - 0.09 * list(thr).index(lab)), f' {lab.split(" ")[0]}',
                 fontsize=7, color='#2ca02c')
    axA.set_xscale('log'); axA.set_xlabel('Mahalanobis distance  d²'); axA.set_ylabel('count')
    axA.set_title('(a) Ca: distance distributions — real vs synthetic anomalies', fontsize=10)
    axA.legend(fontsize=7.5, loc='upper right'); axA.grid(alpha=0.3, which='both')

    tgrid = np.logspace(np.log10(30), np.log10(real.max()), 200)      # R0: 30 to ~real max (720)
    axB.plot(tgrid, np.array([(real >= t).mean() for t in tgrid]) * DAILY, '-', lw=2.6, color=c_real,
             label='maha flags / day (real)')
    for (lab, tv), mk in zip(thr.items(), ['o', 's', '^']):
        fd = (real >= tv).mean() * DAILY
        axB.plot([tv], [fd], mk, ms=9, color='#2ca02c')
        axB.annotate(f'{lab}\n{fd:.1f}/day', (tv, fd), textcoords='offset points', xytext=(8, 6), fontsize=7.5)
    axB.set_xscale('log'); axB.set_yscale('log')
    axB.set_xlabel('operating threshold  d²'); axB.set_ylabel('Mahalanobis flags per day')
    r1 = min((k1 >= tv).mean() for tv in thr.values()); r2 = min((k2 >= tv).mean() for tv in thr.values())
    axB.set_title('(b) Ca: workload knob — recall = 100% across this range' if min(r1, r2) == 1 else
                  '(b) Ca: workload knob', fontsize=10)
    axB.grid(alpha=0.3, which='both')
    note = (f'recall_main (k≥2) = {r2:.2f} and recall_subtle (k=1) = {r1:.2f}\nat every threshold shown'
            + (f' (all anomalies have d² ≥ {gap[1]:.0f} ≫ {gap[0]:.0f} = real max)' if gap[0] < gap[1] else ''))
    axB.text(0.03, 0.06, note, transform=axB.transAxes, fontsize=7.5,
             bbox=dict(boxstyle='round', fc='#eaf5ea', ec='0.6'))
    fig.suptitle('Controlling the Ca daily flag count with the Mahalanobis operating threshold   [color, supplemental]',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, 'figCa_operating_control')
    return dict(real_max=gap[0], weakest=gap[1], frozen_pct=frozen, recall_k1=r1, recall_k2=r2)


def daily(lst):
    r = lst[lst['source'] == 'real']
    g = r.groupby('date')['maha'].agg(n='size', fl='sum')
    g['rate'] = g['fl'] / g['n'] * 100
    return g


def fig_b(ca):
    CA = daily(ca)
    CRE = control_on_dates('CRE', CA.index.min(), CA.index.max()).reindex(CA.index)
    top3 = CA['fl'].sort_values(ascending=False).head(3)
    share = 100 * top3.sum() / CA['fl'].sum()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    xca = np.arange(len(CA))
    ax.bar(xca, CA['rate'].values, color='#d62728', width=0.7, label='Ca daily flag rate (train: 2021-04-24…07-02)')
    ax.plot(xca, CRE['rate'].values, '--', marker='s', ms=5, color='#1f77b4', lw=1.6,
            label='CRE (control item, same days)')
    ax.set_xticks(xca)
    ax.set_xticklabels([str(d)[5:] for d in CA.index], rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('Mahalanobis flag rate per day (%)'); ax.set_xlabel('date (analysis window, 2021)')
    ax.axhline(0.5, color='0.5', ls=':', lw=1.2)
    ax.text(0.5, 2.0, 'nominal 0.5% budget', fontsize=7, color='0.4')
    ev = list(CA.index).index(pd.to_datetime('2021-08-03').date())
    ax.annotate('optical baseline shift\n(pts 5–19; reaction phase unchanged)\n≈2021-08-02/03',
                xy=(ev, CA['rate'].values[ev]), xytext=(ev - 6, 55), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#d62728'))
    ax.set_title('Ca over-flagging is a discrete state change, not date-gap drift'
                 f'  ({share:.0f}% of flags on 3 days)   [color]', fontsize=10)
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); save(fig, 'figCa_state_change_by_date')
    print(f'  (b) top-3 days {[str(d) for d in top3.index]} hold {share:.1f}% of the flags; '
          f'Ca rates {dict(zip([str(d)[5:] for d in CA.index[-4:]], CA["rate"].round(1).tolist()[-4:]))}; '
          f'CRE on the same days: max {CRE["rate"].max():.2f}%, n per day {int(CRE["n"].min())}-{int(CRE["n"].max())}')
    return dict(share_top3=share, top3_days=[str(d) for d in top3.index], cre_max=CRE['rate'].max())


def fig_e(g):
    spike = g[g['rate'] >= 20]
    rest = g[g['rate'] < 20]
    peak = g['rate'].max()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    c = '#d62728'
    ax.plot(g['week'], g['rate'], '-', color=c, lw=1.6, marker='o', ms=3)
    ax.axhspan(0, 2, color='#eaf5ea', zorder=0)
    ax.axhline(1.0, color='0.5', ls=':', lw=1.1)
    ax.text(g['week'].iloc[1], 3.0, f'baseline median {rest["rate"].median():.2f}% ({len(rest)}/{len(g)} weeks)',
            fontsize=8, color='0.35')
    ax.annotate(f'transient optical-baseline event\nAug 2021 (~{len(spike)} weeks, peak {peak:.1f}%)\n'
                'module-2 optics; baseline pts 5–19; reaction normal',
                xy=(g.loc[g['rate'].idxmax(), 'week'], peak), xytext=(g['week'].iloc[12], 62), fontsize=8.5,
                arrowprops=dict(arrowstyle='->', color=c))
    mdate = pd.Timestamp('2021-08-22')
    ax.axvline(mdate, color='#2ca02c', ls='-.', lw=1.6)
    ax.annotate('analyzer #2\ncell + lamp replaced\n2021-08-22\n(equipment log)', xy=(mdate, 30),
                xytext=(g['week'].iloc[24], 40), fontsize=8, arrowprops=dict(arrowstyle='->', color='#2ca02c'))
    ax.set_ylabel('Ca maha flag rate per week (%)'); ax.set_xlabel('week (fixed model: trained Apr–Jul 2021)')
    ax.set_ylim(-3, 100)
    ax.set_title('Ca: a fixed Mahalanobis model is stable for 1.5 years — the excess is one transient event   [color]',
                 fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout(); save(fig, 'figCa_weekly_1p5yr')
    print(f'  (e) weeks >= 20%: {spike["rate"].round(1).tolist()}; other {len(rest)} weeks median '
          f'{rest["rate"].median():.2f}%')
    return dict(spikes=spike['rate'].round(1).tolist(), base_median=rest['rate'].median(), weeks=len(g))


def fig_d(w, eq):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    out = {}
    for ax, az in zip(axes, ['#1', '#2', '#3']):
        s = w[w['analyzer'] == az]
        r = s['rate']
        nsp, med = int((r >= 20).sum()), r[r < 20].median()
        out[az] = dict(weeks=len(s), spikes=r[r >= 20].round(0).astype(int).tolist(), base_median=round(med, 2))
        for i, dt in enumerate(eq.loc[eq['analyzer'] == az, 'date']):
            if s['week'].min() <= dt <= s['week'].max() + pd.Timedelta(days=6):
                ax.axvline(dt, color='#2ca02c', ls=':', lw=1.0, label='cell/lamp maintenance' if i == 0 else None)
        ax.plot(s['week'], r, '-', color='0.1', lw=1.2, marker='o', ms=2.5)
        ax.set_ylim(-3, 100); ax.set_ylabel('flag %'); ax.grid(alpha=0.3)
        ax.set_title(f'TP on analyzer {az}: weekly flag rate (fixed model on first 8000)  |  '
                     f'spike weeks(>=20%): {nsp}  base median {med:.2f}%', fontsize=9)
    h = [Line2D([], [], color='#2ca02c', ls=':', lw=1.4, label='cell/lamp maintenance')]
    axes[0].legend(handles=h, fontsize=8, loc='upper right')
    axes[-1].set_xlabel('week')
    fig.suptitle('Horizontal expansion: transient optical events per analyzer (TP, module-2)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); save(fig, 'figHorizontal_analyzers')
    print(f'  (d) {out}')
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    check_against_s1b()
    ca = bench('Ca')
    bench('CRE')
    summary = {'a': fig_a(ca), 'b': fig_b(ca)}
    summary['e'] = fig_e(weekly_ca())
    summary['d'] = fig_d(*horizontal_tp())
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=float)


if __name__ == '__main__':
    main()
