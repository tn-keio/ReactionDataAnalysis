# Stage 1r-9 (Reviewer #2 D1, option B): Fig. S7 redrawn for the second Cu case of the revised manuscript
# (the case of the original Fig. 5, now §S10.4), in the style of the final Fig. 5 (s1r8).
# Why: the original panels carried the specimen date in the title and the last digits of the sample ID in the
# legend; the redrawn figure carries neither (author's decision 2026-10-21).
# Protocol as real_anomaly_cu_validation.py / s1e (unclipped normalization; identical values for this case):
# training = the 500 most recent pre-event curves of analyzer #3 (single) or of #2+#3 (pooled); test = analyzer #3,
# event date .. +70 days, case excluded; window 5-38.
#   (a) the case curve against the normal +/-2 SD band of the single-analyzer training set; left: points 1-38
#       (points 1-4 shaded), right: points 5-38
#   (b) d^2 / threshold of the contemporaneous curves under each model (single, pooled); the case as vertical lines
# Output: ForPaper/Rev_FigS7_Case1/Figures/ (colour), Results/20261021_Rev_FigS7_Case1/figS7_values.csv
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, '..'))):
    if p not in sys.path:
        sys.path.insert(0, p)
import real_anomaly_cu_validation as C     # noqa: E402
import s1e_cu_case_noclip as E             # noqa: E402

FIGDIR = os.path.join(C.BASE, 'ForPaper', 'Rev_FigS7_Case1', 'Figures')
OUT = os.path.join(C.BASE, 'Results', '20261021_Rev_FigS7_Case1')
FS = 11
C_CASE = '#D55E00'
C_POOL = '#CC79A7'
C_SINGLE = '#0072B2'                                  # (b): each model's line in its histogram's hue
BAND = '#56B4E9'
EXPECT = {'single': 238.88, 'pooled': 344.83}          # S10 Table S10-2 (window 5-38)


def compute():
    df = pd.read_parquet(C.PROC, columns=['SystemName', 'S_ID', 'S_Date'] + C.RAS + C.RAM,
                         filters=[('A_Name', '==', 'Cu')])
    df['d'] = pd.to_datetime(df['S_Date'])
    df['sys'] = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    case = df[df['S_ID'] == C.ANOM_SID]
    assert len(case) == 1, f'expected 1 case row, got {len(case)}'
    curve = C.calc38(case)[0]
    pre = df[df['d'] < C.ANOM_DATE]
    test = df[(df['sys'] == '#3') & (df['d'] >= C.ANOM_DATE) & (df['d'] <= C.ANOM_DATE + pd.Timedelta(days=70))
              & (df['S_ID'] != C.ANOM_SID)]
    cols = C.window_cols(5, 38)
    testc = C.calc38(test)[:, cols]
    res = {}
    for key, tr in (('single', pre[pre['sys'] == '#3']), ('pooled', pre[pre['sys'].isin(['#2', '#3'])])):
        tc = C.calc38(tr.sort_values('d').iloc[-C.N_TRAIN:])
        m = E.maha_fit_noclip(tc[:, cols])
        d_case = float(C.dist(m, curve[None, cols])[0])
        d_test = C.dist(m, testc)
        res[key] = dict(thr=float(m['thr']), d=d_case, ratio=d_case / float(m['thr']), d_test=d_test,
                        rank=int((d_test > d_case).sum() + 1), n_test=len(d_test), tr_mu=tc.mean(0), tr_sd=tc.std(0))
        assert abs(res[key]['ratio'] - EXPECT[key]) < 0.01, (key, res[key]['ratio'])
    return curve, res


def draw(curve, res):
    plt.rcParams.update({'font.size': FS, 'axes.labelsize': FS + 1, 'xtick.labelsize': FS, 'ytick.labelsize': FS})
    x = np.arange(1, 39)
    s = res['single']
    paths = []

    # (a) curve: full and window 5-38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, lo in ((a1, 1), (a2, 5)):
        sl = slice(lo - 1, 38)
        if lo == 1:
            ax.axvspan(0.5, 4.5, color='0.93', zorder=0)
        ax.fill_between(x[sl], (s['tr_mu'] - 2 * s['tr_sd'])[sl], (s['tr_mu'] + 2 * s['tr_sd'])[sl], color=BAND,
                        alpha=0.45, lw=0, zorder=1, label='±2 SD, analyzer #3')
        ax.plot(x[sl], curve[sl], color=C_CASE, lw=1.8, marker='o', ms=3, zorder=3, label='Second case, 83 µg/dL')
        ax.set_xlabel('photometric point'); ax.grid(alpha=0.3)
        ax.set_xlim(lo - 0.5, 38.5)
        ax.set_title('points 1–38' if lo == 1 else 'points 5–38 (model window)', fontsize=FS + 1)
    a1.set_ylabel('Calc (RAM − RAS)'); a2.set_ylabel('Calc (RAM − RAS)')   # y scales differ (zoom)
    a2.legend(fontsize=FS, loc='lower right', framealpha=0.95)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    paths.append(os.path.join(FIGDIR, 'figS7a_cu_curve.png'))
    fig.savefig(paths[-1], dpi=600); plt.close(fig)

    # (b) distances relative to each model's own threshold
    z = {k: np.log10(res[k]['d_test'] / res[k]['thr']) for k in res}
    marks = {k: np.log10(res[k]['ratio']) for k in res}
    lo = min(min(v.min() for v in z.values()), min(marks.values())) - 0.1
    hi = max(max(v.max() for v in z.values()), max(marks.values())) + 0.6     # room for the pooled label
    bins = np.arange(lo, hi + 0.05, 0.05)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    n1, _, _ = ax.hist(z['single'], bins=bins, histtype='stepfilled', alpha=0.45, color=BAND,
                       label='analyzer #3 model')
    n2, _, _ = ax.hist(z['pooled'], bins=bins, histtype='step', lw=1.4, color=C_POOL, label='pooled #2+#3 model')
    ax.axvline(0, color='0.45', ls=':', lw=1.6, label='threshold')
    ax.set_ylim(0, 1.3 * max(n1.max(), n2.max()))
    for k, col, side, tag in (('single', C_SINGLE, 'left', '#3'), ('pooled', C_POOL, 'right', 'pooled')):
        ax.axvline(marks[k], color=col, lw=2.2)
        dx = -0.04 if side == 'left' else 0.04
        ax.text(marks[k] + dx, 0.97, f'{res[k]["ratio"]:,.0f}× ({tag})', transform=ax.get_xaxis_transform(), color=col,
                fontsize=FS, fontweight='bold', ha='right' if side == 'left' else 'left', va='top')
    ax.set_xlim(lo, hi)
    ax.set_xlabel('log10(d² / threshold)'); ax.set_ylabel('number of curves'); ax.grid(alpha=0.3)
    ax.legend(fontsize=FS, loc='center', bbox_to_anchor=(0.5, 0.55), framealpha=0.95)
    fig.tight_layout()
    paths.append(os.path.join(FIGDIR, 'figS7b_cu_distance.png'))
    fig.savefig(paths[-1], dpi=600); plt.close(fig)
    return paths


def main():
    os.makedirs(OUT, exist_ok=True)
    curve, res = compute()
    pd.DataFrame([{'model': k, 'thr': r['thr'], 'd': r['d'], 'ratio': r['ratio'], 'rank': r['rank'],
                   'n_test': r['n_test']} for k, r in res.items()]).to_csv(os.path.join(OUT, 'figS7_values.csv'),
                                                                          index=False)
    for k, r in res.items():
        print(f'{k:7s} {r["ratio"]:8.2f}x  rank {r["rank"]}/{r["n_test"]}')
    for p in draw(curve, res):
        print('wrote', p)


if __name__ == '__main__':
    main()
