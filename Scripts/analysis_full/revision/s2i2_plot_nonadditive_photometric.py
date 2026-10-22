# Stage 2i-2 (Reviewer #2, Methods-2; 2026-10-21): redraw of s2i's Fig. 1 (non-additive anomalies on a typical base
# curve, CRE and IgG) for the supplement, consistent with the revised M2 analysis.
#   Changes from s2i fig1 (the 2026-09-26 file is kept unchanged):
#   (1) the legend bias is the two-point endpoint read-out of s2a2 (photometric points, liquid-volume correction f,
#       median per-analyzer-month calibration slope, % of the assay's median reported value), not the earlier
#       reaction-phase approximation; it is the median over the curves whose reported value the anomaly changed
#       (s2a2's mask: |dA change| > 1% of the median clean |dA|), with that number of curves out of 200;
#   (2) the second-reagent addition is drawn after point 19 (author, 2026-10-20: the last pre-addition point is 19
#       for every assay); the first point modified by the warp/cap (t0 + 2, where t0 is the generator's reaction onset
#       and ts = t0 + 1 the unmodified anchor) is drawn
#       separately (CRE t0 = point 22, IgG t0 = point 20; s2a recovers t0 from the E_baseline k = 1 rows);
#   (3) the photometric read points of the assay are shaded (IgG end points 29-30 precede most of the cap levels).
# Output: Results/20261021_Rev_M2_BatteryFigures/{fig1_nonadditive.png, fig1_nonadditive_bias.csv}
# py3.14 (no TensorFlow).
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2i_plot_battery_examples as S          # noqa: E402  (style, colours, base-curve choice)
import s2a_make_battery_v2 as A                # noqa: E402
import s2a2_bias_photometric as B2             # noqa: E402

OUT = os.path.join(S.BASE, 'Results', '20261021_Rev_M2_BatteryFigures')
CAL = os.path.join(S.BASE, 'Results', '20261020_Rev_M2_BiasPhotometric', 'calibration_fit.csv')
R2_AFTER = 19                                   # second reagent is added after this point (all assays)
READ = '#FFE08A'


def main():
    os.makedirs(OUT, exist_ok=True)
    sp_all = B2.settings()[0]
    cal = pd.read_csv(CAL).set_index('item')
    prof = pd.read_csv(S.PROF)
    items = [('CRE', 'CRE'), ('IgG', 'IGG')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), sharex=True)
    rows = []
    for r, (it, key) in enumerate(items):
        sp = sp_all[it]
        assert sp['key'] == key and sp['group'] == 'main'
        df = pd.read_parquet(os.path.join(A.SYN_DIR, A.SRC['main']), filters=[('item', '==', key)]).reset_index(drop=True)
        clean = A.by_base(df, 'clean', 0.0)
        bid, xb = S.pick_base(clean)
        d0 = pd.Series(B2.delta_a(S.calc(clean), sp), index=clean.base_id.values)
        tol = 1e-2 * np.median(np.abs(d0.values))
        b, cmed = cal.loc[it, 'slope_block_median'], cal.loc[it, 'median_value']
        ext, _ = A.build_item(df)                 # W_timewarp, S_saturation exactly as in s2a / s2a2
        t0 = int(prof[(prof.group == 'main') & (prof.item == key)]['onset_point'].iloc[0])
        for c, (atype, name, sym) in enumerate([('W_timewarp', 'Kinetic slowing (time warp of the reaction phase)', 'r'),
                                                ('S_saturation', 'Premature termination (reaction-phase cap)', 'q')]):
            ax = axes[r, c]
            for pts in (sp['base'], sp['end']):
                lo, hi = A.P0 + min(pts), A.P0 + max(pts)
                ax.axvspan(lo - 0.5, hi + 0.5, color=READ, alpha=0.6, lw=0, zorder=0)
            g = ext[ext.anomaly_type == atype]
            sev = sorted(g.severity.unique(), reverse=True)        # mild -> severe
            cols = S.ramp_n(len(sev))
            ax.plot(S.PTS, xb, color=S.GREY, lw=2, ls='--', label='base (normal curve)', zorder=3)
            for s, col in zip(sev, cols):
                gs = g[np.isclose(g.severity, s)].set_index('base_id').loc[clean.base_id.values]
                dd = B2.delta_a(S.calc(gs), sp) - d0.values
                aff = np.abs(dd) > tol
                bias = 100 * b * dd / cmed
                med = float(np.median(bias[aff])) if aff.any() else 0.0
                y = S.calc(gs.loc[[bid]])[0]
                ax.plot(S.PTS, y, color=col, lw=1.8, zorder=2,
                        label=f'{sym} = {s:g}  ({med:+.1f}%, {int(aff.sum())}/200)' if aff.any() else f'{sym} = {s:g}  (no change)')
                rows.append(dict(item=it, type=atype, severity=s, n_affected=int(aff.sum()),
                                 median_bias_affected_pct=round(med, 2),
                                 # unsigned, as s2a2's react_median, for a direct cross-check
                                 median_abs_bias_affected_pct=round(float(np.median(np.abs(bias[aff]))) if aff.any() else 0.0, 2),
                                 median_bias_all_pct=round(float(np.median(bias)), 2)))
            # ts = t0 + 1 is the unmodified anchor (s2a: warp for x > ts, cap equal at ts), so the first modified
            # point is t0 + 2; the line sits between them
            first_mod = t0 + 2
            ax.axvline(R2_AFTER + 0.5, color=S.INK, lw=1, ls=':')
            ax.axvline(first_mod - 0.5, color=S.MUTED, lw=1, ls='-.')
            tr = ax.get_xaxis_transform()                       # x in data units, y in axes fraction
            ax.text(R2_AFTER + 0.4, 0.99, '2nd reagent ', fontsize=7.5, color=S.INK, va='top', ha='right', transform=tr)
            ax.text(first_mod - 0.4, 0.99, ' anomaly applied', fontsize=7.5, color=S.MUTED, va='top', ha='left', transform=tr)
            lo_, hi_ = ax.get_ylim()
            ticks = [t for t in ax.get_yticks() if lo_ <= t <= hi_]
            ax.set_ylim(lo_ - 0.55 * (hi_ - lo_), hi_)           # room for the legend below the curves
            ax.set_yticks(ticks)                                  # no ticks in the empty legend area
            S.style(ax, f'{it} - {name}', ylabel=(c == 0))
            ax.legend(fontsize=7.3, frameon=False, loc='lower right', title='level  (median bias, curves affected)',
                      title_fontsize=7.3)
    fig.suptitle('Non-additive synthetic anomalies added in the revision (one typical base curve per assay)',
                 fontsize=11, color=S.INK, x=0.01, ha='left')
    fig.text(0.01, 0.005,
             'Shaded: photometric read points (baseline and end points). Bias: change of the reported result by the two-point '
             'endpoint read-out, % of the assay\'s median value, median over the curves whose\nresult changed '
             '(|change in dA| > 1% of the median reaction), with their number out of 200 base curves; "no change": the cap lies '
             'beyond the end points.\nChanges beyond -100% (CRE) mean the perturbed result would be negative '
             '(calibration intercept -4.47 mg/dL against a median of 0.83 mg/dL).\n'
             'q = 0.5 and 0.2 were tested only in the five immunoturbidimetric assays and CRE.',
             fontsize=7.3, color=S.MUTED)
    fig.tight_layout(rect=[0, 0.065, 1, 0.96])
    fig.savefig(os.path.join(OUT, 'fig1_nonadditive.png'), dpi=200, facecolor='white')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'fig1_nonadditive_bias.csv'), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
