# Supplementary Fig. S12 -- the two confirmed reagent-degradation episodes.
#
# The point of drawing this rather than tabulating it is that two things are visible only when the
# reaction-curve signal and the saline-blank control share a time axis:
#   * the LEAD is a gap between two curves, not a pair of dates. It also shows *why* the lead exists:
#     the two indicators start moving together, but the curve leaves its baseline scatter far sooner.
#   * the DISCORDANCE on IgG analyzer #3 -- blank rises, curve stays flat -- is immediate when the
#     two are overlaid, and easy to miss in a table of crossing dates.
# Laying IgM and IgG out in the same grid makes the asymmetric conclusion (led on one episode, missed
# on one analyzer of the other) a property of the figure rather than something the caption asserts.
#
# Layout:
#   (a)      fixed-reference monthly flag rate, five assays, 56 months -- establishes that the
#            episodes are sustained and assay-specific (CRE is the stable control)
#   (b)-(d)  IgM by analyzer: weekly curve flag % (left axis) vs saline blank median (right axis),
#            background shaded by the reagent lot in force
#   (e)-(g)  IgG by analyzer, same construction
#
# py3.14 (pandas + matplotlib). Follows FIGURE_RULES: >=600 dpi, colour-vision-safe, gray + colour.

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
from _common import panel as P   # noqa: E402

WEEKLY = os.path.join(P.RESULTS, '20260726_PerAnalyzer_LotAligned', 'per_analyzer_weekly.csv')
MONTHLY = os.path.join(P.RESULTS, '20260726_FixedReference_IgEvent', 'fixed_reference_monthly.csv')

CURVE_ALERT = 20.0
# Assay-specific saline alert, carried over from analyze_saline.py: baseline median + 5*MAD, floored
# at baseline + 1.0. IgG's blank is much noisier than IgM's, hence the higher level.
SALINE_ALERT = {'IgM': 1.00, 'IgG': 2.56}
# Right-axis range per assay: IgG's blank scatters to ~8.7 and IgM's to ~3.6, so a shared limit
# would clip the IgG series and hide exactly the excursions being discussed.
SALINE_YLIM = {'IgM': (-0.2, 4.2), 'IgG': (-0.4, 9.2)}
# An indicator counts as having alerted at the first week it reaches its level AND stays at or above
# it for two consecutive weeks. Without the persistence requirement a single noisy blank week fires
# the alarm -- on IgG analyzer #2 an isolated November spike would otherwise place the blank alert
# six weeks BEFORE the curve, which is an artefact of the blank's scatter rather than a real lead.
PERSIST_WEEKS = 2
WIN = {'IgM': ('2023-04-01', '2023-11-30'), 'IgG': ('2024-09-01', '2025-05-31')}
SUSPECT = {'IgM': '0383', 'IgG': '0406'}
OVERVIEW = ['IgM', 'IgG', 'IgA', 'KL-6', 'CRE']

BLUE, ORANGE, INK, RED = '#0072B2', '#E69F00', '#222222', '#D55E00'
OV = {'IgM': '#0072B2', 'IgG': '#D55E00', 'IgA': '#009E73', 'KL-6': '#CC79A7', 'CRE': '#999999'}


def load():
    w = pd.read_csv(WEEKLY)
    w['week'] = pd.to_datetime(w['week'])
    w['lot_in_force'] = w['lot_in_force'].fillna('-').astype(str)
    m = pd.read_csv(MONTHLY)
    m['mo'] = pd.to_datetime(m['month'] + '-01')
    return w, m


def shade_lots(ax, sub, suspect):
    """Shade the background where the implicated lot is the only lot in force."""
    on = sub['lot_in_force'].str.contains(suspect, na=False)
    only = on & ~sub['lot_in_force'].str.contains(',', na=False)
    x = sub['week'].values
    start = None
    for i, flag in enumerate(only.values):
        if flag and start is None:
            start = x[i]
        if (not flag or i == len(x) - 1) and start is not None:
            end = x[i]
            ax.axvspan(start, end, color='#FFF3D6', zorder=0)
            start = None
    return only.any()


def sustained(weeks, values, level, k=PERSIST_WEEKS):
    """First week at or above `level` that stays there for k consecutive weeks."""
    v = np.asarray(values, float)
    ok = v >= level
    for i in range(len(ok) - k + 1):
        if ok[i:i + k].all():
            return weeks.iloc[i]
    return None


def panel(ax, sub, item, analyzer, gray, show_ylabels, verdicts):
    c_curve = '0.15' if gray else BLUE
    c_sal = '0.55' if gray else ORANGE
    shade_lots(ax, sub, SUSPECT[item])

    ax.plot(sub['week'], sub['curve_pct'], color=c_curve, lw=1.8, marker='o', ms=2.6,
            label='reaction curves flagged (%)', zorder=3)
    ax.axhline(CURVE_ALERT, color=c_curve, ls=':', lw=1.0, zorder=2)
    ax.set_ylim(-3, 118)
    ax.set_title(f'{item}   analyzer {analyzer}', fontsize=10, loc='left', pad=16)
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(sub['week'], sub['sal_med'], color=c_sal, lw=1.5, ls='--', marker='s', ms=2.6,
             label='saline blank (weekly median)', zorder=3)
    ax2.set_ylim(*SALINE_YLIM[item])
    ax2.axhline(SALINE_ALERT[item], color=c_sal, ls=':', lw=1.0, zorder=2)

    tc = sustained(sub['week'], sub['curve_pct'], CURVE_ALERT)
    ts = sustained(sub['week'], sub['sal_med'].fillna(-1), SALINE_ALERT[item])
    if tc is not None:
        ax.axvline(tc, color=c_curve, lw=1.3, alpha=0.85, zorder=4)
    if ts is not None:
        ax2.axvline(ts, color=c_sal, lw=1.3, ls='--', alpha=0.95, zorder=4)

    if tc is not None and ts is not None:
        d = (ts - tc).days
        txt = f'curve leads {d} d' if d > 0 else (f'blank leads {-d} d' if d < 0 else 'same week')
        ax.annotate(txt, xy=(tc + (ts - tc) / 2, 110), ha='center', fontsize=8.5, color=INK,
                    bbox=dict(fc='white', ec='0.55', lw=0.6, pad=1.6), zorder=6)
    elif tc is None:
        ax.annotate('curve never reaches alert', xy=(sub['week'].iloc[len(sub) // 2], 70),
                    ha='center', fontsize=9, color='0.1' if gray else RED, fontweight='bold',
                    bbox=dict(fc='white', ec='0.3' if gray else RED, lw=1.0, pad=2.4), zorder=6)
    verdicts.append((item, analyzer,
                     f'{tc:%Y-%m-%d}' if tc is not None else '-',
                     f'{ts:%Y-%m-%d}' if ts is not None else '-',
                     (ts - tc).days if (tc is not None and ts is not None) else None))

    ax.tick_params(axis='x', labelrotation=40, labelsize=7.5)
    for lab in ax.get_xticklabels():
        lab.set_ha('right')
    ax.set_ylabel('curves flagged (%)' if show_ylabels else '', fontsize=8.5)
    ax2.set_ylabel('saline blank' if not show_ylabels else '', fontsize=8.5)
    ax2.tick_params(labelsize=7.5)
    ax.tick_params(axis='y', labelsize=7.5)
    return ax2


def make(gray):
    w, m = load()
    fig = plt.figure(figsize=(13.6, 11.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1, 1], hspace=0.78, wspace=0.46)

    # ---- (a) overview ----
    axa = fig.add_subplot(gs[0, :])
    for it in OVERVIEW:
        s = m[m['item'] == it].sort_values('mo')
        col = '0.6' if (gray and it != 'CRE') else ('0.2' if gray else OV[it])
        lw = 1.9 if it in ('IgM', 'IgG') else 1.1
        ls = '-' if it in ('IgM', 'IgG') else ('-.' if it == 'CRE' else '--')
        axa.plot(s['mo'], s['rate_pct'], color=col, lw=lw, ls=ls, label=it)
    axa.axhline(20, color='0.35', ls=':', lw=1.0)
    axa.set_ylabel('curves flagged (%)', fontsize=9)
    axa.set_ylim(-3, 105)
    axa.set_title('(a) Fixed-reference monthly flag rate, 2021–2025 — the two confirmed reagent '
                  'episodes against a stable control (CRE)', fontsize=10.5, loc='left')
    axa.legend(fontsize=8.5, ncol=5, loc='upper left')
    axa.grid(alpha=0.25)
    axa.tick_params(labelsize=8)

    # ---- (b)-(g) per analyzer ----
    tags = 'bcdefg'
    k = 0
    verdicts = []
    for r, item in enumerate(['IgM', 'IgG']):
        lo, hi = WIN[item]
        for c, az in enumerate(['#1', '#2', '#3']):
            ax = fig.add_subplot(gs[r + 1, c])
            sub = w[(w['item'] == item) & (w['analyzer'] == az)].sort_values('week')
            sub = sub[(sub['week'] >= lo) & (sub['week'] <= hi)]
            panel(ax, sub, item, az, gray, show_ylabels=(c == 0), verdicts=verdicts)
            # tag sits above the axes, clear of the title
            ax.text(-0.13, 1.30, f'({tags[k]})', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='top')
            k += 1

    # one shared legend under the whole grid, so no panel loses plotting area to it
    h = [Line2D([], [], color='0.15' if gray else BLUE, lw=1.8, marker='o', ms=3.5,
                label='reaction curves flagged, weekly (%)  — left axis'),
         Line2D([], [], color='0.55' if gray else ORANGE, lw=1.5, ls='--', marker='s', ms=3.5,
                label='saline blank, weekly median  — right axis'),
         Line2D([], [], color='0.4', lw=1.0, ls=':', label='alert level for each indicator'),
         Line2D([], [], color='0.4', lw=1.3, label='week each indicator first alerts (sustained ≥2 weeks)'),
         Patch(fc='#FFF3D6', ec='0.7', label='implicated reagent lot in force')]
    fig.legend(handles=h, fontsize=9, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.005),
               frameon=True, framealpha=0.95)

    print(f'  [{"gray" if gray else "color"}] alert weeks (sustained {PERSIST_WEEKS} wk):')
    for it, az, tc, ts, d in verdicts:
        print(f'    {it:4} {az}  curve {tc}  blank {ts}  '
              f'{"lead " + str(d) + " d" if d is not None else "n/a"}')

    fig.suptitle('Reagent-degradation episodes: reaction-curve signal against the saline-blank '
                 'control, by analyzer and reagent lot', fontsize=12.5, y=0.985)
    out = os.path.join(SCRIPT_DIR, 'Figures')
    os.makedirs(out, exist_ok=True)
    tag = 'gray' if gray else 'color'
    p = os.path.join(out, f'figS12_reagent_episodes_{tag}.png')
    fig.savefig(p, dpi=600, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('  wrote', os.path.basename(p))
    return p


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.parse_args()
    for g in (True, False):
        make(g)
