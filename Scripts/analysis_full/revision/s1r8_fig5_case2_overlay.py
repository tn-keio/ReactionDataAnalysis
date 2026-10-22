# Stage 1r-8: compact colour version of Fig. 5.
# The three measurements of the second Cu case are overlaid in ONE curve panel and ONE distance panel instead of
# three panels each.  Values come from s1r7.compute() unchanged (same protocol as s1r5).
#   (a) the three curves; normal +/-2 SD band of each training analyzer (#3 for the initial measurement,
#       #2 for both repeats); excluded points 1-4 shaded
#   (b) d^2 / threshold of the contemporaneous curves of each analyzer (each scored by its own model), so that the
#       two analyzers share one threshold line (ratio 1, i.e. 0 on the log10 axis); the three measurements as
#       vertical lines; legend below the axes so that it does not cover the lines
#   (c) photograph (same file as s1r7)
# The figure carries no sample identifier or date.
# Output: ForPaper/Rev_Fig5_Case2/Figures_overlay/ (panels and the composed figure, colour only)
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, '..'))):
    if p not in sys.path:
        sys.path.insert(0, p)
import s1r7_fig5_case2 as F7          # noqa: E402

FIGDIR = os.path.join(F7.C.BASE, 'ForPaper', 'Rev_Fig5_Case2', 'Figures_overlay')
ORDER = ['initial', 'repeat1_3x', 'repeat2_5x_reported']
COL = {'initial': '#D55E00', 'repeat1_3x': '#E69F00', 'repeat2_5x_reported': '#0072B2'}   # Okabe-Ito
# Legends are kept short and large; analyzer, training-set size, the shaded points 1-4,
# the 70-day test window and the ranks are given in the caption instead.
NAME = {'initial': 'Initial, −5 µg/dL',
        'repeat1_3x': '3-fold dilution, 93 µg/dL',
        'repeat2_5x_reported': '5-fold dilution, 99 µg/dL (reported)'}
BAND = {'#3': '0.25', '#2': '#56B4E9'}      # #3: dark outline (dashed band in (a), step histogram in (b))
FS = 11                                     # legend and annotation font size (pt)


def fmt_ratio(r):
    return f'{r:,.2f}×' if r < 10 else f'{r:,.0f}×'


def draw(res):
    by = {r['measurement']: r for r in res}
    assert set(by) == set(ORDER), sorted(by)
    # the two repeats share analyzer #2, hence one training set and one threshold
    assert by['repeat1_3x']['analyzer'] == by['repeat2_5x_reported']['analyzer'] == '#2'
    assert by['initial']['analyzer'] == '#3'
    assert by['repeat1_3x']['thr'] == by['repeat2_5x_reported']['thr']
    x = np.arange(1, 39)
    plt.rcParams.update({'font.size': FS, 'axes.labelsize': FS + 1, 'xtick.labelsize': FS, 'ytick.labelsize': FS})
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={'width_ratios': [1.15, 1]})

    # (a) curves; #2 band filled, #3 band as dashed edges so that it stays visible where the two bands overlap
    axa.axvspan(0.5, 4.5, color='0.93', zorder=0)
    r = by['repeat1_3x']
    axa.fill_between(x, r['tr_mu'] - 2 * r['tr_sd'], r['tr_mu'] + 2 * r['tr_sd'], color=BAND['#2'],
                     alpha=0.45, lw=0, zorder=1, label='±2 SD, analyzer #2')
    r = by['initial']
    for k, sgn in enumerate((-2, 2)):
        axa.plot(x, r['tr_mu'] + sgn * r['tr_sd'], color=BAND['#3'], ls='--', lw=1.0, zorder=2,
                 label='±2 SD, analyzer #3' if k == 0 else None)
    for m in ORDER:
        axa.plot(x, by[m]['curve'], color=COL[m], lw=1.8, marker='o', ms=3, zorder=3, label=NAME[m])
    axa.set_xlabel('photometric point'); axa.set_ylabel('Calc (RAM − RAS)')
    axa.set_xlim(0.5, 38.5); axa.grid(alpha=0.3)
    hl = {l_: h_ for h_, l_ in zip(*axa.get_legend_handles_labels())}
    want = [NAME[m] for m in ORDER] + ['±2 SD, analyzer #2', '±2 SD, analyzer #3']   # curves first, bands last
    assert set(want) == set(hl), sorted(hl)
    axa.legend([hl[k] for k in want], want, fontsize=FS, loc='upper right', framealpha=0.95)

    # (b) distances relative to each model's own threshold; the measurements are labelled on the lines themselves
    z = {an: np.log10(by[k]['d_test'] / by[k]['thr']) for an, k in (('#3', 'initial'), ('#2', 'repeat1_3x'))}
    marks = [np.log10(by[m]['ratio']) for m in ORDER]
    lo = min(min(v.min() for v in z.values()), min(marks)) - 0.1
    hi = max(max(v.max() for v in z.values()), max(marks)) + 0.3
    bins = np.arange(lo, hi + 0.05, 0.05)          # fixed width, so the narrow clean distributions keep their shape
    n2, _, _ = axb.hist(z['#2'], bins=bins, histtype='stepfilled', alpha=0.45, color=BAND['#2'],
                        label='analyzer #2')
    n3, _, _ = axb.hist(z['#3'], bins=bins, histtype='step', lw=1.2, color=BAND['#3'], label='analyzer #3')
    axb.axvline(0, color='0.45', ls=':', lw=1.6, label='threshold')
    axb.set_ylim(0, 1.3 * max(n2.max(), n3.max()))
    for m, xm in zip(ORDER, marks):
        axb.axvline(xm, color=COL[m], lw=2.2)
        right = m != 'initial'                       # the initial line sits near the right edge: label on its left
        axb.text(xm + (0.06 if right else -0.06), 0.97, fmt_ratio(by[m]['ratio']), transform=axb.get_xaxis_transform(),
                 color=COL[m], fontsize=FS, fontweight='bold', ha='left' if right else 'right', va='top')
    axb.set_xlim(lo, hi)
    axb.set_xlabel('log10(d² / threshold)'); axb.set_ylabel('number of curves')
    axb.grid(alpha=0.3)
    axb.legend(fontsize=FS, loc='center', bbox_to_anchor=(0.5, 0.55), framealpha=0.95)
    for ax, lab in ((axa, '(a)'), (axb, '(b)')):
        ax.text(-0.12, 1.02, lab, transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, 'figAB_case2_overlay_color.png')
    fig.savefig(p, dpi=600)
    plt.close(fig)
    return p


def compose(pab):
    ab = Image.open(pab).convert('RGB')
    c = Image.open(F7.PHOTO).convert('RGB')
    h = int(ab.height * 0.92)
    c = c.resize((int(c.width * h / c.height), h))
    margin, gap, lab_w = 60, 90, 150
    W = margin + ab.width + gap + lab_w + c.width + margin
    H = margin + ab.height + margin
    canvas = Image.new('RGB', (W, H), 'white')
    canvas.paste(ab, (margin, margin))
    x_c = margin + ab.width + gap + lab_w
    canvas.paste(c, (x_c, margin + (ab.height - h) // 2))
    ImageDraw.Draw(canvas).text((x_c - lab_w + 10, margin), '(c)', fill='black',    # top, level with (a)(b)
                                font=F7.R_FONT(int(14 * 600 / 72)))
    p = os.path.join(FIGDIR, 'fig5_real_anomaly_case2_overlay_color.png')
    canvas.save(p, dpi=(600, 600))
    return p


def main():
    res = F7.compute()
    for r in res:
        print(f'{r["measurement"]:22s} {r["analyzer"]} {r["value"]:6.0f}  {r["ratio"]:9.2f}x  rank {r["rank"]}/{r["n_test"]}')
    print('wrote', compose(draw(res)))


if __name__ == '__main__':
    main()
