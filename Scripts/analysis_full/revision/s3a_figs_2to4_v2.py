# Stage 3a (M1-T11, F1-01, M1-U12): Fig. 2-4 redrawn from the v2 benchmark (unclipped, 21 items, the 13 Table 1
# methods), in two styles for the author to compare (decision 2026-10-19: draw both, then choose):
#   original   : every method in its own colour/marker/line style (the submitted style)
#   simplified : five methods in colour (empirical and robust Mahalanobis, PCA, LOF, the elbow-selected compact AE;
#                LOF included because the text discusses it at N = 5,000), the other eight in
#                thin light grey drawn underneath (zorder), also in Fig. 4(b)
# The plotting functions of ForPaper/04_MethodComparison/Figures_p5_38/make_figures.py are used unchanged;
# only its module globals (records, panel, method list, styles, output folder) are replaced, as the Rev15
# wrapper does.  dae_elbow is resolved per (item, N) with the corrected selection (dae_z_selection_fixed.csv).
# Error bars stay "mean +/- 95% CI over items" (R1-03: the legends must say so).
# Output: ForPaper/Rev_Figs2to4_v2/Figures_{original,simplified}/  (gray and colour)
# py3.14 (no TensorFlow).
import os
import sys
import json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
FIGMOD = os.path.join(BASE, 'ForPaper', '04_MethodComparison', 'Figures_p5_38')
sys.path.insert(0, FIGMOD)
import make_figures as F     # noqa: E402  (builds the old 13-item records at import; replaced below)

V2 = os.path.join(BASE, 'Results', '20260917_Rev_v2Benchmark')
SEL = os.path.join(BASE, 'Results', '20261013_Rev_M3_CompactSelFix', 'dae_z_selection_fixed.csv')
OUT = os.path.join(BASE, 'ForPaper', 'Rev_Figs2to4_v2')
PANEL21 = ['TP', 'ALB', 'TB', 'DB', 'Ca', 'IP', 'CRE', 'UA', 'CRP', 'TC', 'HDL-C', 'LDL-C', 'TG', 'Fe',
           'IgG', 'KL-6', 'IgA', 'IgM', 'GLU', 'Zn', 'Cu']
METHODS13 = ['maha', 'maha_robust', 'pca', 'dae_elbow', 'cae8', 'lof', 'ae2d_ens', 'ae', 'twod_only',
             'ae2d_and', 'ocsvm', 'iforest', 'cnn1d']
HIGHLIGHT = ['maha', 'maha_robust', 'pca', 'lof', 'dae_elbow']
NEW_LABEL = {'ae': '1D-AE (over-complete, 72-d)', 'dae_elbow': 'Compact AE (dense, elbow-selected z)',
             'cae8': 'Compact AE (conv., z = 8 fixed)'}
# vermillion / light purple: away from the teal 1D-AE (#44AA99) and green 1D-CNN (#117733) of the palette
NEW_COLOR = {'dae_elbow': '#D55E00', 'cae8': '#CC79A7'}
NEW_MARK = {'dae_elbow': 'p', 'cae8': 'd'}
NEW_LS = {'dae_elbow': '-', 'cae8': '--'}


def load_records():
    recs = []
    for g in ('main', 'ca', 'm2'):
        for l in open(os.path.join(V2, g, 'Reports', 'comparison.jsonl'), encoding='utf-8'):
            if l.strip():
                recs.append(json.loads(l))
    key = {(r['item'], r['method'], r['N']): r for r in recs}
    sel = pd.read_csv(SEL)
    for _, s in sel.iterrows():
        src = key.get((s['item'], f"dae{int(s['z_elbow'])}", int(s['N'])))
        if src is not None:
            recs.append(dict(src, method='dae_elbow'))
    return [r for r in recs if r['method'] in METHODS13 and r['item'] in PANEL21]


def setup(style):
    F.PANEL = PANEL21
    F.METHODS = METHODS13
    F.LABEL.update(NEW_LABEL)
    if not hasattr(F, '_orig_color'):
        F._orig_color = dict(F.COLOR, **NEW_COLOR)
        F._orig_style = F.style
    orig_style = F._orig_style
    F.COLOR = dict(F._orig_color); F.MARK.update(NEW_MARK); F.LS.update(NEW_LS)
    F.GRAY = {m: f'{v:.2f}' for m, v in zip(METHODS13, [i * 0.72 / 12 for i in range(13)])}
    if style == 'original':
        F.style = orig_style
    else:
        hi_gray = dict(zip(HIGHLIGHT, ['0.0', '0.25', '0.45', '0.35', '0.6']))
        for m in METHODS13:                         # Fig. 4(b) reads F.COLOR directly, not style()
            if m not in HIGHLIGHT:
                F.COLOR[m] = '#C8C8C8'
        F.GRAY = {m: hi_gray.get(m, '0.80') for m in METHODS13}

        def simple(gray, m, emph=True):
            if m in HIGHLIGHT:
                d = orig_style(gray, m, emph)
                d.update(lw=2.8 if m.startswith('maha') else 2.2, zorder=4)
                return d
            return dict(color='0.80' if gray else '#C8C8C8', marker=F.MARK[m], ls='-', lw=1.0, ms=3.5, zorder=2)
        F.style = simple
    F.FIGS = os.path.join(OUT, f'Figures_{style}')
    os.makedirs(F.FIGS, exist_ok=True)


def main():
    F.R_ALL = F.R = load_records()
    F.KEY = {(r['item'], r['method'], r['N']): r for r in F.R}
    miss = [(i, m, n) for i in PANEL21 for m in METHODS13 for n in F.N_GRID if (i, m, n) not in F.KEY]
    assert not miss, f'missing records: {miss[:5]} ... ({len(miss)})'
    for style in ('original', 'simplified'):
        setup(style)
        for gray in (True, False):
            F.fig1_recall_vs_N(gray)          # Fig. 2
            F.fig2_recall_k1(gray)            # Fig. 3
            F.fig3_operating_summary(gray)    # Fig. 4(a)
            F.fig5_compute_cost(gray)         # Fig. 4(b)
        print(style, '->', F.FIGS, sorted(os.listdir(F.FIGS)))


if __name__ == '__main__':
    main()
