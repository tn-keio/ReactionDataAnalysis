# Paper figures for the CONFIRMED 5-38 / 13-item method comparison.
#   source : Results/20260717_MethodComparison_p5_38/Reports/comparison.jsonl
#   rules  : ForPaper/FIGURE_RULES.md  (>=500 DPI; gray=main text, color=supplemental;
#            distinguishable by shade+marker+linestyle; English labels; self-contained folder).
#   split  : MAIN = integrated over the 13-item panel;  SUPPL = per-item small-multiples.
# Read-only over the JSONL; run with py3.14 (no TF).
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
JSONL = os.path.join(BASE, 'Results', '20260717_MethodComparison_p5_38', 'Reports', 'comparison.jsonl')
FIGS = os.path.join(HERE, 'Figures')

METHODS = ['maha', 'maha_robust', 'pca', 'ae', 'cnn1d', 'ae2d_and', 'ae2d_ens',
           'twod_only', 'lof', 'iforest', 'ocsvm']
# Naming convention (unified so each symbol has ONE meaning):
#   "→"  = pipeline (feature extractor FEEDS a detector)           : 1D-CNN→iForest, 2D-CNN→iForest
#   "∧"/"⊕" = COMBINATION of two independent detectors (AND / ensemble), word spelled out to disambiguate
#   no bare "+"  (previously ambiguous: pipeline vs ensemble)
LABEL = {'maha': 'Mahalanobis (Empirical)', 'maha_robust': 'Mahalanobis (Robust)', 'pca': 'PCA',
         'ae': '1D-AE', 'cnn1d': '1D-CNN→iForest', 'ae2d_and': 'AE∧2D-CNN (AND)',
         'ae2d_ens': 'AE⊕2D-CNN (ensemble)', 'twod_only': '2D-CNN→iForest', 'lof': 'LOF',
         'iforest': 'Isolation Forest', 'ocsvm': 'One-Class SVM'}
N_GRID = [500, 5000, 10000]
# 13-item method-comparison panel.  GA is excluded from the study: its reagent formulation changed on
# 2024-08-05 (GA -> GA-L2), splitting the record stream into two non-comparable segments.
PANEL = ['TP', 'ALB', 'TB', 'DB', 'Ca', 'IP', 'CRE', 'UA', 'Fe', 'Cu', 'Zn', 'GLU', 'CRP']  # 13
# Colour-vision-safe palette (Okabe-Ito + Paul Tol), replacing the matplotlib
# tab10 defaults: those put maha on red and pca on green, which collapse to the
# same colour under deuteranopia (CIEDE2000 4.6) -- the paper's two headline
# methods. maha/pca are pinned to the blue/orange already used in Fig.1 and
# Fig.6 so a method keeps one colour across every figure. Verified: all pairs
# among maha/pca/maha_robust/ae separate by dE00 >= 15 under protanopia,
# deuteranopia and tritanopia; series sharing a line style separate by >= 15.
COLOR = {'maha': '#0072B2', 'maha_robust': '#222222', 'pca': '#E69F00', 'ae': '#44AA99',
         'cnn1d': '#117733', 'ae2d_and': '#332288', 'ae2d_ens': '#555555', 'twod_only': '#CC6677',
         'lof': '#AA3377', 'iforest': '#56B4E9', 'ocsvm': '#999933'}
MARK = {'maha': 'o', 'maha_robust': 'D', 'pca': 's', 'ae': '^', 'cnn1d': 'v', 'ae2d_and': 'P',
        'ae2d_ens': 'X', 'twod_only': '<', 'lof': '>', 'iforest': '*', 'ocsvm': 'h'}
LS = {'maha': '-', 'maha_robust': '-', 'pca': '--', 'ae': '--', 'cnn1d': ':', 'ae2d_and': '-.',
      'ae2d_ens': '-.', 'twod_only': ':', 'lof': '--', 'iforest': ':', 'ocsvm': ':'}
GRAY = {m: f'{v:.2f}' for m, v in zip(METHODS, np.linspace(0.0, 0.72, len(METHODS)))}
TYPES = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']

# Contaminated module-2 items are re-analysed on a CLEAN CONTEMPORANEOUS window (#2, 2021-09..2022-04):
#   Ca  - original window straddled the Aug-2021 module-2 optical event + multi-year drift (spec 0.65).
#   ALB - multi-year drift in the full-history synthetic base curves (spec 0.88 -> 0.95).
#   IP  - original training window (..2021-08-19) included ~15% of the Aug-2021 event (training cleaned).
# Replace those items' records with the clean re-runs (see Suppl09b_OperatingPoint_Ca).
CLEAN_JSONLS = [os.path.join(BASE, 'Results', '20260718_MethodComparison_Ca_clean', 'Reports', 'comparison.jsonl'),
                os.path.join(BASE, 'Results', '20260718_MethodComparison_M2_clean', 'Reports', 'comparison.jsonl')]
REPLACED = {'Ca', 'ALB', 'IP'}
R_ALL = [r for r in (json.loads(l) for l in open(JSONL, encoding='utf-8')) if r['item'] not in REPLACED]
for cj in CLEAN_JSONLS:
    if os.path.exists(cj):
        R_ALL += [r for r in (json.loads(l) for l in open(cj, encoding='utf-8')) if r['item'] in REPLACED]
print(f'clean re-runs merged for: {sorted(REPLACED)}')
R = [r for r in R_ALL if r['item'] in PANEL]
KEY = {(r['item'], r['method'], r['N']): r for r in R}


def item_vals(method, N, metric):
    return [KEY[(it, method, N)]['budgets']['99.5'][metric]['mean'] for it in PANEL]


def k1_item_vals(method, N):
    """Per-item mean over the 4 anomaly types of the k=1 recall."""
    out = []
    for it in PANEL:
        b = KEY[(it, method, N)]['budgets']['99.5']
        out.append(np.nanmean([b.get(f'recall_{t}_k1', {'mean': np.nan})['mean'] for t in TYPES]))
    return out


def mean_ci(vals):
    v = np.array(vals, float)
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return float(v.mean()), float(ci)


def style(gray, m, emph=True):
    return dict(color=(GRAY[m] if gray else COLOR[m]), marker=MARK[m], ls=LS[m],
                lw=2.8 if (emph and m.startswith('maha')) else 1.4, ms=6)


def side_legend(ax, nrow_hint=12):
    """Legend outside the axes, one entry per row.

    Inside the axes a 2-column legend covered the low-recall series (One-Class
    SVM, Isolation Forest, 1D-CNN->iForest in fig1; both Mahalanobis curves in
    fig2).  Parking it to the right of the axes keeps every series visible, and
    a single column preserves the top-to-bottom reading order of the methods.
    Callers must widen figsize to pay for the strip and save with tight bbox.
    """
    return ax.legend(fontsize=8, ncol=1, loc='center left', bbox_to_anchor=(1.02, 0.5),
                     frameon=True, borderaxespad=0.0, handlelength=2.6, labelspacing=0.6)


def save(fig, name, gray, tight_bbox=False):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, f'{name}_{"gray" if gray else "color"}.png')
    # tight_bbox is required when an artist sits outside the axes (side_legend):
    # tight_layout() does not reserve room for it, so the default crop cuts it.
    fig.savefig(p, dpi=600, **({'bbox_inches': 'tight'} if tight_bbox else {}))
    plt.close(fig)
    if gray:
        try:
            from PIL import Image
            Image.open(p).convert('L').save(p, dpi=(600, 600))
        except Exception:
            pass   # figure is already gray-styled; single-channel conversion is cosmetic


# ================= MAIN (integrated over 13 items) =================
def fig1_recall_vs_N(gray):
    # wider than the 8in default: the extra ~2in is the legend strip on the right
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in METHODS:
        ys, es = zip(*[mean_ci(item_vals(m, N, 'recall_main')) for N in N_GRID])
        ax.errorbar(N_GRID, ys, yerr=es, label=LABEL[m], capsize=3, **style(gray, m))
    # worst-item overlay for Mahalanobis(Empirical): even the hardest item stays high
    worst = [min(item_vals('maha', N, 'recall_main')) for N in N_GRID]
    ax.plot(N_GRID, worst, color=(GRAY['maha'] if gray else COLOR['maha']), ls=(0, (1, 1)),
            lw=1.6, marker='o', ms=4, mfc='white', label='Mahalanobis — worst item')
    ax.set_xscale('log'); ax.set_xticks(N_GRID); ax.set_xticklabels([str(n) for n in N_GRID])
    ax.set_ylim(0, 1.03); ax.set_xlabel('Training samples (N)')
    ax.set_ylabel('Detection rate (recall of intended anomalies, k≥2)')
    # Panel size is read from PANEL, not hard-coded: the Rev15 wrapper swaps PANEL for the 21-item
    # set, and a fixed "13-item panel" string printed the wrong count on the manuscript's figure.
    ax.set_title(f'Anomaly detection vs training size — points 5–38, {len(PANEL)}-item panel\n'
                 'mean ± 95% CI over items', fontsize=10)
    ax.grid(alpha=0.3); side_legend(ax)
    fig.tight_layout(); save(fig, 'fig1_recall_vs_N', gray, tight_bbox=True)


def fig2_recall_k1(gray):
    fig, ax = plt.subplots(figsize=(10, 5.5))   # see fig1_recall_vs_N
    for m in METHODS:
        ys, es = zip(*[mean_ci(k1_item_vals(m, N)) for N in N_GRID])
        ax.errorbar(N_GRID, ys, yerr=es, label=LABEL[m], capsize=3, **style(gray, m))
    ax.set_xscale('log'); ax.set_xticks(N_GRID); ax.set_xticklabels([str(n) for n in N_GRID])
    ax.set_ylim(0, 1.03); ax.set_xlabel('Training samples (N)')
    ax.set_ylabel('Recall of SUBTLE anomalies (k=1, mean over 4 types)')
    ax.set_title(f'Subtle (1σ) anomaly detection vs training size — points 5–38, '
                 f'{len(PANEL)}-item panel\n'
                 'mean ± 95% CI over items', fontsize=10)
    ax.grid(alpha=0.3); side_legend(ax)
    fig.tight_layout(); save(fig, 'fig2_recall_k1_subtle', gray, tight_bbox=True)


def fig3_operating_summary(gray):
    """N=500 (operating point): per-method mean recall_main with worst-item marker + specificity."""
    N = 500
    order = sorted(METHODS, key=lambda m: -np.mean(item_vals(m, N, 'recall_main')))
    rm = [np.mean(item_vals(m, N, 'recall_main')) for m in order]
    worst = [min(item_vals(m, N, 'recall_main')) for m in order]
    spec = [np.mean(item_vals(m, N, 'specificity_clean')) for m in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Three marks share this panel, so they must separate for every kind of
    # colour vision: pale bars carry the mean, a vermillion triangle the worst
    # item, an ink dash the specificity. (The former #d62728 / #2ca02c pair was
    # the red-green collision this palette exists to avoid.)
    bcol = ['0.55' if gray else '#A6C8E0'] * len(order)
    ax.bar(x, rm, color=bcol, width=0.62, label='Mean recall (k≥2)')
    ax.plot(x, worst, ls='none', marker='v', ms=8,
            color=('0.0' if gray else '#D55E00'), label='Worst-item recall')
    ax.plot(x, spec, ls='none', marker='_', ms=16, mew=2.4,
            color=('0.25' if gray else '#222222'), label='Specificity (clean)')
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in order], rotation=40, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Rate')
    ax.set_title(f'Operating point at N=500, P99.5 (points 5–38, {len(PANEL)}-item panel)', fontsize=10)
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout(); save(fig, 'fig3_operating_summary', gray)


def fig4_recall_by_type(gray):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    for ax, t in zip(axes, TYPES):
        for m in METHODS:
            ys = [mean_ci(item_vals(m, N, f'recall_{t}'))[0] for N in N_GRID]
            ax.plot(N_GRID, ys, label=LABEL[m], **style(gray, m))
        ax.set_xscale('log'); ax.set_xticks(N_GRID); ax.set_xticklabels([str(n) for n in N_GRID])
        ax.set_ylim(0, 1.03); ax.set_title(t); ax.set_xlabel('N'); ax.grid(alpha=0.3)
    axes[0].set_ylabel('Recall (k≥2)')
    axes[-1].legend(fontsize=7, ncol=1, loc='lower right')
    fig.suptitle('Recall by anomaly type vs training size — points 5–38', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, 'fig4_recall_by_type', gray)


def fig5_compute_cost(gray):
    fit = {m: np.nanmean([KEY[(it, m, 10000)]['fit_time_s'] for it in PANEL]) for m in METHODS}
    inf = {m: np.nanmean([KEY[(it, m, 10000)]['infer_per_sample_s'] * 1e3 for it in PANEL]) for m in METHODS}
    order = METHODS
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cols = ['0.4'] * len(order) if gray else [COLOR[m] for m in order]
    axes[0].bar(x, [fit[m] for m in order], color=cols)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Training time (s, log) at N=10000')
    axes[1].bar(x, [inf[m] for m in order], color=cols)
    axes[1].set_yscale('log'); axes[1].set_ylabel('Inference time (ms/sample, log)')
    for a in axes:
        a.set_xticks(x); a.set_xticklabels([LABEL[m] for m in order], rotation=40, ha='right', fontsize=7.5)
        a.grid(alpha=0.3, axis='y')
    fig.suptitle('Computational cost per method', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); save(fig, 'fig5_compute_cost', gray)


# ================= SUPPLEMENTAL (per-item small multiples) =================
def figS_byitem(metric_fn, ylabel, name, gray, title=None):
    ncol = 4; nrow = int(np.ceil(len(PANEL) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, it in zip(axes, PANEL):
        for m in METHODS:
            ys = [metric_fn(it, m, N) for N in N_GRID]
            ax.plot(N_GRID, ys, label=LABEL[m], **style(gray, m, emph=True))
        ax.set_xscale('log'); ax.set_xticks(N_GRID); ax.set_xticklabels([str(n) for n in N_GRID], fontsize=7)
        ax.set_ylim(0, 1.03); ax.set_title(it, fontsize=10); ax.grid(alpha=0.3)
    for ax in axes[len(PANEL):]:
        ax.axis('off')
    for i in range(0, len(PANEL), ncol):
        axes[i].set_ylabel(ylabel, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    # a reader-facing title, not the output file name
    fig.suptitle(title or name.replace('_', ' '), fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97]); save(fig, name, gray)


def figS1(gray):
    figS_byitem(lambda it, m, N: KEY[(it, m, N)]['budgets']['99.5']['recall_main']['mean'],
                'recall (k≥2)', 'figS1_recall_vs_N_byitem', gray,
                title='Detection recall vs training size, by item — points 5–38, 13-item panel')


def figS2(gray):
    figS_byitem(lambda it, m, N: float(np.nanmean(
        [KEY[(it, m, N)]['budgets']['99.5'].get(f'recall_{t}_k1', {'mean': np.nan})['mean'] for t in TYPES])),
                'recall (k=1)', 'figS2_recall_k1_byitem', gray,
                title='Subtle (1σ) anomaly recall vs training size, by item — points 5–38, 13-item panel')


FIGURES = {'fig1': fig1_recall_vs_N, 'fig2': fig2_recall_k1, 'fig3': fig3_operating_summary,
           'fig4': fig4_recall_by_type, 'fig5': fig5_compute_cost, 'figS1': figS1, 'figS2': figS2}


def main(only=()):
    """Regenerate every figure, or only the named ones (e.g. `... fig1 fig2`).

    Selecting a subset keeps a cosmetic change to one panel from rewriting the
    other outputs, which are already synced into the paper tree.
    """
    print('items:', sorted(set(r['item'] for r in R)), '| records:', len(R))
    names = [n for n in FIGURES if not only or n in only]
    if only and (unknown := sorted(set(only) - set(FIGURES))):
        raise SystemExit(f'unknown figure(s): {unknown} | choose from {list(FIGURES)}')
    print('generating:', names)
    for gray in (True, False):
        for n in names:
            FIGURES[n](gray)
    print('DONE ->', FIGS)


if __name__ == '__main__':
    import sys
    main(only=tuple(a for a in sys.argv[1:] if not a.startswith('-')))
