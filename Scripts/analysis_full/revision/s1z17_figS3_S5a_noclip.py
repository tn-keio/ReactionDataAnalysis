# Stage 1z17 (Fig. S3, Fig. S5a): two supplement figures redrawn from unclipped results.
# The paper is computed without np.clip throughout.
#   Fig. S3 (S4, inter-analyzer transferability): inter_analyzer_figures.py drew the clipped
#     Results/20260718_InterAnalyzer/inter_analyzer_results.csv. The unclipped re-run of the same analysis (s1f)
#     wrote the same table to Results/20260917_Rev_M1_S9S4_noclip/inter/. This script runs inter_analyzer_figures.py
#     unchanged with its input and output folders redirected. The text of S4 already quotes the unclipped values
#     (e.g. HDL-C, 37% of cross transfers with a penalty >= 2%; the clipped figure gave 42%).
#   Fig. S5(a) (S8.2-S8.3, clean and benign flag rates): synthetic_validity_qc.py drew the R0 (clipped) benchmark
#     on the R0 14-item set, which includes GA (not in the 21-assay panel) and lacks 8 of the 21 assays. Here the
#     panel is the 21 assays of Table 1 and the values are the unclipped v2 benchmark behind Table 1
#     (Results/20260917_Rev_v2Benchmark, empirical Mahalanobis, N = 500, holdout P99.5). The drawing code of figA is
#     copied unchanged apart from the item list. Panels (b)-(d) of Fig. S5 do not depend on the normalization.
# Output: Results/20261025_Rev_FigS3_S5a_noclip/{Figures, inter_analyzer_summary.csv, s5a_flagrates.csv}
# py3.14 (no TensorFlow).
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(BASE, 'Results')
OUT = os.path.join(RES, '20261025_Rev_FigS3_S5a_noclip')
FIGS = os.path.join(OUT, 'Figures')
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, os.path.join(BASE, 'ForPaper'))

import inter_analyzer_figures as IF          # noqa: E402
from _common import panel as P               # noqa: E402

V2 = os.path.join(RES, '20260917_Rev_v2Benchmark')


def fig_s3():
    IF.RESULTS = os.path.join(RES, '20260917_Rev_M1_S9S4_noclip', 'inter', 'inter_analyzer_results.csv')
    IF.OUT = OUT                               # inter_analyzer_summary.csv
    IF.FIGDIR = FIGS
    IF.main()


def fig_s5a():
    rec = {}
    for g in ('main', 'ca', 'm2'):
        for line in open(os.path.join(V2, g, 'Reports', 'comparison.jsonl'), encoding='utf-8'):
            r = json.loads(line)
            if r['method'] == 'maha' and r['N'] == 500:
                rec[r['item']] = r['budgets']['99.5']
    items = [it for it, _ in P.PANEL21]
    missing = [it for it in items if it not in rec]
    assert not missing, f'no v2 maha N=500 record for {missing}'
    fr = pd.DataFrame([dict(item=it, fp_clean=rec[it]['fp_clean']['mean'], fp_benign=rec[it]['fp_benign']['mean'])
                       for it in items])
    fr.to_csv(os.path.join(OUT, 's5a_flagrates.csv'), index=False)
    print(f'S5(a): 21 assays, mean fp_clean = {fr.fp_clean.mean():.5f}, mean fp_benign = {fr.fp_benign.mean():.5f}')
    for gray in (True, False):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(items)); w = 0.4
        ax.bar(x - w / 2, fr['fp_clean'].values * 100, w, label='clean synthetic (FP on real base curves)',
               color='0.5' if gray else '#4c72b0')
        ax.bar(x + w / 2, fr['fp_benign'].values * 100, w, label='benign_offset (common-mode shift)',
               color='0.75' if gray else '#dd8452')
        ax.axhline(0.5, color='0.2' if gray else '#d62728', ls='--', lw=1.2, label='FP budget 0.5%')
        ax.set_xticks(x); ax.set_xticklabels(items, rotation=40, ha='right', fontsize=8)
        ax.set_ylabel('flag rate (%)'); ax.set_ylim(0, max(2.0, fr[['fp_clean', 'fp_benign']].values.max() * 100 * 1.2))
        ax.set_title('QC: clean ≈ real normal & benign_offset cancelled in Calc (≈ FP budget)'
                     + ('  [grayscale]' if gray else '  [color]'), fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        os.makedirs(FIGS, exist_ok=True)
        p = os.path.join(FIGS, f'figA_clean_benign_flagrate_{"gray" if gray else "color"}.png')
        fig.savefig(p, dpi=600); plt.close(fig)
        if gray:
            from PIL import Image
            Image.open(p).convert('L').save(p, dpi=(600, 600))
    return fr


def main():
    os.makedirs(FIGS, exist_ok=True)
    fig_s3()
    fig_s5a()


if __name__ == '__main__':
    main()
