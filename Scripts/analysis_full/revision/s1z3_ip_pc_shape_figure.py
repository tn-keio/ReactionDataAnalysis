# Stage 1z3: visual check of the s1z2 finding at IP --
# does the type-E (baseline) perturbation coincide with a principal direction of normal variation
# after the module-2 re-base?  Same data as s1z2 (unclipped, N = 500 training curves, k >= 2 anomalies).
#   top    : PC1 / PC2 loadings of the training curves vs the median unit-norm type-E perturbation
#   bottom : training curves and type-E anomalies projected on PC1 / PC2
# Also prints |cos(E, PCk)| and the type-E shift along each PC in units of that PC's SD
# (replaces the rough "about 6 SD" estimate in the 2026-10-13 report).
# Output: Results/20261013_Rev_M1_IP_RebaseDiag/{fig_ip_pc_vs_typeE.png, pc_vs_typeE.csv}
# TF venv (rev_common imports the pipeline).

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1z2_ip_rebase_diagnostic as D     # noqa: E402  (CONF, perturbations, N, OUT)
R, M = D.R, D.M


def main():
    R.set_clip(False)
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    rows = []
    for j, (tag, (loader, syn_path)) in enumerate(D.CONF.items()):
        M.SYN = syn_path
        ctx = loader('IP', 'IP')
        Xtr = ctx['train_pool'][:D.N]; mu = Xtr.mean(0)
        ev, V = np.linalg.eigh(np.cov(Xtr.T)); ev, V = ev[::-1], V[:, ::-1]
        dE = D.perturbations(ctx)['E_baseline']
        u = dE / np.linalg.norm(dE, axis=1, keepdims=True)
        uE = np.median(u, axis=0); uE /= np.linalg.norm(uE)
        for k in range(3):
            v = V[:, k] * np.sign(V[:, k] @ uE or 1.0)
            shift = dE @ V[:, k]
            rows.append(dict(config=tag, pc=k + 1, var_share=ev[k] / ev.sum(), abs_cos_E=abs(float(v @ uE)),
                             E_shift_median_in_pc_sd=float(np.median(np.abs(shift)) / np.sqrt(ev[k]))))
        x = M.XPTS
        a = ax[0, j]
        for k, ls in ((0, '-'), (1, '--')):
            v = V[:, k] * np.sign(V[:, k] @ uE or 1.0)
            a.plot(x, v, ls, color='0.35' if k == 0 else 'tab:blue', lw=1.6,
                   label=f'PC{k + 1} ({100 * ev[k] / ev.sum():.1f}%)')
        a.plot(x, uE, color='tab:red', lw=2.2, alpha=.8, label='type E perturbation (unit norm)')
        a.axhline(0, color='0.8', lw=.8)
        a.set_title(f'IP, {"before" if tag == "pre" else "after"} re-base: loadings')
        a.set_xlabel('reaction point'); a.set_ylabel('loading'); a.legend(fontsize=8)
        b = ax[1, j]
        meta = ctx['meta']; sel = ((meta.anomaly_type == 'E_baseline') & (meta.severity >= 2)).values
        ptr = (Xtr - mu) @ V[:, :2]; pE = (ctx['syn'][sel] - mu) @ V[:, :2]
        b.scatter(ptr[:, 0], ptr[:, 1], s=6, c='0.5', alpha=.5, label=f'training curves (N={D.N})')
        b.scatter(pE[:, 0], pE[:, 1], s=6, c='tab:red', alpha=.5, label='type E anomalies (k ≥ 2)')
        b.set_xlabel('PC1 score'); b.set_ylabel('PC2 score'); b.legend(fontsize=8)
        b.set_title(f'IP, {"before" if tag == "pre" else "after"} re-base: projections')
        del ctx
    fig.tight_layout()
    fig.savefig(os.path.join(D.OUT, 'fig_ip_pc_vs_typeE.png'), dpi=200)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(D.OUT, 'pc_vs_typeE.csv'), index=False, float_format='%.4f')
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
