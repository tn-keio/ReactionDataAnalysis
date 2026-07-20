# Validity QC of the synthetic anomaly battery (paper §6 / reviewer defense).  Three checks:
#   QC1  clean ≈ real normal        : clean-synthetic flag rate (= FP on real base curves) ≈ FP budget.
#   QC2  benign_offset design control: a common-mode offset (same A added to RAM & RAS) leaves Calc=RAM−RAS
#        UNCHANGED -> must not be flagged (specificity built into the dual-wavelength channel).  Verify Calc
#        invariance in the data, and fp_benign ≈ fp_clean.
#   QC3  severity comparability      : achieved deviation ≈ k·σ̄ across items (k makes difficulty comparable).
# py3.14 + scipy stack.  No TF.
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SYN = os.path.join(BASE, 'Data', 'BigData', 'synthetic_anomalies', 'endpoint_synthetic_anomalies_p5_38.parquet')
MAIN = os.path.join(BASE, 'Results', '20260717_MethodComparison_p5_38')
CACLEAN = os.path.join(BASE, 'Results', '20260718_MethodComparison_Ca_clean')
M2CLEAN = os.path.join(BASE, 'Results', '20260718_MethodComparison_M2_clean')
OUT = os.path.join(BASE, 'Results', '20260718_SyntheticValidity')
FIGDIR = os.path.join(BASE, 'ForPaper', 'Suppl07_SyntheticAnomalyValidity', 'Figures')
RAM = [f'RAM{i}' for i in range(5, 39)]; RAS = [f'RAS{i}' for i in range(5, 39)]
# panel proc-name (jsonl) -> synthetic item code
PANEL = ['TP', 'ALB', 'TB', 'DB', 'Ca', 'IP', 'CRE', 'UA', 'Fe', 'Cu', 'Zn', 'GLU', 'GA', 'CRP']
SYNC = {'Ca': 'CA', 'Fe': 'FE', 'Cu': 'CU', 'Zn': 'ZN'}
POS = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']


def save(fig, name, gray):
    for d in (FIGDIR, OUT):
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f'{name}_{"gray" if gray else "color"}.png')
        fig.savefig(p, dpi=600)
    if gray:
        try:
            from PIL import Image
            for d in (FIGDIR, OUT):
                p = os.path.join(d, f'{name}_gray.png'); Image.open(p).convert('L').save(p, dpi=(600, 600))
        except Exception:
            pass
    plt.close(fig)


def merged_metrics():
    recs = [r for r in (json.loads(l) for l in open(os.path.join(MAIN, 'Reports', 'comparison.jsonl'), encoding='utf-8'))
            if r['item'] not in ('Ca', 'ALB', 'IP')]
    for f in [os.path.join(CACLEAN, 'Reports', 'comparison.jsonl'), os.path.join(M2CLEAN, 'Reports', 'comparison.jsonl')]:
        recs += [r for r in (json.loads(l) for l in open(f, encoding='utf-8')) if r['item'] in ('Ca', 'ALB', 'IP')]
    return {(r['item'], r['method'], r['N']): r for r in recs}


def main():
    os.makedirs(OUT, exist_ok=True)
    syn = pd.read_parquet(SYN)
    key = merged_metrics()

    # ---------- QC2: benign_offset Calc invariance (data-level proof) ----------
    print('=== QC2: benign_offset leaves Calc=RAM-RAS unchanged (dual-wavelength specificity) ===')
    inv_rows = []
    for it in PANEL:
        code = SYNC.get(it, it)
        s = syn[syn['item'] == code]
        clean = s[s['anomaly_type'] == 'clean'].set_index('base_id')
        ben = s[s['anomaly_type'] == 'benign_offset']
        if len(clean) == 0 or len(ben) == 0:
            continue
        cc = clean[RAM].values - clean[RAS].values           # (nbase, 34) clean Calc, indexed by base order
        clean_calc = {b: (clean.loc[b, RAM].values - clean.loc[b, RAS].values) for b in clean.index}
        maxdiff = 0.0; ram_shift = []
        for _, r in ben.iterrows():
            b = r['base_id']
            if b not in clean_calc:
                continue
            calc_b = r[RAM].values.astype(float) - r[RAS].values.astype(float)
            maxdiff = max(maxdiff, float(np.abs(calc_b - clean_calc[b]).max()))
            ram_shift.append(float((r[RAM].values.astype(float) - (clean.loc[b, RAM].values)).mean()))
        inv_rows.append(dict(item=it, max_calc_diff=round(maxdiff, 4), mean_RAM_shift=round(np.mean(ram_shift), 1)))
        print(f'  {it:5}: max|ΔCalc(benign−clean)| = {maxdiff:.4f}   (mean RAM offset applied = {np.mean(ram_shift):+.0f})')
    inv = pd.DataFrame(inv_rows)
    print(f'  -> ALL items: max ΔCalc = {inv["max_calc_diff"].max():.4f}  (≈0 = Calc invariant to common-mode offset)')

    # ---------- QC1 + QC2 flag rates (maha N=500 P99.5) ----------
    print('\n=== QC1/QC2: flag rate on clean & benign synthetic (should ≈ budget; benign≈clean) ===')
    fr = []
    for it in PANEL:
        b = key[(it, 'maha', 500)]['budgets']['99.5']
        fr.append(dict(item=it, fp_clean=b['fp_clean']['mean'], fp_benign=b['fp_benign']['mean']))
        print(f'  {it:5}: fp_clean={b["fp_clean"]["mean"]*100:5.2f}%  fp_benign={b["fp_benign"]["mean"]*100:5.2f}%')
    fr = pd.DataFrame(fr)
    print(f'  -> mean fp_clean={fr["fp_clean"].mean()*100:.2f}%  mean fp_benign={fr["fp_benign"].mean()*100:.2f}%  (budget 0.5%)')

    # ---------- QC3: severity comparability (achieved_dev_sigma ≈ k) ----------
    print('\n=== QC3: achieved deviation ≈ k·σ̄ across items (severity comparability) ===')
    sev_rows = []
    for t in POS:
        for k in [1, 2, 3, 5, 8]:
            sub = syn[(syn['anomaly_type'] == t) & (syn['severity'] == k) & (syn['item'].isin([SYNC.get(i, i) for i in PANEL]))]
            if len(sub):
                sev_rows.append(dict(anomaly_type=t, k=k, achieved_med=sub['achieved_dev_sigma'].median(),
                                     achieved_iqr=(sub['achieved_dev_sigma'].quantile(.75) - sub['achieved_dev_sigma'].quantile(.25))))
    sev = pd.DataFrame(sev_rows)
    for t in POS:
        g = sev[sev['anomaly_type'] == t]
        print(f'  {t:11}: ' + '  '.join(f'k{int(r.k)}→{r.achieved_med:.1f}σ' for r in g.itertuples()))

    pd.concat([inv.assign(qc='invariance'), fr.assign(qc='flagrate')], ignore_index=True).to_csv(
        os.path.join(OUT, 'synthetic_validity_qc.csv'), index=False)
    sev.to_csv(os.path.join(OUT, 'severity_scaling.csv'), index=False)

    # ================= FIGURES =================
    # figA: fp_clean vs fp_benign per item
    for gray in (True, False):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(PANEL)); w = 0.4
        ax.bar(x - w / 2, fr['fp_clean'].values * 100, w, label='clean synthetic (FP on real base curves)',
               color='0.5' if gray else '#4c72b0')
        ax.bar(x + w / 2, fr['fp_benign'].values * 100, w, label='benign_offset (common-mode shift)',
               color='0.75' if gray else '#dd8452')
        ax.axhline(0.5, color='0.2' if gray else '#d62728', ls='--', lw=1.2, label='FP budget 0.5%')
        ax.set_xticks(x); ax.set_xticklabels(PANEL, rotation=40, ha='right', fontsize=8)
        ax.set_ylabel('flag rate (%)'); ax.set_ylim(0, max(2.0, fr[['fp_clean', 'fp_benign']].values.max() * 100 * 1.2))
        ax.set_title('QC: clean ≈ real normal & benign_offset cancelled in Calc (≈ FP budget)'
                     + ('  [grayscale]' if gray else '  [color]'), fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
        fig.tight_layout(); save(fig, 'figA_clean_benign_flagrate', gray)

    # figB: severity scaling achieved_dev vs k
    for gray in (True, False):
        fig, ax = plt.subplots(figsize=(7, 5))
        marks = {'A_spike': 'o', 'B_drift': 's', 'C_hump': '^', 'E_baseline': 'D'}
        for t in POS:
            g = sev[sev['anomaly_type'] == t]
            ax.plot(g['k'], g['achieved_med'], marker=marks[t], label=t,
                    color='0.2' if gray else None, ls='-')
        ax.plot([1, 8], [1, 8], '0.6', ls=':', lw=1, label='y = k (ideal)')
        ax.set_xlabel('nominal severity k (× σ̄)'); ax.set_ylabel('achieved deviation (σ, median over items)')
        ax.set_title('QC: injected severity ≈ k·σ̄, comparable across items'
                     + ('  [grayscale]' if gray else '  [color]'), fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); save(fig, 'figB_severity_scaling', gray)

    # figC: benign_offset invariance illustration (one base curve)
    code = SYNC.get('CRE', 'CRE'); s = syn[syn['item'] == code]
    cl = s[s['anomaly_type'] == 'clean'].iloc[0]
    be = s[(s['anomaly_type'] == 'benign_offset') & (s['base_id'] == cl['base_id'])].sort_values('severity').iloc[-1]
    xp = np.arange(5, 39)
    for gray in (True, False):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
        a1.plot(xp, cl[RAM].values, '-', color='0.2' if gray else '#1f77b4', label='RAM (clean)')
        a1.plot(xp, cl[RAS].values, '--', color='0.2' if gray else '#2ca02c', label='RAS (clean)')
        a1.plot(xp, be[RAM].values, '-', color='0.55' if gray else '#ff7f0e', label='RAM (benign +A)')
        a1.plot(xp, be[RAS].values, '--', color='0.55' if gray else '#d62728', label='RAS (benign +A)')
        a1.set_title('(a) RAM & RAS: common-mode offset +A applied', fontsize=10)
        a1.set_xlabel('point'); a1.set_ylabel('absorbance'); a1.legend(fontsize=7.5); a1.grid(alpha=0.3)
        a2.plot(xp, cl[RAM].values - cl[RAS].values, '-', color='0.1' if gray else '#1f77b4', lw=2, label='Calc clean')
        a2.plot(xp, be[RAM].values - be[RAS].values, '--', color='0.6' if gray else '#d62728', lw=1.5, label='Calc benign (identical)')
        a2.set_title('(b) Calc = RAM−RAS: identical → benign not flagged', fontsize=10)
        a2.set_xlabel('point'); a2.set_ylabel('Calc'); a2.legend(fontsize=8); a2.grid(alpha=0.3)
        fig.suptitle('Design specificity: dual-wavelength Calc cancels benign common-mode offsets'
                     + ('  [grayscale]' if gray else '  [color]'), fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, 'figC_benign_invariance', gray)

    print('\nDONE ->', OUT, flush=True)


if __name__ == '__main__':
    main()
