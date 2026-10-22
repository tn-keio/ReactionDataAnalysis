# Stage 2i: example figures of the synthetic anomalies added in the revision, for visual checking.
#   Fig. 1  non-additive types on real base curves: W_timewarp (r 0.95 -> 0.5) and S_saturation /
#           premature termination (q 0.98 -> 0.2; q 0.5/0.2 exist only for the immunoturbidimetric assays and CRE),
#           CRE and IgG, calc channel (main - sub), points 5-38, second-reagent addition marked.
#   Fig. 2  low-severity additive types (A spike, B drift, C hump, E baseline; k 0.02 -> 0.5 sigma-bar), CRE:
#           top = curves (base and k = 0.5), bottom = perturbation divided by the per-point SD of the assay's 200 clean
#           base curves (linear axis, own range per panel): the same k is large before the 2nd reagent (tiny normal spread) and small at the
#           endpoint (spread dominated by concentration), which is why drift is the hardest type.
# Colours: severity is ordinal -> one-hue blue ramp, light (mild) to dark (severe), steps 250-700 of the
# reference sequential ramp; base curve neutral grey, dashed.  Legend always present; no colour-only encoding
# (severity is also written in the legend, with the median reaction-phase bias for that assay where defined).
# Output: Results/20260926_Rev_M2_BatteryFigures/{fig1_nonadditive.png, fig2_lowk_additive.png, examples.csv}
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SYN = os.path.join(BASE, 'Data', 'BigData', 'synthetic_anomalies')
PROF = os.path.join(BASE, 'Results', '20260917_Rev_M2_BatteryV2', 'battery_profile.csv')
OUT = os.path.join(BASE, 'Results', '20260926_Rev_M2_BatteryFigures')
PTS = np.arange(5, 39)
RAM = [f'RAM{i}' for i in PTS]; RAS = [f'RAS{i}' for i in PTS]
BLUE = {250: '#86b6ef', 300: '#6da7ec', 350: '#5598e7', 400: '#3987e5', 450: '#2a78d6', 500: '#256abf',
        550: '#1c5cab', 600: '#184f95', 650: '#104281', 700: '#0d366b'}
GREY, INK, MUTED, GRID = '#6b6b66', '#1f1f1d', '#6b6b66', '#e4e3df'


def ramp(n):
    steps = [250, 350, 450, 550, 650, 700] if n <= 6 else list(BLUE)
    return [BLUE[s] for s in np.linspace(0, len(steps) - 1, n).round().astype(int).choose(steps)] if n > 1 else [BLUE[550]]


def ramp_n(n):
    order = {1: [550], 2: [350, 600], 3: [300, 450, 650], 4: [250, 400, 550, 700], 5: [250, 350, 450, 550, 700],
             6: [250, 350, 450, 550, 650, 700]}[n]
    return [BLUE[s] for s in order]


def calc(df):
    return df[RAM].to_numpy(float) - df[RAS].to_numpy(float)


def load(fname, item, types):
    t = pq.read_table(os.path.join(SYN, fname), filters=[('item', '==', item), ('anomaly_type', 'in', types)])
    return t.to_pandas()


def style(ax, title, ylabel=True):
    ax.set_title(title, fontsize=10, color=INK, loc='left')
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel('Reaction point', fontsize=8, color=MUTED)
    if ylabel:
        ax.set_ylabel('calc = main - sub (abs. units)', fontsize=8, color=MUTED)


def pick_base(clean):
    """base curve closest to the median clean curve (a typical, not an extreme, example)"""
    X = calc(clean)
    med = np.median(X, 0)
    i = int(np.argmin(((X - med) ** 2).sum(1)))
    return clean.iloc[i]['base_id'], X[i]


def fig1(prof):
    items = [('CRE', 'CRE'), ('IGG', 'IgG')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True)
    rows = []
    for r, (it, label) in enumerate(items):
        clean = load('endpoint_synthetic_anomalies_p5_38.parquet', it, ['clean'])
        bid, xb = pick_base(clean)
        ex = load('rev2026_extra_main.parquet', it, ['W_timewarp', 'S_saturation'])
        p = prof[prof['item'] == it].iloc[0] if 'item' in prof else None
        ts = int(p['onset_idx']) + 1 if p is not None else None
        for c, (atype, name, key) in enumerate([('W_timewarp', 'Time warp (reaction slowed)', 'r'),
                                                ('S_saturation', 'Premature termination (reaction capped)', 'q')]):
            ax = axes[r, c]
            g = ex[(ex.anomaly_type == atype)]
            sev = sorted(g.severity.unique(), reverse=True)            # mild -> severe
            cols = ramp_n(len(sev))
            ax.plot(PTS, xb, color=GREY, lw=2, ls='--', label='base (normal curve)', zorder=3)
            for s, col in zip(sev, cols):
                row = g[(g.severity == s) & (g.base_id == bid)]
                if row.empty:
                    continue
                y = calc(row)[0]
                b = g[g.severity == s]['bias_react'].median() * 100
                ax.plot(PTS, y, color=col, lw=1.8, label=f'{key} = {s:g}  (bias {b:+.0f}%)', zorder=2)
                rows.append(dict(item=label, type=atype, severity=s, base_id=bid, median_bias_react_pct=round(b, 1)))
            if ts is not None:
                ax.axvline(PTS[ts] - 0.5, color=MUTED, lw=1, ls=':')
                ax.text(PTS[ts] - 0.3, ax.get_ylim()[1], ' 2nd reagent', fontsize=7.5, color=MUTED, va='top')
            style(ax, f'{label} - {name}', ylabel=(c == 0))
            ax.legend(fontsize=7.5, frameon=False, loc='lower right' if xb[-1] > xb[0] else 'upper right')
    fig.suptitle('Non-additive synthetic anomalies added in the revision (one typical base curve per assay)',
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.text(0.01, 0.005, 'Bias = median change of the reaction-phase signal (endpoint - first point after the 2nd reagent) '
             'over all base curves of that assay; an approximation of the effect on the reported result.',
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(os.path.join(OUT, 'fig1_nonadditive.png'), dpi=200, facecolor='white')
    return rows


def fig2(prof, it='CRE', label='CRE'):
    clean = load('endpoint_synthetic_anomalies_p5_38.parquet', it, ['clean'])
    bid, xb = pick_base(clean)
    sd = calc(clean).std(0)
    ex = load('rev2026_extra_main.parquet', it, ['A_spike', 'B_drift', 'C_hump', 'E_baseline'])
    types = [('A_spike', 'A spike'), ('B_drift', 'B drift'), ('C_hump', 'C hump'), ('E_baseline', 'E baseline')]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True)
    sev = sorted(ex.severity.unique())                                 # 0.02 -> 0.5 (mild -> severe)
    cols = ramp_n(len(sev))
    for c, (atype, name) in enumerate(types):
        g = ex[(ex.anomaly_type == atype) & (ex.base_id == bid)]
        ax = axes[0, c]
        ax.plot(PTS, xb, color=GREY, lw=2, ls='--', label='base', zorder=3)
        top = g[g.severity == max(sev)]
        if len(top):
            ax.plot(PTS, calc(top)[0], color=cols[-1], lw=1.8, label=f'k = {max(sev):g}', zorder=2)
        style(ax, f'{label} - {name}: curve', ylabel=(c == 0))
        ax.legend(fontsize=7.5, frameon=False)
        ax = axes[1, c]
        ax.axhspan(-1, 1, color=GRID, lw=0, label='+-1 SD of normal curves', zorder=1)
        ax.axhline(0, color=MUTED, lw=0.8)
        for s, col in zip(sev, cols):
            row = g[g.severity == s]
            if len(row):
                ax.plot(PTS, (calc(row)[0] - xb) / sd, color=col, lw=1.8, label=f'k = {s:g}', zorder=2)
        lo_, hi_ = ax.get_ylim(); ax.set_ylim(min(lo_, -1.3), max(hi_, 1.3))
        style(ax, f'{name}: change / normal SD per point', ylabel=False)
        if c == 0:
            ax.set_ylabel('(anomaly - base) / per-point SD of normal curves', fontsize=8, color=MUTED)
        ax.legend(fontsize=7.5, frameon=False, ncol=2, loc='upper left')
    fig.suptitle(f'Low-severity additive anomalies added in the revision ({label}; k in units of the median per-point SD)',
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'fig2_lowk_additive.png'), dpi=200, facecolor='white')


def main():
    os.makedirs(OUT, exist_ok=True)
    prof = pd.read_csv(PROF) if os.path.exists(PROF) else pd.DataFrame()
    rows = fig1(prof)
    fig2(prof)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'examples.csv'), index=False)
    print('saved to', OUT)


if __name__ == '__main__':
    main()
