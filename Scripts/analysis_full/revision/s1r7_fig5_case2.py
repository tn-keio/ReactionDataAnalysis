# Stage 1r-7 (Reviewer #2 D1, option B): Fig. 5(a)(b) redrawn for the second Cu case (M-protein; initial -5,
# 3-fold repeat 93, 5-fold repeat 99 reported), with the protocol of s1r5 (unclipped normalization, the 500 most
# recent pre-event curves of the analyzer that measured the curve, P99.5 self-calibrated threshold, window 5-38;
# test set = same analyzer, case date .. +70 days, case rows excluded).
#   (a) each measurement's curve against the training mean +/- 2 SD (points 1-38; the excluded points 1-4 shaded)
#   (b) d^2 of the contemporaneous curves of the same analyzer, the P99.5 threshold and the measurement
#   Fig. 5 = (a) + (b) + (c) photograph (figures/Supp_Fig4_Aggregates.png; it is of this specimen).
# The figure carries no sample identifier or date.  Existing scripts are imported unchanged.
# Output: ForPaper/Rev_Fig5_Case2/Figures/ (panels and the composed figure, gray and color)
#         Results/20261020_Rev_Fig5_Case2/fig5_values.csv (the plotted numbers, checked against s1r5)
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, '..'))):
    if p not in sys.path:
        sys.path.insert(0, p)
import real_anomaly_cu_validation as C     # noqa: E402
import s1e_cu_case_noclip as E             # noqa: E402
import s1r5_cu_case2_aligned as R5         # noqa: E402
PHOTO = os.path.join(C.BASE, 'Figures', 'Supp_Fig4_Aggregates.png')      # specimen photograph (not distributed)
FIGDIR = os.path.join(C.BASE, 'ForPaper', 'Rev_Fig5_Case2', 'Figures')
OUT = os.path.join(C.BASE, 'Results', '20261020_Rev_Fig5_Case2')
TITLE = {'initial': 'Initial measurement (analyzer #3)\n−5 µg/dL',
         'repeat1_3x': 'Repeat, 3-fold dilution (analyzer #2)\n93 µg/dL',
         'repeat2_5x_reported': 'Repeat, 5-fold dilution (analyzer #2)\n99 µg/dL, reported'}


def compute():
    sid = str(pd.read_csv(R5.CASE_ROWS, dtype=str)['S_ID'].iloc[0]).strip()
    raw = R5.raw_case_rows(sid)
    raw['sys'] = raw['SystemName'].str[-2:]
    raw['v'] = pd.to_numeric(raw['Value'], errors='coerce')
    raw = raw.sort_values('M_Date').reset_index(drop=True)
    assert len(raw) == 3 and set(raw['v']) == set(R5.LABEL), f'expected the 3 case measurements, got {len(raw)}'
    rawc = C.calc38(raw)
    df = pd.read_parquet(C.PROC, columns=['SystemName', 'S_ID', 'S_Date'] + C.RAS + C.RAM,
                         filters=[('A_Name', '==', 'Cu')])
    df['d'] = pd.to_datetime(df['S_Date'])
    df['sys'] = df['SystemName'].str.replace('LABOSPECT 008 ALPHA ', '', regex=False)
    rest = df[df['S_ID'].astype(str).str.strip() != sid]
    DAY = R5.case_day()
    pre = rest[rest['d'] < DAY]
    cols = C.window_cols(5, 38)
    out = []
    for i, r in raw.iterrows():
        an = r['sys']
        tc = C.calc38(pre[pre['sys'] == an].sort_values('d').iloc[-C.N_TRAIN:])
        test = rest[(rest['sys'] == an) & (rest['d'] >= DAY) & (rest['d'] <= DAY + pd.Timedelta(days=70))]
        m = E.maha_fit_noclip(tc[:, cols])
        d_case = float(C.dist(m, rawc[i][None, cols])[0])
        d_test = C.dist(m, C.calc38(test)[:, cols])
        out.append(dict(measurement=R5.LABEL[float(r['v'])], analyzer=an, value=float(r['v']), curve=rawc[i],
                        tr_mu=tc.mean(0), tr_sd=tc.std(0), thr=float(m['thr']), d=d_case, d_test=d_test,
                        ratio=d_case / float(m['thr']), rank=int((d_test > d_case).sum() + 1), n_test=len(d_test)))
    return out


def save(fig, name, gray):
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, f'{name}_{"gray" if gray else "color"}.png')
    fig.savefig(p, dpi=600)
    plt.close(fig)
    if gray:
        Image.open(p).convert('L').save(p, dpi=(600, 600))
    return p


def draw(res, gray):
    c_case = '0.0' if gray else '#D55E00'
    c_band = '0.8' if gray else '#9ecae1'
    x = np.arange(1, 39)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, r in zip(axes, res):
        ax.axvspan(0.5, 4.5, color='0.92', zorder=0)
        ax.fill_between(x, r['tr_mu'] - 2 * r['tr_sd'], r['tr_mu'] + 2 * r['tr_sd'], color=c_band, alpha=0.6,
                        label=f'normal ±2 SD (500 training curves, {r["analyzer"]})')
        ax.plot(x, r['tr_mu'], '0.5', ls='--', lw=0.8, label='normal mean')
        ax.plot(x, r['curve'], color=c_case, lw=1.8, marker='o', ms=3, label='this measurement')
        ax.set_title(TITLE[r['measurement']], fontsize=11)
        ax.set_xlabel('photometric point'); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')      # per panel: the training analyzer differs (#3 / #2)
    axes[0].set_ylabel('Calc (RAM − RAS)')
    fig.tight_layout()
    pa = save(fig, 'figA_curve_case2', gray)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, r in zip(axes, res):
        ax.hist(np.log10(r['d_test'] + 1), bins=40, color='0.7' if gray else '#9ecae1',
                label=f'{r["n_test"]} curves, {r["analyzer"]}, following 70 days')
        ax.axvline(np.log10(r['thr'] + 1), color='0.3' if gray else '#222222', ls='--', lw=1.3, label='P99.5 threshold')
        ax.axvline(np.log10(r['d'] + 1), color='0.0' if gray else '#D55E00', lw=2.0,
                   label=f'this measurement: d² = {r["ratio"]:,.2f}× threshold' if r['ratio'] < 10
                   else f'this measurement: d² = {r["ratio"]:,.0f}× threshold')
        ax.set_title(f'{TITLE[r["measurement"]].splitlines()[0]}\nrank {r["rank"]} of {r["n_test"]}', fontsize=11)
        ax.set_xlabel('log10(Mahalanobis d² + 1), points 5–38'); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
    axes[0].set_ylabel('count')
    fig.tight_layout()
    pb = save(fig, 'figB_distance_case2', gray)
    return pa, pb


def compose(pa, pb, gray):
    imgs = [Image.open(p).convert('RGB') for p in (pa, pb)]
    c = Image.open(PHOTO).convert('RGB')
    if gray:
        c = c.convert('L').convert('RGB')
    w = max(i.width for i in imgs)
    c = c.resize((int(c.width * min(1.0, 0.6 * w / c.width)), int(c.height * min(1.0, 0.6 * w / c.width))))
    imgs.append(c)
    margin, lab_w, gap = 60, 150, 90
    W = lab_w + w + margin
    H = margin + sum(i.height for i in imgs) + gap * 2 + margin
    canvas = Image.new('RGB', (W, H), 'white')
    draw_ = ImageDraw.Draw(canvas)
    font = R_FONT(120)
    y = margin
    for lab, im in zip(('(a)', '(b)', '(c)'), imgs):
        canvas.paste(im, (lab_w + (w - im.width) // 2, y))
        draw_.text((30, y + 20), lab, fill='black', font=font)
        y += im.height + gap
    p = os.path.join(FIGDIR, f'fig5_real_anomaly_case2_{"gray" if gray else "color"}.png')
    canvas.save(p, dpi=(600, 600))
    return p


def R_FONT(size):
    from PIL import ImageFont
    for p in (r'C:\Windows\Fonts\arialbd.ttf', r'C:\Windows\Fonts\Arial.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    os.makedirs(OUT, exist_ok=True)
    res = compute()
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('curve', 'tr_mu', 'tr_sd', 'd_test')} for r in res]
                 ).to_csv(os.path.join(OUT, 'fig5_values.csv'), index=False)
    for r in res:
        print(f'{r["measurement"]:22s} {r["analyzer"]} {r["value"]:6.0f}  {r["ratio"]:9.2f}x  rank {r["rank"]}/{r["n_test"]}')
    for gray in (False, True):
        pa, pb = draw(res, gray)
        print('wrote', compose(pa, pb, gray))


if __name__ == '__main__':
    main()
