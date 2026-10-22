# Stage 3c: Fig. 4 = (a) operating-point summary + (b) computational cost, stacked at native
# resolution exactly as the submitted compositor (figures/compose_fig4_fig5.py, compose_2panel) did, from the
# simplified-style v2 panels of s3a; panel (a) is taken from s3d (legend moved above the bars, 2026-10-21) when
# that output exists.  Also copies the simplified Fig. 2 and Fig. 3 under the final names.
# Output: ForPaper/Rev_Figs2to4_v2/Final/  (colour only; the submission uses the colour versions)
import os
import shutil
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SRC = os.path.join(BASE, 'ForPaper', 'Rev_Figs2to4_v2', 'Figures_simplified')
OUT = os.path.join(BASE, 'ForPaper', 'Rev_Figs2to4_v2', 'Final')
MARGIN, LABEL_W, GAP, LABEL_SZE = 60, 150, 90, 120     # same constants as compose_fig4_fig5.py


def font(size):
    for p in (r'C:\Windows\Fonts\arialbd.ttf', r'C:\Windows\Fonts\Arial.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def compose_2panel(top, bottom, out):
    a, b = Image.open(top).convert('RGB'), Image.open(bottom).convert('RGB')
    cw = max(a.width, b.width)
    canvas = Image.new('RGB', (LABEL_W + cw + MARGIN, MARGIN + a.height + GAP + b.height + MARGIN), 'white')
    ay, by = MARGIN, MARGIN + a.height + GAP
    canvas.paste(a, (LABEL_W + (cw - a.width) // 2, ay))
    canvas.paste(b, (LABEL_W + (cw - b.width) // 2, by))
    d = ImageDraw.Draw(canvas)
    d.text((30, ay + 20), '(a)', fill='black', font=font(LABEL_SZE))
    d.text((30, by + 20), '(b)', fill='black', font=font(LABEL_SZE))
    canvas.save(out, dpi=(600, 600))
    return canvas.size


def main():
    os.makedirs(OUT, exist_ok=True)
    shutil.copyfile(os.path.join(SRC, 'fig1_recall_vs_N_color.png'), os.path.join(OUT, 'fig2_recall_vs_N_color.png'))
    shutil.copyfile(os.path.join(SRC, 'fig2_recall_k1_subtle_color.png'), os.path.join(OUT, 'fig3_recall_k1_subtle_color.png'))
    a_s3d = os.path.join(BASE, 'ForPaper', 'Rev_Figs2to4_v2', 'Figures_simplified_s3d', 'fig3_operating_summary_color.png')
    size = compose_2panel(a_s3d if os.path.exists(a_s3d) else os.path.join(SRC, 'fig3_operating_summary_color.png'),
                          os.path.join(SRC, 'fig5_compute_cost_color.png'),
                          os.path.join(OUT, 'fig4_operating_and_cost_color.png'))
    print('Fig. 4', size, '->', OUT, sorted(os.listdir(OUT)))


if __name__ == '__main__':
    main()
