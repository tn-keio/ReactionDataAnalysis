# -*- coding: utf-8 -*-
"""
Compose two-panel main figures by stacking existing 600-DPI panels at native
resolution (pixel-exact, no font re-rendering). Panel labels (a)/(b) drawn in a
left margin. Produces gray (main) and color (suppl) variants.

Fig. 4 = operating-point summary (a) + computational cost (b)
Fig. 5 = real Cu anomaly curve (a) + Mahalanobis distance distribution (b)
"""
import os
from PIL import Image, ImageDraw, ImageFont
REPO = os.environ.get('ABDETECT_BASE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

SRC = os.path.join(REPO, "\ForPaper")
OUT = os.path.dirname(__file__)

MC = os.path.join(SRC, "04_MethodComparison", "Figures_p5_38", "Figures")
RA = os.path.join(SRC, "Suppl09_RealAnomalyValidation", "Figures")

FONT_PATHS = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\Arial.ttf"]
def load_font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

MARGIN = 60          # white border
LABEL_W = 150        # left column for (a)/(b)
GAP = 90             # vertical gap between panels
LABEL_SZE = 120


def compose(top_path, bottom_path, out_name):
    a = Image.open(top_path).convert("RGB")
    b = Image.open(bottom_path).convert("RGB")
    content_w = max(a.width, b.width)
    W = LABEL_W + content_w + MARGIN
    H = MARGIN + a.height + GAP + b.height + MARGIN
    canvas = Image.new("RGB", (W, H), "white")
    # center each panel horizontally within the content column
    ax = LABEL_W + (content_w - a.width) // 2
    ay = MARGIN
    bx = LABEL_W + (content_w - b.width) // 2
    by = MARGIN + a.height + GAP
    canvas.paste(a, (ax, ay))
    canvas.paste(b, (bx, by))
    draw = ImageDraw.Draw(canvas)
    font = load_font(LABEL_SZE)
    draw.text((30, ay + 20), "(a)", fill="black", font=font)
    draw.text((30, by + 20), "(b)", fill="black", font=font)
    out = os.path.join(OUT, out_name)
    canvas.save(out, dpi=(600, 600))
    print("wrote:", os.path.basename(out), canvas.size)


for tag in ("gray", "color"):
    compose(os.path.join(MC, f"fig3_operating_summary_{tag}.png"),
            os.path.join(MC, f"fig5_compute_cost_{tag}.png"),
            f"fig4_operating_and_cost_{tag}.png")
    compose(os.path.join(RA, f"figA_curve_{tag}.png"),
            os.path.join(RA, f"figB_distance_{tag}.png"),
            f"fig5_real_anomaly_{tag}.png")
