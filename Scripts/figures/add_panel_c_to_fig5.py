import os
from PIL import Image, ImageDraw, ImageFont

def load_font(size):
    FONT_PATHS = [  # first existing path wins; falls back to a default PIL font
        r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\Arial.ttf",      # Windows
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",                 # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",              # Linux
    ]
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def recompose(tag):
    import os

    # --- Repository-relative path resolution -------------------------------------
    # Figure output goes to PAPER_FIGURES_DIR when set, otherwise to the repository
    # Figures/ directory.
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    base_dir = os.environ.get('PAPER_FIGURES_DIR', os.path.join(_REPO_ROOT, 'Figures'))
    import os

    # --- Repository-relative path resolution -------------------------------------
    # The analysis root is taken from the ABDETECT_BASE environment variable and
    # falls back to the repository root (two levels above this file). Set it to the
    # directory that holds Data/, Results/ and ForPaper/ when running on the full
    # dataset:  set ABDETECT_BASE=D:\path\to\analysis   (Windows)
    #           export ABDETECT_BASE=/path/to/analysis    (POSIX)
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    SRC = os.environ.get('ABDETECT_BASE', _REPO_ROOT)
    SRC = os.path.join(SRC, r'ForPaper')
    RA = os.path.join(SRC, "Suppl09_RealAnomalyValidation", "Figures")
    
    a_path = os.path.join(RA, f"figA_curve_{tag}.png")
    b_path = os.path.join(RA, f"figB_distance_{tag}.png")
    c_path = os.path.join(base_dir, "Supp_Fig4_Aggregates.png")
    
    out_path = os.path.join(base_dir, f"fig5_real_anomaly_{tag}.png")
    
    a = Image.open(a_path).convert("RGB")
    b = Image.open(b_path).convert("RGB")
    c = Image.open(c_path).convert("RGB")
    
    if tag == "gray":
        c = c.convert("L").convert("RGB")
        
    MARGIN = 60
    LABEL_W = 150
    GAP = 90
    LABEL_SZE = 120
    
    content_w = max(a.width, b.width, c.width)
    W = LABEL_W + content_w + MARGIN
    H = MARGIN + a.height + GAP + b.height + GAP + c.height + MARGIN
    
    canvas = Image.new("RGB", (W, H), "white")
    
    ax = LABEL_W + (content_w - a.width) // 2
    ay = MARGIN
    bx = LABEL_W + (content_w - b.width) // 2
    by = MARGIN + a.height + GAP
    cx = LABEL_W + (content_w - c.width) // 2
    cy = by + b.height + GAP
    
    canvas.paste(a, (ax, ay))
    canvas.paste(b, (bx, by))
    canvas.paste(c, (cx, cy))
    
    draw = ImageDraw.Draw(canvas)
    font = load_font(LABEL_SZE)
    draw.text((30, ay + 20), "(a)", fill="black", font=font)
    draw.text((30, by + 20), "(b)", fill="black", font=font)
    draw.text((30, cy + 20), "(c)", fill="black", font=font)
    
    canvas.save(out_path, dpi=(600, 600))
    print(f"Saved {out_path} size {canvas.size}")

if __name__ == "__main__":
    recompose("color")
    recompose("gray")
