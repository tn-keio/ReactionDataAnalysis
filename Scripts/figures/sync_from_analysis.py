"""Copy the analysis-tree figure outputs into the paper tree, correctly filed.

Fig. 2, Fig. 3 and the figS9*/figS10 supplementary panels are produced by
ForPaper/04_MethodComparison/Figures_p5_38/make_figures.py under a different
numbering. This script applies the rename and puts colour in figures/,
supplementary panels in figures/supplementary/ and grayscale in
figures/grayscale/ -- doing it by hand is how grayscale files end up loose at
the top level.

    py -3 sync_from_analysis.py [--dry-run]
"""

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import os

# --- Repository-relative path resolution -------------------------------------
# The analysis root is taken from the ABDETECT_BASE environment variable and
# falls back to the repository root (two levels above this file). Set it to the
# directory that holds Data/, Results/ and ForPaper/ when running on the full
# dataset:  set ABDETECT_BASE=D:\path\to\analysis   (Windows)
#           export ABDETECT_BASE=/path/to/analysis    (POSIX)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
ANALYSIS = os.environ.get('ABDETECT_BASE', _REPO_ROOT)

SRC = os.path.join(ANALYSIS, "ForPaper", "04_MethodComparison", "Figures_p5_38", "Figures")
SRC_S2 = os.path.join(ANALYSIS, "ForPaper", "Suppl02_EstimatorSelection", "Figures")
SRC_S11 = os.path.join(ANALYSIS, "Results", "20260720_TemporalRobustness_ForPaper21")

# (source name, destination relative to figures/) -- {tag} is color or gray
MAIN = [
    ("fig1_recall_vs_N_{tag}.png", "fig2_recall_vs_N_{tag}.png"),
    ("fig2_recall_k1_subtle_{tag}.png", "fig3_recall_k1_subtle_{tag}.png"),
]

# (source dir, source name, destination name) -- all land in supplementary/
SUPPL = [
    (SRC, "figS1_recall_vs_N_byitem_color.png", "figS9a_recall_vs_N_byitem.png"),
    (SRC, "figS2_recall_k1_byitem_color.png", "figS9b_recall_k1_byitem.png"),
    (SRC, "fig4_recall_by_type_color.png", "figS10_recall_by_type.png"),
    (SRC_S2, "estimator_recall_by_type_color.png", "figS2b_estimator_recall_by_type.png"),
    (SRC_S2, "estimator_severity_worstitem_color.png", "figS2c_estimator_severity_worstitem.png"),
    (SRC_S2, "estimator_spec_vs_bdrift_color.png", "figS2d_estimator_spec_vs_bdrift.png"),
    (SRC_S11, "triage_worst12.png", "figS11b_triage_worst12.png"),
    (SRC_S11, "overview_worstmonth.png", "figS11c_overview_worstmonth.png"),
]


def copy(src, dst, dry):
    if not os.path.exists(src):
        print(f"  MISSING {os.path.basename(src)}")
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    rel = os.path.relpath(dst, HERE)
    if dry:
        print(f"  would copy -> {rel}")
    else:
        shutil.copy2(src, dst)
        print(f"  {rel}")
    return True


def main(dry=False):
    if not os.path.isdir(SRC):
        sys.exit(f"analysis output not found: {SRC}")

    n = 0
    for tag in ("color", "gray"):
        # colour is the submitted version; grayscale is the print fallback
        out_dir = HERE if tag == "color" else os.path.join(HERE, "grayscale")
        for s, d in MAIN:
            n += copy(os.path.join(SRC, s.format(tag=tag)),
                      os.path.join(out_dir, d.format(tag=tag)), dry)

    for src_dir, s, d in SUPPL:
        n += copy(os.path.join(src_dir, s), os.path.join(HERE, "supplementary", d), dry)

    print(f"{n} file(s) {'to copy' if dry else 'copied'}")
    if not dry:
        print("next: py -3 flatten_rgba.py")


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
