# Shared configuration for the revision (Reviewer #2) re-analyses.
#
# RULE: existing scripts are NOT modified.  Everything here imports the reviewed pipeline
# (ExperimentalCode/Scripts/run_paper_method_comparison.py) and overrides configuration / functions
# at run time, exactly like run_method_comparison_p5_38.py, rerun_ca_clean.py, rerun_m2_clean.py and
# ForPaper/Rev01_Panel21_MethodComparison/run_panel21_new8.py did for the submitted results.
#
# Requires TensorFlow (importing the pipeline imports TensorFlow): Python 3.12, TensorFlow 2.20.
#
# The 21-item panel is scored in three loader groups, reproducing the submitted Table 1 sources
# (ForPaper/_common/panel.py::load_metrics precedence: re-based re-runs > new-8 run > main run):
#   main : 18 items, endpoint_synthetic_anomalies_p5_38.parquet, M.load_item (full history)
#   ca   : Ca,       endpoint_synthetic_anomalies_p5_38_Ca_clean.parquet, rerun_ca_clean.clean_load_item
#   m2   : ALB, IP,  endpoint_synthetic_anomalies_p5_38_m2clean.parquet,  rerun_m2_clean.clean_load_item

import os
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_SCRIPTS = os.path.abspath(os.path.join(HERE, '..'))
BASE_DIR = os.path.abspath(os.path.join(EXP_SCRIPTS, '..', '..'))
for p in (EXP_SCRIPTS, os.path.join(BASE_DIR, 'ForPaper')):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_paper_method_comparison as M        # noqa: E402  (imports TF)

# Keep references to the ORIGINAL functions before any re-run module is imported:
# importing rerun_m2_clean replaces M.load_item as a side effect.
ORIG_LOAD_ITEM = M.load_item
ORIG_APPLY_NORM = M.apply_norm

import rerun_ca_clean as _CA                    # noqa: E402  (side effects on M are reset below)
import rerun_m2_clean as _M2                    # noqa: E402
from _common import panel as P                  # noqa: E402

M.load_item = ORIG_LOAD_ITEM                    # undo the rerun_m2_clean side effect

P0 = 5
M.RAM = [f'RAM{i}' for i in range(P0, 39)]
M.RAS = [f'RAS{i}' for i in range(P0, 39)]
M.XPTS = np.arange(P0, 39)
M.SELF_CAL_BUDGET = 99.5

SYN_DIR = os.path.join(BASE_DIR, 'Data', 'BigData', 'synthetic_anomalies')
REBASED = {'Ca', 'ALB', 'IP'}
GROUPS = {
    'main': dict(items=[it for it in P.PANEL21 if it[0] not in REBASED],
                 syn=os.path.join(SYN_DIR, 'endpoint_synthetic_anomalies_p5_38.parquet'),
                 loader=ORIG_LOAD_ITEM),
    'ca': dict(items=[('Ca', 'CA')],
               syn=os.path.join(SYN_DIR, 'endpoint_synthetic_anomalies_p5_38_Ca_clean.parquet'),
               loader=_CA.clean_load_item),
    'm2': dict(items=[('ALB', 'ALB'), ('IP', 'IP')],
               syn=os.path.join(SYN_DIR, 'endpoint_synthetic_anomalies_p5_38_m2clean.parquet'),
               loader=_M2.clean_load_item),
}
assert sum(len(g['items']) for g in GROUPS.values()) == 21


def apply_norm_unclipped(a, lo, hi):
    """Same scalar min-max map as the pipeline, WITHOUT np.clip (pure affine)."""
    return (a - lo) / (hi - lo + 1e-8)


def set_clip(clip):
    # load_item / clean_load_item look up apply_norm at call time (module global / M.apply_norm),
    # so replacing the module attribute is sufficient.
    M.apply_norm = ORIG_APPLY_NORM if clip else apply_norm_unclipped


def use_group(name, out_dir, cache_tag):
    g = GROUPS[name]
    M.SYN = g['syn']
    M.load_item = g['loader']
    M.ITEMS = list(g['items'])
    M.OUT = os.path.join(out_dir, name)
    M.REPORTS = os.path.join(M.OUT, 'Reports')
    M.JSONL = os.path.join(M.REPORTS, 'comparison.jsonl')
    # local disk: .keras checkpoint rewrites fail on the Google-Drive virtual FS
    M.CACHE = os.path.join(os.environ.get('LOCALAPPDATA', tempfile.gettempdir()), f'rev_cca_{cache_tag}_{name}')
    return g


def results_dir(tag):
    return os.path.join(BASE_DIR, 'Results', tag)
