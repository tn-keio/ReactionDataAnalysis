# FINAL method comparison at the CONFIRMED reaction-point window 5-38 (21-item panel).
#
# Thin config-override wrapper around run_paper_method_comparison.py so the heavy, reviewed scoring
# pipeline is reused verbatim (no 400-line copy).  It only changes:
#   * window            : points 5-38 (RAM5..RAM38 / RAS5..RAS38, XPTS)  [was 11-38]
#   * synthetic battery : endpoint_synthetic_anomalies_p5_38.parquet     [regenerated at 5-38]
#   * item panel        : the 21-item panel (Ca, ALB and IP are taken from the clean re-runs,
#                          rerun_ca_clean.py / rerun_m2_clean.py), plus GA for make_figures.py
#   * output folder     : Results/20260717_MethodComparison_p5_38        [11-38 results preserved]
#   * SELF_CAL_BUDGET   : 99.5  -> ALSO emit a self-calibrated (deployment) operating point, in
#                          ADDITION to the FAIR holdout-clean + FP-matched calibration (the primary
#                          metric for a fair method comparison).
#
# Prerequisites (run first, ONCE):
#   1) py ExperimentalCode/Scripts/generate_endpoint_synthetic_anomalies_p5_38.py   (build 5-38 battery)
#   2) the 1D-CNN decoder fix in Code/src/model_1d_cnn.py (length-aware; identical at 28 pts)
# Deep-learning methods require a TensorFlow environment (Python 3.12, TensorFlow 2.20).

import os
import sys
import tempfile
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import run_paper_method_comparison as M      # noqa: E402  (imports TF; reuse its scoring pipeline)

# ---- override configuration for the confirmed 5-38 run ----
P0 = 5
M.RAM = [f'RAM{i}' for i in range(P0, 39)]
M.RAS = [f'RAS{i}' for i in range(P0, 39)]
M.XPTS = np.arange(P0, 39)

M.ITEMS = [('TP', 'TP'), ('ALB', 'ALB'), ('TB', 'TB'), ('DB', 'DB'), ('Ca', 'CA'), ('IP', 'IP'),
           ('CRE', 'CRE'), ('UA', 'UA'), ('CRP', 'CRP'), ('Fe', 'FE'), ('Cu', 'CU'), ('Zn', 'ZN'),
           ('GLU', 'GLU'), ('TG', 'TG'), ('LDL-C', 'LDL-C'), ('HDL-C', 'HDL-C'), ('TC', 'TC'),
           ('IgG', 'IGG'), ('KL-6', 'KL-6'), ('IgA', 'IGA'), ('IgM', 'IGM'),   # 21-item panel
           ('GA', 'GA')]          # not in the 21-item panel; kept for figures/make_figures.py

M.SYN = os.path.join(M.BASE_DIR, 'Data', 'BigData', 'synthetic_anomalies',
                     'endpoint_synthetic_anomalies_p5_38.parquet')
M.OUT = os.path.join(M.BASE_DIR, 'Results', '20260717_MethodComparison_p5_38')
M.REPORTS = os.path.join(M.OUT, 'Reports')
# CACHE on LOCAL disk (NOT the Google-Drive H: path): the per-epoch _ae.keras/_cnn.keras checkpoint
# re-writes intermittently fail on the synced virtual FS with "OSError: [Errno 22] Invalid argument".
# CACHE only holds volatile checkpoints + the ResNet-feature .npy cache; resume state (JSONL, list.csv)
# lives under OUT and is unaffected.  A persistent LOCAL dir keeps feature-cache resume benefit.
M.CACHE = os.path.join(os.environ.get('LOCALAPPDATA', tempfile.gettempdir()),
                       'mc_p5_38_cache')          # local, persistent across relaunches
M.JSONL = os.path.join(M.REPORTS, 'comparison.jsonl')
M.SELF_CAL_BUDGET = 99.5

if __name__ == '__main__':
    assert os.path.exists(M.SYN), f'Missing 5-38 synthetic battery: {M.SYN} (run generate_..._p5_38.py first)'
    print(f'Method comparison @ points {P0}-38 | items={len(M.ITEMS)} | out={M.OUT} | self_cal={M.SELF_CAL_BUDGET}',
          flush=True)
    M.run(M.ITEMS, M.METHODS_OUT, do_list=True)
