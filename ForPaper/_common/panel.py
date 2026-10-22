# Single source of truth for item panels and result locations, shared by all Rev* revision scripts.
# Import this instead of hard-coding item lists.  py3.14 (no TF needed).
#
# Naming: two spellings exist for the same assay and BOTH are needed.
#   proc  = the A_Name value in Data/BigData/processed (mixed case: 'Ca','Fe','Cu','Zn','IgG',...)
#   synth = the 'item' value in the synthetic-anomaly parquet (upper case: 'CA','FE','CU','ZN','IGG',...)
# The comparison JSONL / list.csv are keyed by the PROC name.

import os

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
RESULTS = os.path.join(BASE, 'Results')

# ---------------------------------------------------------------- panels
# (proc, synth) pairs, ordered high- to low-volume within each block.

# Already computed in Results/20260717_MethodComparison_p5_38 (+ Ca/ALB/IP re-runs).
PANEL13 = [('TP', 'TP'), ('ALB', 'ALB'), ('TB', 'TB'), ('DB', 'DB'), ('Ca', 'CA'), ('IP', 'IP'),
           ('CRE', 'CRE'), ('UA', 'UA'), ('CRP', 'CRP'), ('Fe', 'FE'), ('Cu', 'CU'),
           ('Zn', 'ZN'), ('GLU', 'GLU')]

# NOT yet computed -- the 8 items Rev01 must add to reach the 21-item ForPaper panel.
PANEL8_NEW = [('TG', 'TG'), ('LDL-C', 'LDL-C'), ('HDL-C', 'HDL-C'), ('TC', 'TC'),
              ('IgG', 'IGG'), ('KL-6', 'KL-6'), ('IgA', 'IGA'), ('IgM', 'IGM')]

# The full ForPaper panel (parameter_mapped.csv ForPaper == 1).  GA is NOT in it
# (reagent formulation change GA -> GA-L2 on 2024-08-05).
PANEL21 = PANEL13 + PANEL8_NEW

# Robustness panel used by S4 (inter-analyzer) and S9 (maintenance/lot).  DIFFERENT SET:
# high-volume assays only, because both designs need 500 samples either side of an event/quarter.
# Overlaps PANEL13 in 9 items.  Do not conflate the two in the manuscript.
PANEL_ROBUSTNESS13 = ['TP', 'ALB', 'TB', 'DB', 'Ca', 'IP', 'CRE', 'UA', 'CRP',
                      'TC', 'HDL-C', 'LDL-C', 'TG']

METHODS = ['maha', 'maha_robust', 'pca', 'ae', 'cnn1d', 'ae2d_and', 'ae2d_ens',
           'twod_only', 'lof', 'iforest', 'ocsvm']

# Display order used in Table 1 (descending mean recall in the 13-item run).
METHODS_DISPLAY = ['maha', 'maha_robust', 'pca', 'lof', 'twod_only', 'ae2d_ens',
                   'ae', 'ae2d_and', 'ocsvm', 'iforest', 'cnn1d']

LABEL = {
    'maha': 'Mahalanobis (empirical)', 'maha_robust': 'Mahalanobis (robust)', 'pca': 'PCA',
    'lof': 'LOF', 'twod_only': '2D-CNN -> iForest', 'ae2d_ens': 'AE (+) 2D-CNN (ensemble)',
    'ae': '1D-AE', 'ae2d_and': 'AE (&) 2D-CNN (AND)', 'ocsvm': 'One-Class SVM',
    'iforest': 'Isolation Forest', 'cnn1d': '1D-CNN -> iForest',
}

POS_TYPES = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']   # scored types (k*sigma severities)
D_TYPE = 'D_kinetic'                                          # lag severities 1-5, reported separately
M_BASE = 200                                                  # base curves per item (clustering unit)

# --------------------------------------------------- result-run locations
# The main comparison run, plus the two re-based re-runs that OVERRIDE it for Ca/ALB/IP.
RUN_MAIN = os.path.join(RESULTS, '20260717_MethodComparison_p5_38')
RUN_CA = os.path.join(RESULTS, '20260718_MethodComparison_Ca_clean')
RUN_M2 = os.path.join(RESULTS, '20260718_MethodComparison_M2_clean')
# Rev01 writes here; absent until Rev01 has been run.
RUN_NEW8 = os.path.join(RESULTS, '20260726_MethodComparison_Panel21_new8')

REBASE_SRC = {'Ca': RUN_CA, 'ALB': RUN_M2, 'IP': RUN_M2}


def panel(which):
    """which in {13, 21, '13', '21'} -> list of (proc, synth)."""
    return {13: PANEL13, 21: PANEL21}[int(which)]


def proc_names(which):
    return [p for p, _ in panel(which)]


def _iter_jsonl(path):
    import json
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_metrics(which=13, require_all=True):
    """Merged comparison records keyed by (proc_item, method, N).

    Precedence: re-based re-runs > new-8 run > main run.  Raises if an item in the
    requested panel is missing entirely and require_all is set (so a 21-item request
    fails loudly before Rev01 has been run, rather than silently reporting 13 items).
    """
    store = {}
    for run in (RUN_MAIN, RUN_NEW8, RUN_CA, RUN_M2):
        for rec in _iter_jsonl(os.path.join(run, 'Reports', 'comparison.jsonl')):
            store[(rec['item'], rec['method'], rec['N'])] = rec
    if require_all:
        missing = sorted({p for p in proc_names(which)
                          if not any(k[0] == p for k in store)})
        if missing:
            raise SystemExit(
                f'Panel {which} requested but no results for: {missing}\n'
                f'Run ForPaper/Rev01_Panel21_MethodComparison first (writes {RUN_NEW8}).')
    return store


def list_csv_path(proc_item, N):
    """Per-sample 0/1 flags for one (item, N).  Honours the re-base precedence."""
    for run in (REBASE_SRC.get(proc_item), RUN_NEW8, RUN_MAIN):
        if run is None:
            continue
        p = os.path.join(run, proc_item, f'N{N}', 'list.csv')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'no list.csv for {proc_item} N={N}')


def out_dir(name):
    """Create and return Results/20260726_<name>/."""
    d = os.path.join(RESULTS, f'20260726_{name}')
    os.makedirs(d, exist_ok=True)
    return d
