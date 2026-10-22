# Stage 1z5 (Reviewer #2, M2/M3 bookkeeping): re-run the v2 outputs that read the contaminated
# compact-AE selection (dae_z_selection.csv; see s1z4_compact_selection_fix.py) with the corrected one.
#   s2d_clinical_severity : k50_by_item.csv, clinical_severity{,_desirablebias}.csv, clinical_summary{...}.csv
#   s2f_stats_v2 --family all : the 15-method sensitivity analysis (its _allvariants files are not on disk)
#   s2f --family all with the OLD selection as well, to see whether the cited 15-method values used it
# Existing scripts are imported unchanged; only their output folder (module global V2) and the selection
# table are redirected.  The JSONL inputs are still read from Results/20260917_Rev_v2Benchmark.
# Output: Results/20261013_Rev_M3_CompactSelFix/{downstream_fixed, downstream_oldsel}/
# py3.14 (no TensorFlow).

import os
import sys
import json
import shutil
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2d_clinical_severity as D     # noqa: E402
import s2f_stats_v2 as F              # noqa: E402

V2 = D.V2                              # original v2 benchmark folder (JSONL inputs)
FIX = os.path.join(D.RES, '20261013_Rev_M3_CompactSelFix')
SEL = {'downstream_fixed': os.path.join(FIX, 'dae_z_selection_fixed.csv'),
       'downstream_oldsel': os.path.join(V2, 'dae_z_selection.csv')}


def jsonl_recs():
    out = {}
    for g in ('main', 'ca', 'm2'):
        for l in open(os.path.join(V2, g, 'Reports', 'comparison.jsonl'), encoding='utf-8'):
            if l.strip():
                d = json.loads(l)
                out[(d['item'], d['method'], d['N'])] = d
    return out


def f_load():
    """s2f_stats_v2.load() with the JSONL from the original folder and the selection from F.V2."""
    recs = jsonl_recs()
    zsel = pd.read_csv(os.path.join(F.V2, 'dae_z_selection.csv'))
    for _, r in zsel[zsel.N == F.N].iterrows():
        for tag, col in (('dae_elbow', 'z_elbow'), ('dae_oracle', 'z_oracle')):
            src = recs.get((r['item'], f"dae{int(r[col])}", F.N))
            if src is not None:
                recs[(r['item'], tag, F.N)] = src
    return recs


def main():
    D.load_recs = jsonl_recs
    F.load = f_load
    for tag, sel in SEL.items():
        out = os.path.join(FIX, tag)
        os.makedirs(out, exist_ok=True)
        shutil.copyfile(sel, os.path.join(out, 'dae_z_selection.csv'))
        D.V2 = F.V2 = out
        if tag == 'downstream_fixed':
            for ys in ('TEa_pct', 'desirable_bias_pct'):
                print(f'\n######## s2d, selection={tag}, yardstick={ys}', flush=True)
                D.YARDSTICK = ys
                D.main()
        print(f'\n######## s2f --family all, selection={tag}', flush=True)
        sys.argv = ['s2f_stats_v2.py', '--family', 'all']
        F.main()


if __name__ == '__main__':
    main()
