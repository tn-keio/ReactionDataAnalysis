# Stage 1g: analyses built on
# ForPaper/Suppl10_ReviewBurden/measure_review_burden_unified.py (module "B"), re-run WITHOUT np.clip:
#   rev06 : Table S3 review burden, 21-item panel (ForPaper/Rev06_ReviewBurden_Panel21)   -> Section 4.4, 92.3/day
#   rev11 : fixed-reference IgG/IgM monthly + weekly onset (ForPaper/Rev11_FixedReference_IgEvent) -> Section 3.12
#   rev13 : per-analyzer lot-aligned curve vs saline blank (ForPaper/Rev13_PerAnalyzer_LotAligned) -> S14.4-14.6
#   rev14 : lot-stratified daily analysis (ForPaper/Rev14_LotStratified)
# Nothing is modified: B.apply_norm (the only clip) is replaced, and _common.panel.out_dir (the only
# output-folder factory these scripts use; their INPUTS are read from fixed Results/20260726_* paths)
# is redirected to Results/20260917_Rev_M1_S11S14_noclip/<name>, so submitted outputs are untouched.
# py3.14 (no TensorFlow).

import os
import sys
import argparse
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
FORPAPER = os.path.join(BASE, 'ForPaper')
for p in (os.path.join(FORPAPER, 'Suppl10_ReviewBurden'), FORPAPER):
    if p not in sys.path:
        sys.path.insert(0, p)
import measure_review_burden_unified as B     # noqa: E402
from _common import panel as P                # noqa: E402

OUT = os.path.join(BASE, 'Results', '20260917_Rev_M1_S11S14_noclip')

SCRIPTS = {
    'rev06': [('Rev06_ReviewBurden_Panel21', 'measure_burden_panel21.py', ['--panel', '21'])],
    'rev11': [('Rev11_FixedReference_IgEvent', 'probe_fixed_reference.py', []),
              ('Rev11_FixedReference_IgEvent', 'probe_onset_weekly.py', [])],
    'rev13': [('Rev13_PerAnalyzer_LotAligned', 'per_analyzer_lot.py', [])],
    'rev14': [('Rev14_LotStratified', 'analyze_lot_stratified.py', [])],
}


def apply_norm_noclip(a, lo, hi):
    return (a - lo) / (hi - lo + 1e-8)


def out_dir_redirect(name):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    return d


def load_by_path(folder, fname):
    path = os.path.join(FORPAPER, folder, fname)
    spec = importlib.util.spec_from_file_location(f'rev_{folder}_{fname[:-3]}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', nargs='+', default=list(SCRIPTS))
    a = ap.parse_args()
    B.apply_norm = apply_norm_noclip
    P.out_dir = out_dir_redirect
    for key in a.which:
        for folder, fname, argv in SCRIPTS[key]:
            mod = load_by_path(folder, fname)
            assert mod.B is B and mod.P is P, 'module identity mismatch: patches would not apply'
            print(f'===== {key}: {folder}/{fname} (noclip) =====', flush=True)
            sys.argv = [fname] + argv
            mod.main()


if __name__ == '__main__':
    main()
