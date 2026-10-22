# Stage 1z11 (S14.4-14.6, Fig. S12): per_analyzer_lot.py picks the IgG lot from the data
# ("the lot in force at the peak saline week" of each analyzer).  On analyzer #1 that rule picks lot 000408
# (switched 2025-03-21), whereas the Fig. S12 legend names 000406 as the implicated IgG lot and the raw lot
# records show 000406 on #1 from late 2024.  Re-run IgG only, unclipped (as s1g rev13), with the lot fixed to
# 000406, and compare with the data-chosen lot.  lead_days (saline alert - curve alert) does not depend on the
# lot; the lot only moves the anchor (days after the switch) and the shading of Fig. S12 (e)-(g).
# Nothing is modified: the module's EVENTS and the two s1g patches are applied in memory.
# Output: Results/20261020_Rev_IgG_Lot406/{datachosen,lot406}/per_analyzer_{verdict,weekly}.csv
# py3.14 (no TensorFlow).
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1g_burden_fixedref_noclip as G     # noqa: E402

OUT = os.path.join(G.BASE, 'Results', '20261020_Rev_IgG_Lot406')


def main():
    G.B.apply_norm = G.apply_norm_noclip
    mod = G.load_by_path('Rev13_PerAnalyzer_LotAligned', 'per_analyzer_lot.py')
    assert mod.B is G.B and mod.P is G.P
    lo, hi = mod.EVENTS['IgG'][1:]
    res = {}
    for tag, lot in (('datachosen', None), ('lot406', '000406')):
        d = os.path.join(OUT, tag)
        os.makedirs(d, exist_ok=True)
        G.P.out_dir = lambda name, d=d: d
        mod.EVENTS['IgG'] = (lot, lo, hi)
        sys.argv = ['per_analyzer_lot.py', '--items', 'IgG']
        mod.main()
        res[tag] = pd.read_csv(os.path.join(d, 'per_analyzer_verdict.csv'), dtype={'lot': str})
    cmp_ = res['datachosen'].merge(res['lot406'], on=['item', 'analyzer'], suffixes=('_datachosen', '_lot406'))
    cmp_.to_csv(os.path.join(OUT, 'compare.csv'), index=False)
    print(cmp_.T.to_string())


if __name__ == '__main__':
    main()
