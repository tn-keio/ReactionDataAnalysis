# Stage 1z15 (Fig. S12): redraw the reagent-episode figure from the unclipped series.
# make_figS12.py reads the R0 (clipped) outputs. Panel (a) then disagrees with the S14.4 table, which was
# moved to the unclipped values: in 2021-08 IgA is 26.4% clipped against 47.1% unclipped, KL-6 49.3%
# against 55.9%. In panels (b)-(g) the alert weeks and leads are the same either way (curve flag rate
# differs by at most 1.75 points), but the figure is redrawn from one consistent set of inputs.
# Inputs (unclipped), the same as s1z14_tables_s14_5_6.py:
#   (a)      s1g rev13 monthly series (FixedReference_IgEvent)
#   (b)-(d)  IgM weekly series from s1g rev13 (PerAnalyzer_LotAligned)
#   (e)-(g)  IgG weekly series from s1z11 with the lot fixed to 000406
# The drawing code is make_figS12.py unchanged; only its inputs, its output folder, and the persistence
# rule are replaced. The persistence rule is the calendar-consecutive one of s1z14, so the figure and the
# tables use one definition (a missing week breaks the run; the alert weeks are the same as before).
# Output: Results/20261025_Rev_FigS12_noclip/Figures/figS12_reagent_episodes_{color,gray}.png
# py3.14 (pandas + matplotlib).
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(BASE, 'Results')
OUT = os.path.join(RES, '20261025_Rev_FigS12_noclip')
sys.path.insert(0, os.path.join(BASE, 'ForPaper', 'Rev16_FigureS12_ReagentEpisodes'))
sys.path.insert(0, HERE)

import make_figS12 as F                       # noqa: E402
from s1z14_tables_s14_5_6 import sustained    # noqa: E402

NOCLIP = os.path.join(RES, '20260917_Rev_M1_S11S14_noclip')
WEEKLY = {'IgM': os.path.join(NOCLIP, 'PerAnalyzer_LotAligned', 'per_analyzer_weekly.csv'),
          'IgG': os.path.join(RES, '20261020_Rev_IgG_Lot406', 'lot406', 'per_analyzer_weekly.csv')}
MONTHLY = os.path.join(NOCLIP, 'FixedReference_IgEvent', 'fixed_reference_monthly.csv')


def load():
    w = pd.concat([pd.read_csv(p).query('item == @it') for it, p in WEEKLY.items()], ignore_index=True)
    w['week'] = pd.to_datetime(w['week'])
    w['lot_in_force'] = w['lot_in_force'].fillna('-').astype(str)
    m = pd.read_csv(MONTHLY)
    m['mo'] = pd.to_datetime(m['month'] + '-01')
    return w, m


def main():
    F.load = load
    F.sustained = sustained
    F.SCRIPT_DIR = OUT                         # make() writes to SCRIPT_DIR/Figures
    os.makedirs(OUT, exist_ok=True)
    for g in (True, False):
        F.make(g)


if __name__ == '__main__':
    main()
