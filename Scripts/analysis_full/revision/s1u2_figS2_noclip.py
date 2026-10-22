# Stage 1u-2: redraw Fig. S2(b)-(d) (ForPaper/Suppl02_EstimatorSelection/make_estimator_selection_figure.py,
# unchanged) from the unclipped S3.4 rolling run (s1u_estimator_rolling_noclip.py --clip off).
# Only the module-globals RES and FIG are replaced; the submitted ForPaper figures are NOT overwritten.
# Output: Results/20261012_Rev_M1_EstimatorRolling/noclip/Figures/
# py3.14 (no TensorFlow).

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'ForPaper', 'Suppl02_EstimatorSelection'))
import make_estimator_selection_figure as F     # noqa: E402

F.RES = os.path.join(BASE, 'Results', '20261012_Rev_M1_EstimatorRolling', 'noclip')
F.FIG = os.path.join(F.RES, 'Figures')

if __name__ == '__main__':
    F.main()
