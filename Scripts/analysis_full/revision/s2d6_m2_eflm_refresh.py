# Stage 2d-6: the clinical-limit analysis re-run with the current EFLM Biological Variation Database estimates.
#   On 2026-10-22 the EFLM database showed meta-analysis estimates for Ca, CRE and Cu that differ from those
#   read through the API on 2026-09-17 (IgM unchanged), so the whole panel is re-read through the same API.
#   (1) parse: the API response (api/meta_calculations, saved as eflm_api_meta_calculations_20261022.json) is turned
#       into the two tables of 2026-09-17 by the same rules: the analyte names of eflm_desirable_bias_panel21.csv,
#       matrix Serum/plasma, desirable bias = 0.25 (CVI^2 + CVG^2)^1/2 rounded to 2 decimals, desirable imprecision
#       = 0.5 CVI, desirable TEa = 1.65 x desirable imprecision + desirable bias.  The parser is checked first by
#       rebuilding the 2026-09-17 tables from the 2026-09-17 response.
#   (2) s2d2 with its EFLM input redirected; k50 does not depend on the limit and is checked unchanged.
#   (3) s2d4 summarize on copies of the flags of 20261021_Rev_M2_AffectedRecall (no re-scoring, so the compact-AE
#       values stay those of the 2026-10-21 run; TensorFlow training is not bit-reproducible here).
#   (4) s2d5 tables.
#   (5) old vs new: limits, k* and recall per item and anomaly type, and the supplement tables.
# s2d2/s2d4/s2d5 are imported unchanged; only their module globals are redirected.
# Nothing in the earlier result folders is modified (their EFLM-based files are read from SUP, see below).
# Output: Results/20261022_Rev_M2_EFLMRefresh/
# py3.14 (no TensorFlow).
import os
import sys
import json
import glob
import shutil
import difflib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(BASE, 'Results')
# results from the 2026-09-17 estimates were moved (2026-10-22) into SUP of each folder, so that nothing reads them
# by mistake; the files left in the parent folders (k50, flags) do not depend on EFLM and are still used here.
SUP = '_superseded_20261022_EFLM'
OLD_BAT = os.path.join(RES, '20260917_Rev_M2_BatteryV2', SUP)
OLD_JSON = os.path.join(OLD_BAT, 'eflm_api_meta_calculations_20260917.json')
OUT = os.path.join(RES, '20261022_Rev_M2_EFLMRefresh')
NEW_JSON = os.path.join(OUT, 'eflm_api_meta_calculations_20261022.json')
OLD_YS = os.path.join(RES, '20261020_Rev_M2_Yardsticks')
OLD_YS_SUP = os.path.join(OLD_YS, SUP)
OLD_AFF = os.path.join(RES, '20261021_Rev_M2_AffectedRecall')
OLD_AFF_SUP = os.path.join(OLD_AFF, SUP)
OLD_TAB = os.path.join(RES, '20261021_Rev_M2_SupplementTables', SUP, 'm2_supplement_tables.md')
MATRIX = 'Serum/plasma'


def parse(path, names):
    """item -> CVI/CVG (median, CI, number of studies) for the Serum/plasma meta-analysis, as the 2026-09-17 table."""
    data = json.load(open(path, encoding='utf-8'))
    assert data['code'] == 'success'
    by_name = {}
    for d in data['data']:
        by_name.setdefault(d['analyte']['display_name'], []).append(d)
    rows = []
    for it, nm in names.items():
        if not isinstance(nm, str):                   # KL-6: not in the database
            rows.append(dict(item=it))
            continue
        assert len(by_name.get(nm, [])) == 1, (it, nm, len(by_name.get(nm, [])))
        ms = by_name[nm][0]['metas']
        r = dict(item=it, eflm_analyte=nm)
        for v in ('cvi', 'cvg'):
            m = [x for x in ms if x['var_type'] == f':{v}' and x['matrix']['matrix_expansion'] == MATRIX]
            assert len(m) == 1, (it, v, len(m))
            m = m[0]
            r.update({v: float(m['median']), f'{v}_ci': f"{m['lower']}-{m['upper']}", f'{v}_n': float(m['number_used']),
                      f'{v}_updated_at': m['updated_at']})
        rows.append(r)
    df = pd.DataFrame(rows)
    df['desirable_bias_pct'] = (0.25 * np.sqrt(df['cvi'] ** 2 + df['cvg'] ** 2)).round(2)
    df['desirable_imprecision_pct'] = 0.5 * df['cvi']
    df['TEa_pct'] = 1.65 * df['desirable_imprecision_pct'] + df['desirable_bias_pct']
    return df


BIAS_COLS = ['item', 'eflm_analyte', 'cvi', 'cvi_ci', 'cvi_n', 'cvg', 'cvg_ci', 'cvg_n', 'desirable_bias_pct']
APS_COLS = ['item', 'eflm_analyte', 'cvi', 'cvg', 'desirable_bias_pct', 'desirable_imprecision_pct', 'TEa_pct']


def roundtrip(df, cols, path):
    df[cols].to_csv(path, index=False)
    return pd.read_csv(path)


def check_parser(names):
    """The 2026-09-17 response must give back the 2026-09-17 tables."""
    old = parse(OLD_JSON, names)
    tmp = os.path.join(OUT, '_parser_check')
    os.makedirs(tmp, exist_ok=True)
    for cols, f in ((BIAS_COLS, 'eflm_desirable_bias_panel21.csv'), (APS_COLS, 'eflm_aps_panel21.csv')):
        a = roundtrip(old, cols, os.path.join(tmp, f))
        b = pd.read_csv(os.path.join(OLD_BAT, f))
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=0)
        print(f'parser check: {f} reproduced from the 2026-09-17 response', flush=True)


def compare_limits(new):
    old = pd.read_csv(os.path.join(OLD_BAT, 'eflm_desirable_bias_panel21.csv')).merge(
        pd.read_csv(os.path.join(OLD_BAT, 'eflm_aps_panel21.csv'))[['item', 'TEa_pct']], on='item')
    c = old[['item', 'cvi', 'cvg', 'cvi_n', 'cvg_n', 'desirable_bias_pct', 'TEa_pct']].merge(
        new[['item', 'cvi', 'cvi_ci', 'cvg', 'cvg_ci', 'cvi_n', 'cvg_n', 'desirable_bias_pct', 'TEa_pct',
             'cvi_updated_at', 'cvg_updated_at']], on='item', suffixes=('_old', '_new'))
    for v in ('cvi', 'cvg', 'desirable_bias_pct', 'TEa_pct'):
        c[f'{v}_diff'] = c[f'{v}_new'] - c[f'{v}_old']
    c['changed'] = c[[f'{v}_diff' for v in ('cvi', 'cvg')]].abs().gt(1e-9).any(axis=1)
    c.to_csv(os.path.join(OUT, 'eflm_old_vs_new.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(c[['item', 'cvi_old', 'cvi_new', 'cvg_old', 'cvg_new', 'desirable_bias_pct_old', 'desirable_bias_pct_new',
                 'TEa_pct_old', 'TEa_pct_new', 'changed']].round(4).to_string(index=False))
    return c


def compare_kstar(new_ys, new_aff):
    rows = []
    for sub, fn in (('desirable_bias_pct', 'clinical_severity_desirablebias.csv'), ('TEa_pct', 'clinical_severity.csv')):
        o = pd.read_csv(os.path.join(OLD_YS_SUP, sub, fn))
        n = pd.read_csv(os.path.join(new_ys, sub, fn))
        keys = ['item', 'anomaly_type', 'bias_def']
        m = o[keys + ['limit_pct', 'severity_at_TEa']].merge(n[keys + ['limit_pct', 'severity_at_TEa']], on=keys,
                                                             how='outer', suffixes=('_old', '_new'), validate='1:1')
        m.insert(0, 'yardstick', sub)
        rows.append(m)
    k = pd.concat(rows, ignore_index=True)
    same = lambda a, b: (a == b) | (a.isna() & b.isna())
    k['k_star_changed'] = ~same(k['severity_at_TEa_old'], k['severity_at_TEa_new'])
    k.to_csv(os.path.join(OUT, 'kstar_old_vs_new.csv'), index=False)
    ch = k[k['k_star_changed']]
    print(f'\n== k* changed in {len(ch)} of {len(k)} (yardstick, item, type, bias_def) cells ==')
    print(ch.to_string(index=False))
    o = pd.read_csv(os.path.join(OLD_AFF_SUP, 'affected_summary.csv'))
    n = pd.read_csv(new_aff)
    a = o.merge(n, on=['yardstick', 'anomaly_type', 'method'], suffixes=('_old', '_new'), validate='1:1')
    a.to_csv(os.path.join(OUT, 'affected_summary_old_vs_new.csv'), index=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    assert os.path.exists(NEW_JSON), 'save the API response first (curl https://biologicalvariation.eu/api/meta_calculations)'
    names = pd.read_csv(os.path.join(OLD_BAT, 'eflm_desirable_bias_panel21.csv')).set_index('item')['eflm_analyte'].to_dict()
    check_parser(names)

    # ---- (1) the current estimates ----
    new = parse(NEW_JSON, names)
    bat = os.path.join(OUT, 'eflm_inputs')
    os.makedirs(bat, exist_ok=True)
    new[BIAS_COLS].to_csv(os.path.join(bat, 'eflm_desirable_bias_panel21.csv'), index=False)
    new[APS_COLS].to_csv(os.path.join(bat, 'eflm_aps_panel21.csv'), index=False)
    new.to_csv(os.path.join(OUT, 'eflm_panel21_full.csv'), index=False)       # with CI, n and update times
    compare_limits(new)

    # ---- (2) s2d2: limits, k*, recall at k* (all curves) ----
    import s2d2_m2_k50_yardsticks as Y
    Y.BAT_SRC = bat
    Y.OUT = os.path.join(OUT, 'yardsticks')
    Y.main()
    ko = pd.read_csv(os.path.join(OLD_YS, 'k50_summary_13.csv'))
    kn = pd.read_csv(os.path.join(Y.OUT, 'k50_summary_13.csv'))
    pd.testing.assert_frame_equal(ko, kn, check_exact=False, rtol=1e-12, atol=0)
    print('k50 summary unchanged (does not depend on the limit)', flush=True)

    # ---- (3) s2d4: recall at k* over the affected curves, on the 2026-10-21 flags ----
    import s2d4_m2_affected_recall as A
    A.YS = Y.OUT
    A.OUT = os.path.join(OUT, 'affected')
    os.makedirs(A.OUT, exist_ok=True)
    fl = sorted(glob.glob(os.path.join(OLD_AFF, 'flags_*.parquet')))
    assert len(fl) == 21, len(fl)
    for f in fl:
        shutil.copyfile(f, os.path.join(A.OUT, os.path.basename(f)))
    A.summarize()

    # ---- (4) s2d5: supplement tables ----
    import s2d5_m2_supplement_tables as T
    T.YS = Y.OUT
    T.AFF = os.path.join(A.OUT, 'affected_summary.csv')
    T.OUT = os.path.join(OUT, 'tables')
    T.main()

    # ---- (5) old vs new ----
    compare_kstar(Y.OUT, T.AFF)
    a = open(OLD_TAB, encoding='utf-8').read().splitlines()
    b = open(os.path.join(T.OUT, 'm2_supplement_tables.md'), encoding='utf-8').read().splitlines()
    d = list(difflib.unified_diff(a, b, 'old (2026-10-21)', 'new (2026-10-22)', lineterm='', n=0))
    open(os.path.join(OUT, 'm2_supplement_tables_diff.txt'), 'w', encoding='utf-8').write('\n'.join(d))
    print('\n== supplement tables: old vs new ==\n' + '\n'.join(d))


if __name__ == '__main__':
    main()
