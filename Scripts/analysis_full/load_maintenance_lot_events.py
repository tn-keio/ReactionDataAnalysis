# Load & normalize the facility maintenance and reagent-lot-change logs into a single event table.
# Spec: Results/20260710_PerAnalyzer_EventAlignment/experiment_plan.md.
#   equipment CSV -> maintenance events (date x analyzer)
#   reagent   CSV -> lot-change events (date x analyzer x item)   [lot NUMBER not parsed; date = change date]
# Both files are UTF-8-BOM but the Japanese text is cp932-garbled; the fields we need (date, analyzer,
# item abbreviation) are ASCII, so we access columns BY POSITION.  Item abbreviations are mapped to the
# processed A_Name; unmatched abbreviations are reported for human review.

import os, sys
import numpy as np, pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DATA = os.path.join(BASE, 'Data')
OUT = os.path.join(BASE, 'Results', '20260710_PerAnalyzer_EventAlignment')
EQ_CSV = os.path.join(DATA, '20220601-20251231_equipment-comments.csv')
RG_CSV = os.path.join(DATA, '20220601-20251231_reagent-comments.csv')
COLS = ['id', 'date', 'analyzer', 'casetype', 'cd', 'abbr', 'comment']   # positional column names

# reagent-CSV item abbreviation -> processed A_Name (partition value).  Extend as needed after review.
ABBR_MAP = {
    'UA': 'UA', 'DB': 'DB', 'TB': 'TB', 'TP': 'TP', 'ALB': 'ALB', 'CA': 'Ca', 'IP': 'IP', 'CRE': 'CRE',
    'CU': 'Cu', 'FE': 'Fe', 'ZN': 'Zn', 'MG': 'Mg', 'LI': 'Li', 'GLU': 'GLU', 'TC': 'TC', 'TG': 'TG',
    'HDL-C': 'HDL-C', 'LDL-C': 'LDL-C', 'CRP': 'CRP', 'LRG': 'LRG', 'FER': 'FER', 'B2M': 'B2M',
    'C3': 'C3', 'C4': 'C4', 'GA': 'GA', 'GAALB': 'GAALB', 'IGG': 'IgG', 'IGA': 'IgA', 'IGM': 'IgM',
    'IGG4': 'IgG4', 'IgG4': 'IgG4', 'KL-6': 'KL-6', 'MMP-3': 'MMP-3', 'UIBC': 'UIBC', 'UN': 'UN',
    'AMY': 'AMY', 'P-AMY': 'P-AMY', 'LIP': 'LIP', 'AST': 'AST', 'ALT': 'ALT', 'ALP': 'ALP-IFw',
    'CPK': 'CK', 'CK': 'CK', 'ChE': 'ChE', 'CH-E': 'ChE', 'NH3': 'NH3', 'CH-50': 'CH50', 'RF': 'RF',
    'VCM': 'VCM', 'TEIC': 'TEIC', 'mTP': 'mTP', 'GGTP': 'GGT', 'LD': 'LD-IF', 'LDH': 'LD-IF',
    'CRTNN': 'CRE', 'ALP IFCC': 'ALP-IFw', 'LD-IFCC': 'LD-IF', 'RPR2': 'RPR', 'RPR-2': 'RPR',
    # half-width katakana item names used in the reagent log for some assays
    'ﾌｴﾘﾁﾝ': 'FER', 'ﾘﾁｳﾑ': 'Li', 'ﾊﾞﾙﾌﾟﾛ酸': 'VALP', 'ﾘﾊﾟｰｾﾞ': 'LIP',
    'ﾃｲｺﾌﾟﾗ': 'TEIC', 'ﾌｴﾉﾊﾞﾙ': 'PHNB', 'ﾌｴﾆﾄｲﾝ': 'PHNT',
}
# NOTE: U-M-ALB / U-TP are URINE assays (not the serum panel) -> intentionally left unmatched.


def read_log(path):
    df = pd.read_csv(path, encoding='utf-8-sig', header=0)
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['analyzer'] = df['analyzer'].astype(str).str.strip()
    df['abbr'] = df['abbr'].astype(str).str.strip()
    return df[df['date'].notna() & df['analyzer'].str.contains('LABOSPECT', na=False)]


def main():
    os.makedirs(OUT, exist_ok=True)
    eq = read_log(EQ_CSV); rg = read_log(RG_CSV)

    # maintenance events
    maint = eq[['date', 'analyzer']].copy()
    maint['event_type'] = 'maintenance'; maint['item_abbr'] = ''; maint['proc_name'] = ''

    # lot-change events
    lot = rg[['date', 'analyzer', 'abbr']].copy().rename(columns={'abbr': 'item_abbr'})
    lot['event_type'] = 'lot'
    lot['proc_name'] = lot['item_abbr'].map(ABBR_MAP).fillna('')

    events = pd.concat([maint, lot], ignore_index=True)[
        ['date', 'analyzer', 'event_type', 'item_abbr', 'proc_name']].sort_values(['date', 'analyzer'])
    events.to_csv(os.path.join(OUT, 'event_table.csv'), index=False)

    # abbr mapping report (matched / unmatched)
    abbr_counts = rg['abbr'].value_counts()
    mp = pd.DataFrame({'abbr': abbr_counts.index, 'n_lot_events': abbr_counts.values})
    mp['proc_name'] = mp['abbr'].map(ABBR_MAP).fillna('')
    mp['matched'] = mp['proc_name'] != ''
    mp.to_csv(os.path.join(OUT, 'item_abbr_mapping.csv'), index=False)

    print('=== EVENT TABLE ===')
    print(f'maintenance events: {len(maint)}  (date {maint.date.min().date()}..{maint.date.max().date()})')
    print(f'lot-change events : {len(lot)}   matched-to-proc: {(lot.proc_name!="").sum()} / {len(lot)}')
    print('\nmaintenance per analyzer:'); print(maint['analyzer'].value_counts().to_string())
    print('\n=== abbr mapping: UNMATCHED (top by lot count; extend ABBR_MAP if needed) ===')
    print(mp[~mp.matched].head(30).to_string(index=False))
    print(f'\nmatched abbr: {mp.matched.sum()} / {len(mp)}   unmatched lot events: {int(mp[~mp.matched].n_lot_events.sum())}')
    print('\n=== lot events for panel items (proc_name in 35-panel) - counts ===')
    panel = ['TP','ALB','TB','DB','Ca','IP','CRE','UA','Fe','Cu','Zn','GLU','CRP','GA','HDL-C','LDL-C',
             'TC','TG','FER','B2M','C3','C4','IgG','IgA','IgM','IgG4','KL-6','LRG','MMP-3','UIBC','GAALB']
    pc = lot[lot.proc_name.isin(panel)].groupby('proc_name').size().sort_values(ascending=False)
    print(pc.to_string())
    print('\n-> event_table.csv, item_abbr_mapping.csv in', OUT)


if __name__ == '__main__':
    main()
