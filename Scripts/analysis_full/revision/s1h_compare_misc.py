# Stage 1h: comparisons for the M1 scope extension and the R2 summary.
#   S9 : Results/20260718_EventRobustness  vs 20260917_Rev_M1_S9S4_noclip/event
#   S4 : Results/20260718_InterAnalyzer    vs 20260917_Rev_M1_S9S4_noclip/inter
#   Cu : Results/20260917_Rev_M1_CuCase/{clip,noclip}
#   R2 : Results/20260917_Rev_R2_RollingContemporaneous -> panel tables
# Writes CSV/markdown next to each new result.  py3.14.

import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RES = os.path.join(BASE, 'Results')
NEW = os.path.join(RES, '20260917_Rev_M1_S9S4_noclip')


def s9():
    a = pd.read_csv(os.path.join(RES, '20260718_EventRobustness', 'event_robustness_results.csv'))
    b = pd.read_csv(os.path.join(NEW, 'event', 'event_robustness_results.csv'))
    key = ['analyzer', 'item', 'event_type', 'event_date', 'idx']
    # same code path, same seed and write order -> align by position (keys repeat for same-day events)
    assert len(a) == len(b) and (a[key].astype(str).values == b[key].astype(str).values).all()
    m = a[key].copy()
    for c in ('post_rate', 'pre_rate'):
        m[c + '_clip'] = a[c].values; m[c + '_noclip'] = b[c].values
    m['d_post'] = m.post_rate_noclip - m.post_rate_clip
    rows = []
    for et, g in m.groupby('event_type'):
        rows.append(dict(event_type=et, n=len(g),
                         surge5_clip=(g.post_rate_clip >= 0.05).mean(), surge5_noclip=(g.post_rate_noclip >= 0.05).mean(),
                         surge2_clip=(g.post_rate_clip >= 0.02).mean(), surge2_noclip=(g.post_rate_noclip >= 0.02).mean(),
                         post_median_clip=g.post_rate_clip.median(), post_median_noclip=g.post_rate_noclip.median(),
                         post_max_clip=g.post_rate_clip.max(), post_max_noclip=g.post_rate_noclip.max(),
                         max_abs_diff=g.d_post.abs().max(), n_changed=int((g.d_post.abs() > 0).sum())))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(NEW, 'event', 'compare_clip_vs_noclip.csv'), index=False)
    return t, m


def s4():
    a = pd.read_csv(os.path.join(RES, '20260718_InterAnalyzer', 'inter_analyzer_results.csv'))
    b = pd.read_csv(os.path.join(NEW, 'inter', 'inter_analyzer_results.csv'))
    key = ['item', 'window', 'train_src', 'test_analyzer', 'kind']
    assert len(a) == len(b) and (a[key].astype(str).values == b[key].astype(str).values).all()
    m = a[key].copy()
    for c in ('penalty', 'flag_rate'):
        m[c + '_clip'] = a[c].values; m[c + '_noclip'] = b[c].values
    rows = []
    for k, g in m.groupby('kind'):
        rows.append(dict(kind=k, n=len(g),
                         penalty_p95_clip=g.penalty_clip.quantile(.95), penalty_p95_noclip=g.penalty_noclip.quantile(.95),
                         flag_max_clip=g.flag_rate_clip.max(), flag_max_noclip=g.flag_rate_noclip.max(),
                         flag_median_clip=g.flag_rate_clip.median(), flag_median_noclip=g.flag_rate_noclip.median(),
                         max_abs_diff=(g.flag_rate_noclip - g.flag_rate_clip).abs().max()))
    cross = m[m.kind == 'cross']
    per_item = cross.groupby('item')[['flag_rate_clip', 'flag_rate_noclip']].agg(['median', 'max'])
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(NEW, 'inter', 'compare_clip_vs_noclip.csv'), index=False)
    per_item.to_csv(os.path.join(NEW, 'inter', 'cross_by_item_clip_vs_noclip.csv'))
    return t, per_item


def cu():
    d = os.path.join(RES, '20260917_Rev_M1_CuCase')
    out = []
    for tag in ('clip', 'noclip'):
        j = json.load(open(os.path.join(d, tag, 'detection_result.json'), encoding='utf-8'))
        for r in j['results']:
            out.append(dict(mode=tag, **{k: r[k] for k in ('variant', 'window', 'd_anomaly', 'threshold_P995',
                                                           'ratio_d_over_thr', 'rank_in_test')}))
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(d, 'compare_clip_vs_noclip.csv'), index=False)
    return t


def r2():
    d = os.path.join(RES, '20260917_Rev_R2_RollingContemporaneous')
    s = pd.read_csv(os.path.join(d, 'series.csv'))
    rows = []
    for (clip, m), g in s.groupby(['clip', 'method']):
        per = g.groupby('item').agg(
            flag_med=('routine_flag_rate', 'median'), flag_p90=('routine_flag_rate', lambda v: v.quantile(.9)),
            flag_max=('routine_flag_rate', 'max'),
            rec_mean_u=('unscreened_recall_main', 'mean'), rec_min_u=('unscreened_recall_main', 'min'),
            rec_mean_s=('screened90_recall_main', 'mean'), rec_min_s=('screened90_recall_main', 'min'),
            k1_mean_u=('unscreened_recall_k1', 'mean'), k1_min_u=('unscreened_recall_k1', 'min'),
            k05_mean_u=('unscreened_recall_k0.5', 'mean'), k02_mean_u=('unscreened_recall_k0.2', 'mean'),
            k01_mean_u=('unscreened_recall_k0.1', 'mean'),
            cleanflag_u=('unscreened_clean_flag', 'mean'), cleanflag_s=('screened90_clean_flag', 'mean'),
            n=('month', 'size'))
        per.insert(0, 'method', m); per.insert(0, 'clip', clip)
        rows.append(per.reset_index())
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(os.path.join(d, 'per_item_summary.csv'), index=False)
    panel = t.groupby(['clip', 'method']).agg(
        items=('item', 'size'), flag_med_mean=('flag_med', 'mean'), flag_max_max=('flag_max', 'max'),
        rec_mean_u=('rec_mean_u', 'mean'), worst_item_min_month_u=('rec_min_u', 'min'),
        items_min_month_below_0875_u=('rec_min_u', lambda v: int((v < 0.875).sum())),
        rec_mean_s=('rec_mean_s', 'mean'), worst_item_min_month_s=('rec_min_s', 'min'),
        items_min_month_below_0875_s=('rec_min_s', lambda v: int((v < 0.875).sum())),
        k1_mean_u=('k1_mean_u', 'mean'), k05_mean_u=('k05_mean_u', 'mean'),
        k02_mean_u=('k02_mean_u', 'mean'), k01_mean_u=('k01_mean_u', 'mean')).reset_index()
    panel.to_csv(os.path.join(d, 'panel_summary.csv'), index=False)
    # month-level flag-rate outliers for E/noclip (to be matched to known events)
    e = s[(s['method'] == 'E') & (s['clip'] == 'noclip')]
    e[e.routine_flag_rate >= 0.05][['item', 'month', 'routine_flag_rate', 'test_span_d',
                                    'unscreened_recall_main']].sort_values(['item', 'month']).to_csv(
        os.path.join(d, 'E_noclip_months_flag_ge5pct.csv'), index=False)
    return panel, t


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    t, _ = s9(); print('== S9 ==\n', t.round(4).to_string(index=False))
    t, pi = s4(); print('== S4 ==\n', t.round(4).to_string(index=False)); print(pi.round(3).to_string())
    print('== Cu ==\n', cu().to_string(index=False))
    p, _ = r2(); print('== R2 ==\n', p.round(3).to_string(index=False))
