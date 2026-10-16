# Paper method comparison (calc channel, min-max, 3 training sizes) — computation + outputs.
# Spec: ForPaper/04_MethodComparison/{experiment_plan, implementation_notes, output-style, methods_compared}.md
#
# Per item: date-sorted Calc(=RAM-RAS). Splits (disjoint, fixed across N):
#   train_pool=[:10000], holdout(clean calib)=[10000:12000], analysis(real)=[12000:15000].
# Training sizes N in {500,5000,10000} (nested).  Normalization: one scalar min-max pair shared by
#   all reaction points, constants from the 10,000-sample training pool, FIXED across N
#   (implementation_notes §2).  No clipping: the map is purely affine, so values outside the
#   training range are kept as they are.
# Threshold: holdout-clean P{99.5,99,95} for ALL methods (fair).  Stochastic methods: seeds 0-3;
#   deterministic (maha, maha_robust, pca, lof): seed 0 only.
# Outputs:
#   Reports/comparison.jsonl                         (item x method x N x seed metrics; resume-aware)
#   <Item>/N<N>/list.csv                             (ALL eval samples, per-method 0/1, seed0 @P99.5)
#   <Item>/N<N>/<serial>.png                         (flagged reaction patterns only; filename=serial)
#   run_config.json                                  (reproducibility metadata)
# NOTE: figures are produced by plot_paper_method_comparison.py.  This script does NOT run itself.

import os, sys, gc, json, time, argparse
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from Code.src.model_autoencoder import AutoencoderModel
from Code.src.model_1d_cnn import CNN1DFeatureExtractor
from Code.src.model_compact_autoencoder import DenseAE, ConvCompactAE

# ---------------- config ----------------
# (proc A_Name, synth item).  Full 12-item robustness panel (2026-07-06 expansion).
# proc names follow processed/ partition dirs (Ca/Cu/Fe/Zn lower-case); synth uses upper-case codes.
ITEMS = [('TP', 'TP'), ('ALB', 'ALB'), ('TB', 'TB'), ('DB', 'DB'), ('Ca', 'CA'), ('IP', 'IP'),
         ('CRE', 'CRE'), ('UA', 'UA'), ('Fe', 'FE'), ('Cu', 'CU'), ('Zn', 'ZN'), ('GLU', 'GLU')]
N_GRID = [500, 5000, 10000]
SEEDS = [0, 1, 2, 3]
BUDGETS = [99.5, 99.0, 95.0]
FP_TARGETS = [0.5, 1.0, 5.0]                 # clean-FP-matched recall points (percent)
N_TRAINPOOL, N_HOLDOUT, N_ANALYSIS = 10000, 2000, 3000
POS_TYPES = ['A_spike', 'B_drift', 'C_hump', 'E_baseline']
AE_EPOCHS = CNN_EPOCHS = 50
BATCH = 32
RESNET_BATCH = 64
IMG_PX = 224
LIST_SEED = 0                                # seed used for the 0/1 list & images
LIST_BUDGET = 99.5
SELF_CAL_BUDGET = None                        # if set (e.g. 99.5): ALSO report a self-calibrated
                                             # operating point (threshold from the TRAINING scores).
                                             # None = original behaviour (holdout-clean + FP-matched only).
                                             # NOTE: self-cal floods flexible over-fitting methods (AE)
                                             # via in-sample optimism; it is the DEPLOYMENT point, kept as
                                             # a supplement — the FAIR method comparison uses holdout/FP-match.

# Compact-bottleneck autoencoders: 'dae<z>' = dense 34-16-z-16-34, 'cae<z>' = the 1D-AE topology with a
# Dense(z) code.  The reported dense AE ('dae_elbow') picks z per item and N from the clean validation
# MSE of dae1..dae16 (select_compact_ae.py); cae8 uses a fixed z = 8.
COMPACT_AE = ['dae1', 'dae2', 'dae4', 'dae8', 'dae16', 'cae8']
METHODS_OUT = ['maha', 'maha_robust', 'pca', 'ae', 'cnn1d', 'ae2d_and', 'ae2d_ens',
               'twod_only', 'lof', 'iforest', 'ocsvm'] + COMPACT_AE
COMPUTE_METHODS = ['maha', 'maha_robust', 'pca', 'ae', 'cnn1d', 'ae2d', 'twod_only',
                   'lof', 'iforest', 'ocsvm'] + COMPACT_AE    # 'ae2d' emits ae2d_and + ae2d_ens
STOCHASTIC = {'ae', 'cnn1d', 'ae2d', 'twod_only', 'iforest', 'ocsvm'} | set(COMPACT_AE)
USES_2D = {'ae2d', 'twod_only'}
RENDER_YLIM = (-0.05, 1.05)                  # 2D-CNN rendering range: normalized range + 5% margin

RAM = [f'RAM{i}' for i in range(11, 39)]
RAS = [f'RAS{i}' for i in range(11, 39)]
XPTS = np.arange(11, 39)
PROC = os.path.join(BASE_DIR, 'Data', 'BigData', 'processed')
SYN = os.path.join(BASE_DIR, 'Data', 'BigData', 'synthetic_anomalies', 'endpoint_synthetic_anomalies.parquet')
OUT = os.path.join(BASE_DIR, 'Results', '20260705_MethodComparison_Paper')   # new folder (preserve old)
REPORTS, CACHE = os.path.join(OUT, 'Reports'), os.path.join(OUT, 'Cache')
JSONL = os.path.join(REPORTS, 'comparison.jsonl')
COMPACT_LOG = 'compact_ae_selection.jsonl'   # per-fit validation MSE of the compact AEs, under REPORTS


def set_seeds(s):
    os.environ['PYTHONHASHSEED'] = str(s); np.random.seed(s); tf.random.set_seed(s)


# ---- min-max normalization (constants from training pool, fixed across N) ----
def norm_fit(a):
    return float(a.min()), float(a.max())


def apply_norm(a, lo, hi):
    return (a - lo) / (hi - lo + 1e-8)      # affine, no clipping


def mcd_like(D, keep=0.95, iters=5):
    # Robust mean/cov by iterative trimming (the concentration/C-step of FAST-MCD, single start).
    # NOT the full MCD (no random restarts); paper: "iterative-trimming robust covariance".
    loc = D.mean(0); cov = np.cov(D.T); n = int(len(D) * keep)
    for _ in range(iters):
        d = np.sum((D - loc) @ np.linalg.pinv(cov) * (D - loc), axis=1)
        core = D[np.argsort(d)[:n]]; loc, cov = core.mean(0), np.cov(core.T)
    return loc, cov


def maha_scores(loc, cov, X):
    P = np.linalg.pinv(cov)
    return np.sum((X - loc) @ P * (X - loc), axis=1)


# ---------------- data ----------------
def load_item(proc_name, synth_name):
    df = pd.read_parquet(PROC, filters=[('A_Name', '==', proc_name)], columns=['S_Date', 'S_ID'] + RAM + RAS)
    df = df.sort_values('S_Date').reset_index(drop=True).iloc[:N_TRAINPOOL + N_HOLDOUT + N_ANALYSIS]
    ram = df[RAM].values.astype('float64'); ras = df[RAS].values.astype('float64')
    calc = ram - ras
    lo, hi = norm_fit(calc[:min(N_TRAINPOOL, len(calc))])       # FIXED constants (training pool)
    Xn = apply_norm(calc, lo, hi)
    sl_tr = slice(0, N_TRAINPOOL); sl_ho = slice(N_TRAINPOOL, N_TRAINPOOL + N_HOLDOUT)
    sl_an = slice(N_TRAINPOOL + N_HOLDOUT, N_TRAINPOOL + N_HOLDOUT + N_ANALYSIS)
    s = pd.read_parquet(SYN, filters=[('item', '==', synth_name)])
    syn_ram = s[RAM].values.astype('float64'); syn_ras = s[RAS].values.astype('float64')
    syn_calc = syn_ram - syn_ras
    return dict(
        train_pool=Xn[sl_tr], holdout=Xn[sl_ho], analysis=Xn[sl_an],
        syn=apply_norm(syn_calc, lo, hi),
        meta=s[['anomaly_type', 'severity', 'base_id']].reset_index(drop=True),
        # raw curves for plotting / band
        an_ram=ram[sl_an], an_ras=ras[sl_an], an_sid=df['S_ID'].values[sl_an].astype(str),
        syn_ram=syn_ram, syn_ras=syn_ras,
        tr_calc_mean=calc[sl_tr].mean(0), tr_calc_std=calc[sl_tr].std(0),
        norm=(lo, hi))


# ---------------- ResNet features (cached; normalization fixed so reusable across N) ----------------
_RESNET = None
def resnet():
    global _RESNET
    if _RESNET is None:
        from tensorflow.keras.applications import ResNet50
        _RESNET = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return _RESNET


def render_feats(X, tag=''):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from tensorflow.keras.applications.resnet50 import preprocess_input
    n, L = X.shape; feats, buf = [], []
    fig = plt.figure(figsize=(IMG_PX / 100., IMG_PX / 100.), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_ylim(*RENDER_YLIM); ax.set_xlim(0, L - 1)
    line, = ax.plot(np.zeros(L), color='black', lw=1.5); cv = fig.canvas
    for i in range(n):
        line.set_ydata(X[i]); cv.draw()
        buf.append(np.asarray(cv.buffer_rgba())[:, :, :3].astype('float32'))
        if len(buf) == RESNET_BATCH or i == n - 1:
            feats.append(resnet().predict(preprocess_input(np.array(buf)), verbose=0)); buf = []
        if n > 3000 and (i + 1) % 3000 == 0:
            print(f'      feat[{tag}] {i + 1}/{n}', flush=True)
    plt.close(fig)
    return np.concatenate(feats, axis=0).astype('float32')


def feats_cached(item, name, X):
    p = os.path.join(CACHE, f'{item}_{name}.npy')
    if os.path.exists(p):
        f = np.load(p)
        if len(f) == len(X):
            return f
    f = render_feats(X, f'{item}:{name}'); np.save(p, f); return f


# ---------------- metrics ----------------
def metrics_at(flag_syn, meta, s_ho, thr):
    at = meta['anomaly_type'].values; sev = meta['severity'].values
    clean = at == 'clean'; benign = at == 'benign_offset'
    m = {'specificity_clean': float(1 - flag_syn[clean].mean()),
         'fp_clean': float(flag_syn[clean].mean()),
         'fp_benign': float(flag_syn[benign].mean()),
         'fp_holdout': float((s_ho >= thr).mean()),
         'recall_main': float(flag_syn[np.isin(at, POS_TYPES) & (sev >= 2)].mean())}
    for t in POS_TYPES:
        sel = at == t
        m[f'recall_{t}'] = float(flag_syn[sel & (sev >= 2)].mean())
        for k in sorted(np.unique(sev[sel])):
            m[f'recall_{t}_k{int(k)}'] = float(flag_syn[sel & (sev == k)].mean())
    sel = at == 'D_kinetic'                     # Type D, recall by lag (points)
    for k in sorted(np.unique(sev[sel])):
        m[f'recall_D_lag{k:g}'] = float(flag_syn[sel & (sev == k)].mean())
    return m


def recall_at_fpr(s_syn, meta, s_clean, fp_pct):
    """Recall_main at a threshold giving clean-FP = fp_pct (%) on the synthetic clean set."""
    thr = np.percentile(s_clean, 100 - fp_pct)
    at = meta['anomaly_type'].values; sev = meta['severity'].values
    main = np.isin(at, POS_TYPES) & (sev >= 2)
    return float((s_syn[main] >= thr).mean())


def agg_scalar(vals):
    v = np.array(vals, float)
    return {'mean': float(v.mean()),
            'ci95': float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0}


def agg(dl):
    return {k: agg_scalar([d[k] for d in dl]) for k in dl[0]}


# ---------------- per-method scoring on the fixed sets ----------------
def score_sets(method, ctx, N, seed):
    Xtr = ctx['train_pool'][:N]; Xho = ctx['holdout']; Xsyn = ctx['syn']; Xan = ctx['analysis']
    L = Xtr.shape[1]; t0 = time.time()
    if method in ('maha', 'maha_robust'):
        loc, cov = mcd_like(Xtr, 0.95) if method == 'maha_robust' else \
            (lambda e: (e.location_, e.covariance_))(EmpiricalCovariance().fit(Xtr))
        ft = time.time() - t0; f = lambda X: maha_scores(loc, cov, X)
    elif method == 'pca':
        m = PCA(n_components=0.95).fit(Xtr); ft = time.time() - t0
        f = lambda X: np.mean((X - m.inverse_transform(m.transform(X))) ** 2, axis=1)
    elif method == 'lof':
        m = LocalOutlierFactor(novelty=True).fit(Xtr); ft = time.time() - t0
        f = lambda X: -m.decision_function(X)
    elif method == 'iforest':
        m = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xtr); ft = time.time() - t0
        f = lambda X: -m.decision_function(X)
    elif method == 'ocsvm':
        m = OneClassSVM(gamma='scale').fit(Xtr); ft = time.time() - t0
        f = lambda X: -m.decision_function(X)
    elif method == 'ae':
        set_seeds(seed); X3 = Xtr.reshape(-1, L, 1); vs = max(1, int(len(X3) * 0.2))
        ae = AutoencoderModel(input_length=L)
        ae.train(X3[:-vs], X3[-vs:], epochs=AE_EPOCHS, batch_size=BATCH, save_path=os.path.join(CACHE, '_ae.keras'))
        ft = time.time() - t0
        f = lambda X: np.mean((X.reshape(-1, L, 1) - ae.model.predict(X.reshape(-1, L, 1), batch_size=512, verbose=0)) ** 2, axis=(1, 2))
    elif method == 'cnn1d':
        set_seeds(seed); X3 = Xtr.reshape(-1, L, 1); vs = max(1, int(len(X3) * 0.2))
        cnn = CNN1DFeatureExtractor(input_length=L, feature_dim=64)
        cnn.train(X3[:-vs], X3[-vs:], epochs=CNN_EPOCHS, batch_size=BATCH, save_path=os.path.join(CACHE, '_cnn.keras'))
        iff = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(cnn.extract_features(X3))
        ft = time.time() - t0
        f = lambda X: -iff.decision_function(cnn.extract_features(X.reshape(-1, L, 1)))
    elif method[:3] in ('dae', 'cae') and method[3:].isdigit():
        set_seeds(seed); vs = max(1, int(len(Xtr) * 0.2)); z = int(method[3:])
        ae = (DenseAE if method[:3] == 'dae' else ConvCompactAE)(L, z)
        info = ae.train(Xtr[:-vs], Xtr[-vs:], epochs=AE_EPOCHS, batch_size=BATCH)
        ft = time.time() - t0
        with open(os.path.join(REPORTS, COMPACT_LOG), 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(dict(item=ctx.get('item'), method=method, kind=method[:3], z=z, N=N,
                                     seed=seed, **info)) + '\n')
        f = ae.score
    elif method in USES_2D:
        ftr = ctx['feat_train'][:N]
        iff = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(ftr)
        ft = time.time() - t0
        return {'train': -iff.decision_function(ftr), 'holdout': -iff.decision_function(ctx['feat_holdout']),
                'syn': -iff.decision_function(ctx['feat_syn']), 'analysis': -iff.decision_function(ctx['feat_analysis'])}, ft, None
    else:
        raise ValueError(method)
    t1 = time.time(); s_an = f(Xan); infer_per = (time.time() - t1) / max(len(Xan), 1)
    return {'train': f(Xtr), 'holdout': f(Xho), 'syn': f(Xsyn), 'analysis': s_an}, ft, infer_per


def _z(sc, key):
    mu = sc['holdout'].mean(); sd = sc['holdout'].std() + 1e-12
    return (sc[key] - mu) / sd


# ---------------- list.csv + flagged images (seed 0, P99.5) ----------------
def write_list_and_images(proc, N, ctx, det):
    """det[method_out] = {'flag': bool[syn+analysis], 'score': float[syn+analysis]} at LIST_SEED,
    already computed with the SAME logic as the JSONL metrics (P99.5)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    outdir = os.path.join(OUT, proc, f'N{N}'); os.makedirs(outdir, exist_ok=True)
    meta = ctx['meta']; nsyn = len(meta); nan = len(ctx['an_ram'])
    flags = {mo: det[mo]['flag'] for mo in METHODS_OUT}
    scores = {mo: det[mo]['score'] for mo in METHODS_OUT}
    n_flagged = np.sum([flags[mo].astype(int) for mo in METHODS_OUT], axis=0)
    # ground-truth columns
    gt = list(meta['anomaly_type'].values) + ['real_unknown'] * nan
    sev = list(meta['severity'].values) + [''] * nan
    ids = list(meta['base_id'].values.astype(str)) + list(ctx['an_sid'])
    source = ['synthetic'] * nsyn + ['real'] * nan
    rows = {'serial': np.arange(nsyn + nan), 'source': source, 'ground_truth': gt,
            'severity_k': sev, 'id': ids}
    for mo in METHODS_OUT:
        rows[mo] = flags[mo].astype(int)
    rows['n_flagged'] = n_flagged
    for mo in METHODS_OUT:
        rows[f'score_{mo}'] = np.round(scores[mo], 4)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, 'list.csv'), index=False, encoding='utf-8')
    # raw curves for plotting
    ram = np.concatenate([ctx['syn_ram'], ctx['an_ram']]); ras = np.concatenate([ctx['syn_ras'], ctx['an_ras']])
    calc = ram - ras; mu, sd = ctx['tr_calc_mean'], ctx['tr_calc_std']
    # Images: only review-worthy flagged rows = real analysis (unknown) + synthetic clean/benign (FP cases).
    # The ~thousands of synthetic true positives (A-E, D_kinetic) are known & remain in list.csv only.
    exclude = set(POS_TYPES + ['D_kinetic'])
    img_serials = [s for s in np.where(n_flagged >= 1)[0] if gt[s] not in exclude]
    for serial in img_serials:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.fill_between(XPTS, mu - 2 * sd, mu + 2 * sd, color='0.85', label='Normal ±2SD')
        ax.plot(XPTS, mu, '0.5', ls='--', lw=.7)
        ax.plot(XPTS, calc[serial], '#d62728', lw=1.6, label='Calc')
        ax.plot(XPTS, ram[serial], '#1f77b4', lw=0.8, alpha=0.5, label='RAM')
        ax.plot(XPTS, ras[serial], '#2ca02c', lw=0.8, alpha=0.5, label='RAS')
        ax.set_title(f'#{serial}  {gt[serial]}{("/k"+str(sev[serial])) if sev[serial] != "" else ""}  '
                     f'flagged={int(n_flagged[serial])}/{len(METHODS_OUT)}', fontsize=7)
        ax.tick_params(labelsize=6); ax.legend(fontsize=5, loc='upper left')
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f'{serial}.png'), dpi=120); plt.close()
    print(f'    list+images: {proc} N={N}  flagged(any)={int((n_flagged>=1).sum())}  '
          f'images(review)={len(img_serials)}', flush=True)


# ---------------- main ----------------
def done_keys():
    d = set()
    if os.path.exists(JSONL):
        for l in open(JSONL, encoding='utf-8'):
            try:
                r = json.loads(l); d.add((r['item'], r['method'], r['N']))
            except Exception:
                pass
    return d


def run(items, methods_out, do_list):
    for dd in [REPORTS, CACHE]:
        os.makedirs(dd, exist_ok=True)
    with open(os.path.join(OUT, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'items': [i[0] for i in items], 'N_grid': N_GRID, 'seeds': SEEDS,
                   'window_points': [int(XPTS[0]), int(XPTS[-1])], 'synthetic': os.path.basename(SYN),
                   'self_cal_budget': SELF_CAL_BUDGET,
                   'budgets': BUDGETS, 'fp_targets_pct': FP_TARGETS, 'channel': 'Calc(RAM-RAS)',
                   'normalization': 'scalar min-max (one pair for all points), constants from 10k training pool '
                                    '(fixed across N), no clipping',
                   'render_ylim': list(RENDER_YLIM),
                   'splits': {'train_pool': N_TRAINPOOL, 'holdout': N_HOLDOUT, 'analysis': N_ANALYSIS},
                   'methods': METHODS_OUT, 'list_seed': LIST_SEED, 'list_budget': LIST_BUDGET,
                   'date': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2, ensure_ascii=False)
    dk = done_keys()
    compute = [m for m in COMPUTE_METHODS
               if (m == 'ae2d' and any(x in methods_out for x in ('ae2d_and', 'ae2d_ens'))) or (m in methods_out)]
    need2d = any(m in USES_2D for m in compute)
    def list_ok(p, N):
        return (not do_list) or os.path.exists(os.path.join(OUT, p, f'N{N}', 'list.csv'))

    for proc, synth in items:
        # Restart-cheap: if every method+N is already in the JSONL AND its list.csv exists,
        # skip the item entirely (no data load, no deep recompute).  Makes frequent kills cheap.
        if all((proc, mo, N) in dk for mo in METHODS_OUT for N in N_GRID) and all(list_ok(proc, N) for N in N_GRID):
            print(f'\n===== {proc}: already complete, skipping =====', flush=True)
            continue
        print(f'\n===== {proc} =====', flush=True)
        ctx = load_item(proc, synth)
        ctx['item'] = proc
        if need2d:
            for nm, arr in [('train', ctx['train_pool']), ('holdout', ctx['holdout']),
                            ('analysis', ctx['analysis']), ('syn', ctx['syn'])]:
                ctx[f'feat_{nm}'] = feats_cached(proc, nm, arr)
        meta = ctx['meta']
        for N in N_GRID:
            # Skip this N if all metrics done and (list not needed or already written) — avoids
            # recomputing deep scores just to rebuild an existing list.csv on resume.
            if all((proc, mo, N) in dk for mo in METHODS_OUT) and list_ok(proc, N):
                print(f'  {proc} N={N}: metrics+list already complete, skipping', flush=True)
                continue
            det = {}   # captured seed-0 scores for list/images
            for method in compute:
                m_seeds = SEEDS if method in STOCHASTIC else [SEEDS[0]]
                # ---- ae2d: emits ae2d_and + ae2d_ens ----
                if method == 'ae2d':
                    variants = ['ae2d_and', 'ae2d_ens']
                    if all((proc, v, N) in dk for v in variants):
                        pass
                    else:
                        pb = {v: {str(b): [] for b in BUDGETS} for v in variants}
                        fprec = {v: {str(t): [] for t in FP_TARGETS} for v in variants}
                        fts = []
                        for sd in m_seeds:
                            sc_ae, ft_ae, _ = score_sets('ae', ctx, N, sd)
                            sc_2d, ft_2d, _ = score_sets('ae2d', ctx, N, sd)
                            fts.append(ft_ae + ft_2d)
                            z_ho = np.maximum(_z(sc_ae, 'holdout'), _z(sc_2d, 'holdout'))
                            z_syn = np.maximum(_z(sc_ae, 'syn'), _z(sc_2d, 'syn'))
                            z_an = np.maximum(_z(sc_ae, 'analysis'), _z(sc_2d, 'analysis'))
                            z_cl = z_syn[meta['anomaly_type'].values == 'clean']
                            for b in BUDGETS:
                                thr_ae = np.percentile(sc_ae['holdout'], b); thr_2d = np.percentile(sc_2d['holdout'], b)
                                fa = (sc_ae['syn'] >= thr_ae) & (sc_2d['syn'] >= thr_2d)
                                m = metrics_at(fa, meta, np.zeros(1), 0.5)
                                m['fp_holdout'] = float(((sc_ae['holdout'] >= thr_ae) & (sc_2d['holdout'] >= thr_2d)).mean())
                                pb['ae2d_and'][str(b)].append(m)
                                thr_e = np.percentile(z_ho, b)
                                pb['ae2d_ens'][str(b)].append(metrics_at(z_syn >= thr_e, meta, z_ho, thr_e))
                            cl_ae = sc_ae['syn'][meta['anomaly_type'].values == 'clean']
                            cl_2d = sc_2d['syn'][meta['anomaly_type'].values == 'clean']
                            for t in FP_TARGETS:
                                ta = np.percentile(cl_ae, 100 - t); td = np.percentile(cl_2d, 100 - t)
                                at = meta['anomaly_type'].values; sv = meta['severity'].values
                                main = np.isin(at, POS_TYPES) & (sv >= 2)
                                fprec['ae2d_and'][str(t)].append(float(((sc_ae['syn'] >= ta) & (sc_2d['syn'] >= td))[main].mean()))
                                fprec['ae2d_ens'][str(t)].append(recall_at_fpr(z_syn, meta, z_cl, t))
                            if sd == LIST_SEED:
                                # ae2d_and: SAME two-threshold AND logic as the JSONL metrics (P99.5)
                                ta = np.percentile(sc_ae['holdout'], LIST_BUDGET)
                                td = np.percentile(sc_2d['holdout'], LIST_BUDGET)
                                fa_and = np.concatenate([(sc_ae['syn'] >= ta) & (sc_2d['syn'] >= td),
                                                         (sc_ae['analysis'] >= ta) & (sc_2d['analysis'] >= td)])
                                sc_and = np.concatenate([np.minimum(_z(sc_ae, 'syn'), _z(sc_2d, 'syn')),
                                                         np.minimum(_z(sc_ae, 'analysis'), _z(sc_2d, 'analysis'))])
                                det['ae2d_and'] = {'flag': fa_and, 'score': sc_and}
                                # ae2d_ens: SAME single-threshold-on-max-z logic as the JSONL metrics
                                te = np.percentile(z_ho, LIST_BUDGET)
                                det['ae2d_ens'] = {'flag': np.concatenate([z_syn >= te, z_an >= te]),
                                                   'score': np.concatenate([z_syn, z_an])}
                            tf.keras.backend.clear_session()
                        for v in variants:
                            if (proc, v, N) in dk:
                                continue
                            rec = {'item': proc, 'method': v, 'N': N, 'n_seeds': len(m_seeds),
                                   'fit_time_s': float(np.mean(fts)), 'infer_per_sample_s': 0.0,
                                   'budgets': {b: agg(pb[v][b]) for b in pb[v]},
                                   'recall_at_fpr': {t: agg_scalar(fprec[v][t]) for t in fprec[v]}}
                            open(JSONL, 'a', encoding='utf-8').write(json.dumps(rec) + '\n')
                            print(f'  {proc} {v:11} N={N:<6} recall_main={rec["budgets"]["99.5"]["recall_main"]["mean"]:.3f}', flush=True)
                    gc.collect(); continue
                # ---- generic method ----
                per_budget = {str(b): [] for b in BUDGETS}
                fprec = {str(t): [] for t in FP_TARGETS}
                per_self = []                              # self-calibrated operating point (optional)
                fts, ips = [], []
                skip_metrics = (proc, method, N) in dk
                for sd in m_seeds:
                    sc, ft, ip = score_sets(method, ctx, N, sd)
                    fts.append(ft); ips.append(ip if ip else 0.0)
                    cl = sc['syn'][meta['anomaly_type'].values == 'clean']
                    for b in BUDGETS:
                        thr = np.percentile(sc['holdout'], b)
                        per_budget[str(b)].append(metrics_at(sc['syn'] >= thr, meta, sc['holdout'], thr))
                    for t in FP_TARGETS:
                        fprec[str(t)].append(recall_at_fpr(sc['syn'], meta, cl, t))
                    if SELF_CAL_BUDGET is not None:        # deployment point: threshold from TRAIN scores
                        thr_s = np.percentile(sc['train'], SELF_CAL_BUDGET)
                        per_self.append(metrics_at(sc['syn'] >= thr_s, meta, sc['train'], thr_s))
                    if sd == LIST_SEED:
                        thr_l = np.percentile(sc['holdout'], LIST_BUDGET)
                        det[method] = {'flag': np.concatenate([sc['syn'] >= thr_l, sc['analysis'] >= thr_l]),
                                       'score': np.concatenate([sc['syn'], sc['analysis']])}
                    if method in ('ae', 'cnn1d') or method in COMPACT_AE:
                        tf.keras.backend.clear_session()
                if not skip_metrics:
                    rec = {'item': proc, 'method': method, 'N': N, 'n_seeds': len(m_seeds),
                           'fit_time_s': float(np.mean(fts)), 'infer_per_sample_s': float(np.mean(ips)),
                           'budgets': {b: agg(per_budget[b]) for b in per_budget},
                           'recall_at_fpr': {t: agg_scalar(fprec[t]) for t in fprec}}
                    if SELF_CAL_BUDGET is not None and per_self:
                        rec['self_cal'] = {'budget': SELF_CAL_BUDGET, **agg(per_self)}
                    open(JSONL, 'a', encoding='utf-8').write(json.dumps(rec) + '\n')
                    print(f'  {proc} {method:11} N={N:<6} recall_main={rec["budgets"]["99.5"]["recall_main"]["mean"]:.3f} '
                          f'spec_clean={rec["budgets"]["99.5"]["specificity_clean"]["mean"]:.3f} fit={rec["fit_time_s"]:.1f}s', flush=True)
                gc.collect()
            # ---- list.csv + images for this (item, N) ----
            if do_list and all(mo in det for mo in METHODS_OUT):
                write_list_and_images(proc, N, ctx, det)
        del ctx; gc.collect(); tf.keras.backend.clear_session()
    print('\nDONE', JSONL, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', nargs='+', default=[i[0] for i in ITEMS])
    ap.add_argument('--methods', nargs='+', default=METHODS_OUT)
    ap.add_argument('--no-list', action='store_true')
    args = ap.parse_args()
    items = [(p, s) for (p, s) in ITEMS if p in args.items]
    run(items, args.methods, not args.no_list)


if __name__ == '__main__':
    main()
