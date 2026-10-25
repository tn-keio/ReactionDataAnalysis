# Scripts manifest

Mapping of every deposited script to the manuscript section, table or figure it
supports. Analysis code lives in `analysis_full/`, figure generators in `figures/`,
the deep-learning baselines in `../Code/src/`, and a runnable sample-data demo is
`demo_mahalanobis.py`. The analyses added or changed in the revision, and the
generators of the revised figures, are in `analysis_full/revision/` (see
[Revision analyses](#revision-analyses) below and the README in that folder).

The analysis covers **21 endpoint assays** and **13 anomaly detectors**;
the proposed method is the **empirical-covariance Mahalanobis distance**.

**Normalization.** The calc channel is min–max scaled without clipping (a purely affine map).
The method comparison uses one scalar pair for all reaction points, taken from the
10,000-curve training pool; the deployment-style analyses (Cu case, maintenance and lot
events, transferability, estimator selection) use one pair per reaction point taken from
each training window, and the workload analysis uses a scalar pair from each window.

## Paths and data

Analysis scripts resolve the project root from the `ABDETECT_BASE` environment
variable, falling back to the repository root (two levels above the script). They
read the **full internal dataset** from `<BASE>/Data/BigData/processed` (Parquet)
and write to `<BASE>/Results`. The full dataset is de-identified clinical data,
available on request under the IRB conditions (see repository `README.md`). The
shipped `Data/*.csv` is a 2,000-curve/item random sample for demonstration.

To run an analysis script against a local copy of the full data:

```
export ABDETECT_BASE=/path/to/full/dataset/root   # contains Data/BigData/processed
python analysis_full/<script>.py
```

Two further variables are honoured where relevant: `PAPER_FIGURES_DIR` (figure
output directory) and `PAPER_REPO_DIR` (a checkout of the manuscript repository,
used only by `measure_review_burden_unified.py`; it otherwise falls back to
`Data/item_counts_ForPaper21.csv` shipped here).

## Running on the shipped sample (no full data needed)

| Script | Purpose |
| :-- | :-- |
| `demo_mahalanobis.py` | Empirical Mahalanobis vs PCA/LOF/IsolationForest/One-Class SVM on the CSV sample + synthetic anomaly battery. Reproduces the qualitative headline result. |
| `csv_to_parquet.py` | Converts `Data/*.csv` into the Hive-partitioned Parquet store (`Data/BigData/processed/A_Name=<item>/`) that `analysis_full/` expects, mapping `sample_id`→`S_ID`. Run this once to make the core analysis scripts runnable on the sample (sample-scale results). Does **not** enable scripts needing the full volume, the specific known anomaly, or the instrument logs. |

## Data preparation

| Script | Manuscript |
| :-- | :-- |
| `export_processed_to_csv.py` | Builds the shipped `Data/*.csv` sample from the internal store. |
| `analysis_full/count_exclusion_flow.py` | Per-stage exclusion flow, 40,054,608 → 22,781,420 curves (Methods §2.4; Results §3.1; Suppl. S2). |

## Synthetic anomaly battery

| Script | Manuscript |
| :-- | :-- |
| `analysis_full/generate_endpoint_synthetic_anomalies_p5_38.py` | Generates the A–E + benign_offset battery on points 5–38 (Methods §2.6). |
| `analysis_full/synthetic_validity_qc.py` | Validity QC: clean-synthetic ≈ real normal, ΔCalc = 0 for benign_offset, severity calibrated as k × σ̄ (Suppl. S8). |

## Method comparison (Results §3.2–§3.6; Table 1)

| Script | Manuscript |
| :-- | :-- |
| `analysis_full/run_paper_method_comparison.py` | Core benchmark engine (the 11 original detectors plus the compact autoencoders); applies the common clean-holdout false-positive budget. Imports the deep-learning baselines from `../Code/src/` (needs TensorFlow). Methods §2.7–§2.8; Suppl. S5. |
| `analysis_full/select_compact_ae.py` | Chooses the bottleneck size z of the dense compact autoencoder from the clean validation MSE alone (elbow rule) and writes its metrics as `dae_elbow`. Table 1; Suppl. S5. |
| `analysis_full/run_method_comparison_p5_38.py` | Final configuration wrapper: fixed 5–38 reaction-point window, 21-item panel (Ca, ALB and IP are taken from the clean re-runs below). **Table 1**; Fig. 2–4; Suppl. S5. |
| `analysis_full/method_comparison_posthoc.py` | Friedman omnibus, mean ranks and Nemenyi all-pairs post-hoc over the 21 assays × 13 detectors; item-block bootstrap of the mean and worst-item recall (Methods §2.11; Results §3.6; Suppl. S6.2, S6.3, S6.8; Tables S6-1, S6-3). |
| `analysis_full/method_comparison_stat_tests.py` | Assay-paired Wilcoxon signed-rank (one-sided, Holm–Bonferroni); sample-level McNemar discordance with a base-curve cluster bootstrap (Methods §2.11; Results §3.6; Suppl. S6.4, S6.5, S6.7; Table S6-2). |

### Deep-learning baselines — `../Code/src/`

Imported as `Code.src.model_*` by `run_paper_method_comparison.py`, which resolves
the repository root two levels above itself. Kept at the repository root so the
import path matches the manuscript citation. TensorFlow is required for these two
only; it is commented out in `requirements.txt` because it was run under a separate
Python 3.12.7 environment.

| Module | Manuscript |
| :-- | :-- |
| `../Code/src/model_autoencoder.py` | 1D convolutional autoencoder baseline, reconstruction error; linear output layer (Methods §2.7; Suppl. S5.2). |
| `../Code/src/model_1d_cnn.py` | 1D-CNN feature extractor feeding Isolation Forest; linear decoder output (Methods §2.7; Suppl. S5.2). |
| `../Code/src/model_compact_autoencoder.py` | Compact-bottleneck autoencoders: dense 34-16-z-16-34 (`dae<z>`) and the 1D-AE topology with a Dense(z) code (`cae<z>`) (Methods §2.7; Suppl. S5). |

The 2D-CNN pipeline renders each normalized curve on the y-range [−0.05, 1.05]
(`RENDER_YLIM`); points outside this range are not drawn.

## Clean re-analysis of module-2 items (Results §3.2, §3.8; Suppl. S10b)

Three module-2 assays (Ca, ALB, IP) had their synthetic batteries regenerated on a
contemporaneous clean window after a transient optical event contaminated their
reference sets. The correction was applied identically to all detectors.

| Script | Manuscript |
| :-- | :-- |
| `analysis_full/generate_ca_clean_synthetic.py`, `rerun_ca_clean.py` | Ca on a contemporaneous clean window. |
| `analysis_full/generate_m2_clean_synthetic.py`, `rerun_m2_clean.py` | ALB and IP on a contemporaneous clean window. |

## Robustness

| Script | Manuscript |
| :-- | :-- |
| `analysis_full/load_maintenance_lot_events.py` | Parses the instrument-maintenance and reagent-lot logs (2021–2025). |
| `analysis_full/event_robustness_analysis.py`, `event_robustness_figures.py` | 500-before / 500-after models against a placebo null of event-free dates (Methods §2.10; Results §3.8; Suppl. S9, Fig. S6). |
| `analysis_full/inter_analyzer_analysis.py`, `inter_analyzer_figures.py` | Quarterly cross-analyzer transfer matrices, pooling penalty, leave-one-analyzer-out (Methods §2.10; Results §3.9; Suppl. S4, Fig. S3). |

## Temporal robustness (Results §3.10; Suppl. S13)

| Script | Manuscript |
| :-- | :-- |
| `analysis_full/run_temporal_robustness_all.py` | Rolling recent-500 scheme; month-to-month decision consistency (Cohen's κ) over 56 months across all 21 items → **Fig. 6a** of the submitted manuscript; Suppl. S13. |
| `analysis_full/run_temporal_robustness_demo.py` | Single-item (CRE) monthly subtle-recall time series → **Fig. 6b** of the submitted manuscript. |
| `analysis_full/contemporaneous_synth.py`, `run_temporal_robustness_screened.py` | Helpers imported by the revision analyses (anomalies injected into contemporaneous curves; the screened rolling scheme). |

In the revised manuscript the κ analysis is removed, and Fig. 6 and Fig. S11 show
each rolling model evaluated on the curves that followed it
(`analysis_full/revision/s1d_rolling_contemporaneous.py`, `s1d3_fig6_figS11_rolling.py`).

## Real anomaly (Results §3.11; Suppl. S10)

| Script | Manuscript |
| :-- | :-- |
| `analysis_full/real_anomaly_cu_validation.py` | Cu abnormal reaction (specimen containing a monoclonal protein, analyzer #3): 76-point fingerprint location, leakage-free training on the most recent 500 pre-event samples, 239× / 345× threshold, rank 1 of 416. Methods §2.9; **Fig. 5** of the submitted manuscript (now Suppl. S10.4, **Fig. S7**). The case identifier is not published: the script uses the placeholder 900000001 unless `CU_CASE_ID` is set. |

The Cu case found during the revision (now §3.11 and **Fig. 5**) is analysed by
`analysis_full/revision/s1r5_cu_case2_aligned.py` and drawn by `s1r7`/`s1r8`; Fig. S7
is drawn by `s1r9`.

## Deployment and workload

| Script | Manuscript |
| :-- | :-- |
| `analysis_full/measure_review_burden_unified.py` | Daily review burden under the rolling recent-500 scheme: flag rate 1.34%, 92.3 flagged curves per calendar day (Discussion §4.4; Suppl. S11.2, Table S3). |

## Supplementary analyses

| Script | Manuscript |
| :-- | :-- |
| `analysis_full/run_estimator_selection.py` | Empirical vs robust covariance, including the Ledoit–Wolf shrinkage variant under the rolling protocol (Suppl. S3). |
| `analysis_full/run_pointwise_variability.py`, `make_pointwise_variability_figure.py` | Model-free point-wise variability that justifies the 5–38 window (Methods §2.5; Suppl. S7, **Fig. S4**). |
| `analysis_full/run_reaction_points_forpaper.py` | Reaction-point window sweep. |

## Figures

| Script | Manuscript |
| :-- | :-- |
| `figures/make_fig1_curves.py` | Reaction-curve morphology, calc channel (points 5–38), synthetic anomaly battery → **Fig. 1**. |
| `figures/make_figures.py` | Detection performance vs training size; subtle (1σ) recall; operating-point and computational-cost panels → **Fig. 2, 3**, and the source panels for Fig. 4. |
| `figures/compose_fig4_fig5.py` | Stacks the source panels into the two-panel layouts → **Fig. 4, Fig. 5(a,b)**. |
| `figures/add_panel_c_to_fig5.py` | Adds the cuvette photograph panel → **Fig. 5c**. |
| `figures/make_fig6_temporal.py` | Month-to-month decision consistency and monthly subtle recall → **Fig. 6**. |
| `figures/make_reaction_point_window_figure.py` | Reaction-point window on representative curves; companion panel to the variability figure → **Fig. S4**. |
| `figures/make_graphical_abstract.py` | Graphical abstract. |
| `figures/convert_tif_to_png.py` | Format conversion helper for submitted artwork (utility). |
| `figures/flatten_rgba.py` | Flattens transparency for journal artwork requirements (utility). |
| `figures/sync_from_analysis.py` | Copies regenerated figures from the analysis tree into the manuscript repository (utility). |

Figure generators consume the outputs of the analysis scripts (Parquet / `Results/`
/ figure panels); set `ABDETECT_BASE` accordingly or run after the analyses.
The figures changed in the revision are drawn by the scripts in
`analysis_full/revision/` (next section).

## Revision analyses

`analysis_full/revision/` holds the analyses added or changed in the revised
manuscript. They import the modules above and the helpers in `../ForPaper/`
unchanged and replace, at run time, only what the revision changes (for example the
clipping in the normalization). They need the full dataset. Every script is listed
with its purpose in `analysis_full/revision/README.md`; the main entries are:

| Script | Manuscript |
| :-- | :-- |
| `revision/s1b_classical_rerun.py`, `s1e`–`s1j`, `s1u`–`s1y`, `s1z*` | Normalization without clipping, re-computed for the method comparison and the deployment-style analyses (Methods §2.5; values throughout). |
| `revision/s2a_make_battery_v2.py`, `s2b_benchmark_v2.py` | Extended battery (sub-1σ, time warp, saturation) and the 13-detector comparison (Methods §2.6–§2.8; **Table 1**; Results §3.2–§3.5; Suppl. S5, S8.5). |
| `revision/s2a2_bias_photometric.py`, `s2d*` | Effect of the anomalies on the reported result; recall at the clinical limit (Results §3.3; Suppl. S8.5–S8.6). |
| `revision/s1z9a_samplelevel_flags_13.py`, `s1z9b_stats_13methods.py`, `s2f`, `s2g` | Statistics over the 13 detectors (Results §3.6; Suppl. S6). |
| `revision/s1d_rolling_contemporaneous.py`, `s1d3`, `s1d4`, `s1d2` | Rolling models evaluated on subsequent curves (Results §3.10; Suppl. S13; **Fig. 6**, Fig. S11). |
| `revision/s1r5_cu_case2_aligned.py`, `s1r6`–`s1r9`, `s1r_extract_times.py` | The Cu cases (Results §3.11; Suppl. S10; **Fig. 5**, Fig. S7). |
| `revision/s1o`, `s1q`, `s1s`, `s1s2` | Reagent-lot dates and exact onsets from the raw data (Results §3.8; Suppl. S9; Fig. S6). |
| `revision/s1k`–`s1n`, `s1p`, `s1z11`, `s1z14` | Fixed-reference monitoring and the reagent-degradation episodes (Results §3.12; Suppl. S14; Fig. S12). |
| `revision/s1t_*` | Order of curves within a day (Suppl. S2.6). |
| `revision/s3a`–`s3d` | Fig. 2–4, Fig. S9, Fig. S10 and the graphical abstract. |

### Shared helpers — `../ForPaper/`

Imported by the revision analyses at the paths they were run from. These are the
versions used for the submitted manuscript; the revision scripts swap the
normalization when they import them.

| Module | Used for |
| :-- | :-- |
| `../ForPaper/_common/panel.py` | Assay panels, assay-name spellings and result locations shared by the revision scripts, including the precedence of the result sources behind Table 1. |
| `../ForPaper/Suppl10_ReviewBurden/measure_review_burden_unified.py` | Rolling flag rates and workload (Table S3, S11.2), fixed-reference monitoring (S14). Same as `analysis_full/measure_review_burden_unified.py` except for the clipping and the path of the curve-count table. |
| `../ForPaper/Suppl10_ReviewBurden/compare_methods_unified.py` | Per-method review budget under the deployment protocol. |
| `../ForPaper/Suppl02_EstimatorSelection/make_estimator_selection_figure.py` | Fig. S2 (called by `revision/s1u2_figS2_noclip.py`). |
| `../ForPaper/04_MethodComparison/Figures_p5_38/make_figures.py` | Plotting functions reused by `revision/s3a`, `s3d` and `s1d3`. |
| `../ForPaper/Rev13_PerAnalyzer_LotAligned/per_analyzer_lot.py` | Per-analyzer, lot-anchored weekly series of the fixed-reference flag rate and the saline blank (S14.5–S14.6, Fig. S12); run through `revision/s1g` (arm rev13) and `revision/s1z11`. |

---

## Changelog

**2026-10-25** — Tables S14.5 and S14.6; two scripts and one data column that were missing.

- Added `analysis_full/revision/s1z14_tables_s14_5_6.py`, which writes the two
  per-analyzer timing tables of S14.5 and S14.6 with the two-consecutive-week alert
  rule stated in the supplement.
- Added `../ForPaper/Rev13_PerAnalyzer_LotAligned/per_analyzer_lot.py`, the module that
  `revision/s1g` and `revision/s1z11` load. Its verdict file records the first week each
  signal crosses its level, not the two-week rule.
- Added `analysis_full/revision/s2d6_m2_eflm_refresh.py` (the clinical-limit tables of
  S8.5–S8.6 with the EFLM estimates read on 2026-10-22) and
  `analysis_full/revision/s1d4_fig6_flagrate_panel.py` (panel (b) of the rolling-evaluation
  figure).
- `Data/item_counts_ForPaper21.csv`: `in_method_comparison` set to Y for the eight assays
  where it was blank (TG, LDL-C, HDL-C, TC, IgG, KL-6, IgA, IgM); all 21 assays are in the
  method comparison.

**2026-10-22** — Revision analyses; de-identification of the sample data.

- Added `analysis_full/revision/` (70 scripts and a README) with the analyses of the
  revised manuscript, the helpers they import (`../ForPaper/`), and
  `analysis_full/contemporaneous_synth.py` and `run_temporal_robustness_screened.py`.
- `Data/*.csv`: `sample_id` is now a dummy sequential number (1000000001, …); see the
  repository README, Privacy.
- `real_anomaly_cu_validation.py`: the case identifier is no longer in the code
  (placeholder 900000001, or `CU_CASE_ID`); the case date is taken from the case row.
- `run_method_comparison_p5_38.py`: removed a local environment path from a comment.

**2026-10-17** — Statistics over the 13 detectors.

- `method_comparison_posthoc.py` and `method_comparison_stat_tests.py` now cover
  the 21-item panel and the 13 detectors (the 11 original ones plus the two compact
  autoencoders), and add the item-block bootstrap (Table S6-3) and the base-curve
  cluster bootstrap of the McNemar discordance (Table S6-2).
- `run_method_comparison_p5_38.py` runs the 21-item panel (previously a 14-item list;
  GA is still computed for `figures/make_figures.py`).

**2026-10-16** — Normalization without clipping; compact autoencoders.

- Removed the clipping to [0, 1] from every min–max normalization
  (`run_paper_method_comparison.py`, `run_estimator_selection.py`,
  `event_robustness_analysis.py`, `inter_analyzer_analysis.py`,
  `real_anomaly_cu_validation.py`, `measure_review_burden_unified.py`,
  `run_temporal_robustness_all.py`, `run_temporal_robustness_demo.py`).
- Corrected the description of the benchmark normalization: it is one scalar pair
  for all reaction points, not a per-wavelength pair.
- Deep-learning baselines: linear output layers in `model_autoencoder.py` and
  `model_1d_cnn.py`; 2D-CNN rendering range widened from [0, 1] to [−0.05, 1.05].
- Added the compact autoencoders (`Code/src/model_compact_autoencoder.py`) and the
  bottleneck selection (`analysis_full/select_compact_ae.py`).
- The benchmark now also reports Type D recall by lag (`recall_D_lag<k>`).

**2026-07-29** — Aligned with the final manuscript.

- Corrected the method-comparison panel size from 14 items to **21**.
- Re-mapped every entry onto the current supplement numbering (S1–S14 as deposited
  with the manuscript); several entries previously pointed at superseded numbers.
- Added the deep-learning baselines `Code/src/model_autoencoder.py` and
  `model_1d_cnn.py`, which `run_paper_method_comparison.py` imports and which were
  missing from the deposit.
- Added `analysis_full/measure_review_burden_unified.py` (Suppl. S11.2, Table S3),
  which was missing from the deposit.
- Added the figure utilities `add_panel_c_to_fig5.py`,
  `make_reaction_point_window_figure.py`, `convert_tif_to_png.py`,
  `flatten_rgba.py` and `sync_from_analysis.py`.
- Fixed a path bug in `make_fig1_curves.py`, `make_fig6_temporal.py` and
  `compose_fig4_fig5.py`: a leading backslash inside `os.path.join(REPO, "\...")`
  made the argument absolute, so `ABDETECT_BASE` was silently discarded.
- Documented the `PAPER_FIGURES_DIR` and `PAPER_REPO_DIR` variables.
