# Scripts manifest

Mapping of every deposited script to the manuscript section, table or figure it
supports. Analysis code lives in `analysis_full/`, figure generators in `figures/`,
the deep-learning baselines in `../Code/src/`, and a runnable sample-data demo is
`demo_mahalanobis.py`.

The submitted analysis covers **21 endpoint assays** and **11 anomaly detectors**;
the proposed method is the **empirical-covariance Mahalanobis distance**.

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
| `analysis_full/run_paper_method_comparison.py` | Core 11-detector benchmark engine; applies the common clean-holdout false-positive budget. Imports the deep-learning baselines from `../Code/src/` (needs TensorFlow). Methods §2.7–§2.8; Suppl. S5. |
| `analysis_full/run_method_comparison_p5_38.py` | Final configuration wrapper: fixed 5–38 reaction-point window, 21-item panel. **Table 1**; Fig. 2–4; Suppl. S5. |
| `analysis_full/method_comparison_posthoc.py` | Friedman omnibus + Nemenyi all-pairs post-hoc on mean ranks; base-curve-clustered bootstrap (Methods §2.11; Results §3.6; Suppl. S6.3, S6.7). |
| `analysis_full/method_comparison_stat_tests.py` | Assay-paired Wilcoxon signed-rank (one-sided, Holm–Bonferroni) + McNemar at the sample level (Methods §2.11; Results §3.6; Suppl. S6). |

### Deep-learning baselines — `../Code/src/`

Imported as `Code.src.model_*` by `run_paper_method_comparison.py`, which resolves
the repository root two levels above itself. Kept at the repository root so the
import path matches the manuscript citation. TensorFlow is required for these two
only; it is commented out in `requirements.txt` because it was run under a separate
Python 3.12.7 environment.

| Module | Manuscript |
| :-- | :-- |
| `../Code/src/model_autoencoder.py` | 1D convolutional autoencoder baseline, reconstruction error (Methods §2.7; Suppl. S5.2). |
| `../Code/src/model_1d_cnn.py` | 1D-CNN feature extractor feeding Isolation Forest (Methods §2.7; Suppl. S5.2). |

## Clean re-analysis of module-2 items (Results §3.2, §3.8; Suppl. S10b)

Three module-2 assays (Ca, ALB, IP) had their synthetic batteries regenerated on a
contemporaneous clean window after a transient optical event contaminated their
reference sets. The correction was applied identically to all eleven detectors.

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
| `analysis_full/run_temporal_robustness_all.py` | Rolling recent-500 scheme; month-to-month decision consistency (Cohen's κ) over 56 months across all 21 items → **Fig. 6a**; Suppl. S13. |
| `analysis_full/run_temporal_robustness_demo.py` | Single-item (CRE) monthly subtle-recall time series → **Fig. 6b**. |

## Real anomaly (Results §3.11; Suppl. S10)

| Script | Manuscript |
| :-- | :-- |
| `analysis_full/real_anomaly_cu_validation.py` | The one cause-known real anomaly (Cu reagent / monoclonal protein, analyzer #3, 2023-01-20): 76-point fingerprint location, leakage-free training on the most recent 500 pre-event samples, 239× / 345× threshold, rank 1 of 416. Methods §2.9; **Fig. 5**. |

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
| `figures/flatten_rgba.py` | Flattens transparency before submission (utility). |
| `figures/sync_from_analysis.py` | Copies regenerated figures from the analysis tree into the manuscript repository (utility). |

Figure generators consume the outputs of the analysis scripts (Parquet / `Results/`
/ figure panels); set `ABDETECT_BASE` accordingly or run after the analyses.

---

## Changelog

**2026-07-29** — Aligned with the final submitted manuscript.

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
