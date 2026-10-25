# Revision analyses

Scripts for the analyses added or changed in the revised manuscript: normalization without clipping, the
extended synthetic battery with non-additive disturbances, the compact autoencoders and the statistics over
thirteen detectors, the rolling evaluation on subsequent curves, the second Cu case, reagent-lot timing,
fixed-reference monitoring, and the within-day order of the curves.

## Layout and requirements

- The scripts keep the layout in which they were run: they import the pipeline modules one level up
  (`Scripts/analysis_full/`) and the shared helpers under `ForPaper/` at the repository root
  (`ForPaper/_common/panel.py`, `ForPaper/Suppl10_ReviewBurden/`, `ForPaper/Suppl02_EstimatorSelection/`,
  `ForPaper/04_MethodComparison/Figures_p5_38/`). Data are read from `Data/BigData/` and results are written
  to `Results/` under the repository root (or under `ABDETECT_BASE` where a script reads it).
- Earlier pipeline modules are imported unchanged; where a revision changes their behaviour (for example
  removing the clipping from the normalization), the revision script replaces the function at run time.
  The modules under `ForPaper/` are therefore the versions used for the submitted manuscript (with
  clipping); the scripts here swap the normalization when they import them.
- They need the full internal dataset; they do not run on the CSV sample in `Data/`.
- Scripts that import `rev_common.py` or `models_v2.py` need TensorFlow (Python 3.12, TensorFlow 2.20);
  the others run on Python 3.14 without TensorFlow.
- The real-case analyses (`s1e`, `s1r5`–`s1r9`, `s1d2`) need the identifier of the case specimen in the
  internal dataset and a local case file; neither is published. `real_anomaly_cu_validation.py` uses the
  placeholder 900000001 unless `CU_CASE_ID` is set. The scripts that located the case in the raw instrument
  exports are not included.
- `s1t_build_timeordered.py` and `s1t_run_timeorder.py` write and read a time-ordered copy of the processed
  store (`TIMEORDER_DIR`, default `Data/BigData/processed_timeordered`).
- The prefix gives the order of work: `s1*` re-run the submitted analyses, `s2*` build and evaluate the
  extended (v2) benchmark, `s3*` draw the figures. The header comment of each script gives its design,
  its inputs and its output folder.

## Scripts

Section numbers refer to the revised manuscript (§) and its supplement (S).

### Shared

| Script | Purpose |
| :-- | :-- |
| `rev_common.py` | Shared settings for the re-runs: imports the submitted pipeline, reproduces its three loader groups for the 21-assay panel, and provides the unclipped min–max normalization. |
| `models_v2.py` | Detector changes for the v2 benchmark: linear output layers for the 1D autoencoder and the 1D-CNN decoder (the submitted sigmoid outputs cannot reconstruct values outside 0–1), and the dense compact autoencoder 34-16-z-16-34. |

### Normalization without clipping (§2.5; values re-computed throughout)

| Script | Purpose |
| :-- | :-- |
| `s1a_clip_frequency.py` | How often clean and perturbed curves leave [0, 1] under the benchmark normalization. |
| `s1b_classical_rerun.py` | Re-runs the classical detectors of the 21-assay comparison with and without clipping. |
| `s1c_compare.py` | Compares those re-runs with the submitted results. |
| `s1z6_norm_from_N.py` | Takes the normalization constants from the N training curves only and checks that the classical detectors give the same results. |
| `s1z10_deploy_clipfreq.py` | Clipping frequency in the deployment-style analyses (workload, S9, S4, Cu case, S3.4), and whether the Mahalanobis decisions there equal those on the raw, unnormalized values. |
| `s1z13_render_range_freq.py` | How often the 2D-CNN rendering range [−0.05, 1.05] cuts an unclipped curve. |
| `s1z_tables_s5_2_s5_3.py` | Tables S5-2 and S5-3 from the unclipped runs. |
| `s1j_table_s5_noclip.py` | Per-method review budget under the deployment protocol, without clipping. |
| `s1e_cu_case_noclip.py` | The Cu case of S10 with and without clipping. |
| `s1f_event_interanalyzer_noclip.py` | Maintenance and reagent-lot robustness (S9) and cross-analyzer transferability (S4) without clipping. |
| `s1g_burden_fixedref_noclip.py` | Workload (Table S3, §4.4), fixed-reference IgG/IgM monitoring (§3.12) and the lot-aligned analyses of S14, without clipping. |
| `s1w_burden_monthly_noclip.py` | Monthly flag rates of S11.2 at N = 500 and N = 5,000, without clipping. |
| `s1i_compare_s11_s14.py` | Compares the submitted and unclipped outputs of `s1g`. |
| `s1h_compare_misc.py` | Comparison tables for S9, S4 and the Cu case, and the summary tables of the rolling evaluation. |
| `s1u_estimator_rolling_noclip.py` | Estimator selection under the rolling protocol (S3.4) with and without clipping. |
| `s1u2_figS2_noclip.py` | Fig. S2(b)–(d) from the unclipped run of `s1u`. |
| `s1v_optical_event_noclip.py` | The transient optical event of 2021-08 (S10b.2, S14.2) with and without clipping. |
| `s1x_prerebase_classical.py` | The module-2 assays (Ca, ALB, IP) on their original batteries, classical detectors, with and without clipping (S5.6, Table S5-3). |
| `s1y_prerebase_v2.py` | The same three assays on their original batteries through the v2 benchmark (Table S5-3). |

### Extended battery, v2 benchmark and statistics (§2.6–§2.8, §3.2–§3.6; S5, S6, S8.5–S8.6)

| Script | Purpose |
| :-- | :-- |
| `s2a_make_battery_v2.py` | Extra synthetic rows: additive anomalies below 1 σ, kinetic slowing (time warp) and saturation. They are appended at evaluation time; the submitted batteries are not modified. |
| `s2a2_bias_photometric.py` | Effect of each perturbation on the reported result, computed as the analyzer computes a two-point endpoint result. |
| `s2b_benchmark_v2.py` | The v2 method comparison: 21 assays, 13 detectors, unclipped normalization. |
| `s2c_summarize_v2.py` | Summarizes the v2 benchmark against the submitted results. |
| `s2d_clinical_severity.py` | Relates detection to the bias of the reported result. |
| `s2d2_m2_k50_yardsticks.py` | Severity at 50% recall (k50) on three yardsticks for all 13 detectors. |
| `s2d3_m2_spike_hit_recall.py` | Spike recall at the clinical limit over the curves whose reported value the spike changes. |
| `s2d4_m2_affected_recall.py` | Recall at the clinical limit over the affected curves, for every anomaly type whose effect varies by curve. |
| `s2d5_m2_supplement_tables.py` | Tables of S8.5–S8.6 from the result files (no new estimation). |
| `s2d6_m2_eflm_refresh.py` | Re-runs `s2d2`, `s2d4` (on the saved flags) and `s2d5` with the EFLM estimates read on 2026-10-22, which give the values in S8.5–S8.6; compares them with the 2026-09-17 estimates. |
| `s2i_plot_battery_examples.py` | Example plots of the added anomalies, for visual checking. |
| `s2i2_plot_nonadditive_photometric.py` | Non-additive anomalies on a typical base curve (CRE, IgG) for Fig. S5. |
| `s2h_2dcnn_ablation.py` | Ablation explaining the small change of the 2D-CNN → Isolation Forest recall between the submitted and v2 runs. |
| `s1z4_compact_selection_fix.py` | Corrected bottleneck selection of the dense compact autoencoder: the elbow rule applied to the dense-autoencoder rows only (an earlier summary had pooled rows of later runs). |
| `s1z5_downstream_compact_fix.py` | Re-runs the outputs that read the bottleneck selection, with the corrected one. |
| `s1z12_table_s5_1_v2.py` | Table S5-1 (recall at exactly matched clean false-positive rates, N = 500) from the v2 benchmark. |
| `s1z9a_samplelevel_flags_13.py` | Per-sample flags of the 13 detectors, input to the sample-level tests. |
| `s1z9b_stats_13methods.py` | Statistics of §3.6 and S6 over the 13 detectors. |
| `s2f_stats_v2.py` | Assay-blocked rank tests on the v2 benchmark (§3.6, S6). |
| `s2g_sample_level_tests.py` | Sample-level tests on the v2 benchmark (S6.5, S6.7). |

### Module-2 re-base: diagnostics for IP

| Script | Purpose |
| :-- | :-- |
| `s1z2_ip_rebase_diagnostic.py` | Why the compact-autoencoder worst-item recall at IP falls after the module-2 re-base (diagnostic only). |
| `s1z3_ip_pc_shape_figure.py` | Visual check of `s1z2`: does the baseline (type E) perturbation follow a principal direction of normal variation after the re-base? |
| `s1z7_ip_threshold_diagnostic.py` | Why the compact-autoencoder threshold at IP rises after the re-base (diagnostic only). |
| `s1z8_ip_widerange_swap.py` | Intervention test for `s1z7`: swaps wide-range curves into or out of the training set. |

### Rolling evaluation on subsequent curves (§3.10; S13; Fig. 6, Fig. S11)

| Script | Purpose |
| :-- | :-- |
| `s1d_rolling_contemporaneous.py` | Each monthly rolling recent-500 model is scored on the next, non-overlapping curves and on anomalies injected into those same curves. |
| `s1d3_fig6_figS11_rolling.py` | Fig. 6 and Fig. S11 from `s1d`. |
| `s1d4_fig6_flagrate_panel.py` | Replaces panel (b) of the `s1d3` figure: per assay, the median and 95th-percentile month of the routine flag rate for the same three methods. |
| `s1d2_cu_case_in_rolling.py` | Where the two abnormal curves of the Cu case fall in the rolling evaluation, and their ratio to the threshold (S13.6). |

### Cu cases (§3.11; S10; Fig. 5, Fig. S7)

The scripts call the case found during the revision, now the main case of §3.11 and Fig. 5, the "second Cu
case"; the case of the submitted Fig. 5 is now S10.4 and Fig. S7.

| Script | Purpose |
| :-- | :-- |
| `s1r_extract_times.py` | One pass over the raw instrument exports: all rows of the Cu case specimen (repeats and dilutions included; identifier written only as a salted hash) and a per-measurement time table with reagent lots for the 21 assays. |
| `s1r5_cu_case2_aligned.py` | The case found during the revision, scored with the protocol of S10. |
| `s1r6_cu_trainpool_case2_excluded.py` | Re-runs Cu in the main benchmark with and without the two case curves that lay in its training pool. |
| `s1r7_fig5_case2.py` | Fig. 5(a)(b) for that case. |
| `s1r8_fig5_case2_overlay.py` | Compact colour version of Fig. 5 (the published one). |
| `s1r9_figS7_case1.py` | Fig. S7 for the case of the submitted Fig. 5, in the style of the new Fig. 5. |

### Reagent lots and event timing (§3.8; S9; Fig. S6)

| Script | Purpose |
| :-- | :-- |
| `s1o_extract_lots.py` | Extracts the reagent-lot numbers from the raw instrument exports. |
| `s1q_s9_rawlot.py` | S9 re-run with the reagent-lot dates from the raw data. |
| `s1s_s9_exact_time.py` | S9 with exact reagent-lot onsets from the per-test measurement time. |
| `s1s2_figS6_exact_time.py` | Fig. S6 from `s1s`. |

### Fixed-reference monitoring and reagent-degradation episodes (§3.12; S14; Fig. S12)

| Script | Purpose |
| :-- | :-- |
| `s1k_fixed_reference_panel.py` | How often the fixed-reference monitor raises a cluster that is not a known event, over the 21 assays. |
| `s1l_drift_vs_qc.py` | Whether the unexplained clusters (Ca, ALB, LDL-C) reflect a stale reference or a change the control material does not show. |
| `s1m_fixed_reference_per_analyzer.py` | Whether the fixed-reference excess of `s1k` is an artefact of transfer across analyzers. |
| `s1n_cluster_classification.py` | Classifies the cluster episodes as known fault, reagent-lot change, calibration or unexplained, with a permutation control. |
| `s1p_lot_reanalysis.py` | Re-analyses the clusters with per-measurement reagent lots from the raw data. |
| `s1z11_igg_lot406.py` | IgG re-run with the reagent lot on analyzer #1 fixed to the lot named in Fig. S12, against the lot chosen from the data. |
| `s1z14_tables_s14_5_6.py` | The tables in S14.5 and S14.6: per-analyzer curve and saline-blank alert weeks and the curve lead, under the two-consecutive-week rule, from the unclipped weekly series of `s1g` (arm rev13) and `s1z11`. |
| `s1z15_figS12_noclip.py` | Fig. S12 redrawn from the unclipped series of `s1g` (arm rev13) and `s1z11`; uses `ForPaper/Rev16_FigureS12_ReagentEpisodes/make_figS12.py` and `s1z14`. |
| `s1z16_figS8_noclip.py` | Fig. S8(a)–(e) (operating point and the transient optical event) redrawn from unclipped runs. |
| `s1z17_figS3_S5a_noclip.py` | Fig. S3 (inter-analyzer transfer) and Fig. S5(a) (clean and benign flag rates) redrawn from unclipped results. |

### Order of curves within a day (S2.6)

| Script | Purpose |
| :-- | :-- |
| `s1t_build_timeordered.py` | Builds a copy of the processed store in which the curves of each day are in measurement-time order; dates are unchanged. |
| `s1t_run_timeorder.py` | Re-runs the main analyses on that copy. |
| `s1t_compare.py` | Compares file order and time order side by side. |

### Figures

| Script | Purpose |
| :-- | :-- |
| `s3a_figs_2to4_v2.py` | Fig. 2–4 from the v2 benchmark (21 assays, 13 detectors). |
| `s3c_compose_fig4_v2.py` | Composes Fig. 4: (a) operating-point summary and (b) computational cost. |
| `s3d_fig4a_and_figS9_S10_v2.py` | Fig. 4(a) with the legend above the bars, and Fig. S9 and Fig. S10 with 13 detectors. |
| `s3b_graphical_abstract_v2.py` | Graphical abstract (no data dependency). |
