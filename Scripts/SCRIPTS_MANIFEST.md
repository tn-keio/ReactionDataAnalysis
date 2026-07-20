# Scripts manifest

Mapping of deposited code to the manuscript. All analysis scripts live in
`analysis_full/`; figure generators in `figures/`; a runnable sample-data demo is
`demo_mahalanobis.py`.

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

## Running on the shipped sample (no full data needed)
| Script | Purpose |
| :-- | :-- |
| `demo_mahalanobis.py` | Empirical Mahalanobis vs PCA/LOF/IsolationForest/One-Class SVM on the CSV sample + synthetic anomaly battery. Reproduces the qualitative headline result. |
| `csv_to_parquet.py` | Converts `Data/*.csv` into the Hive-partitioned Parquet store (`Data/BigData/processed/A_Name=<item>/`) that `analysis_full/` expects, mapping `sample_id`→`S_ID`. Run this once to make the core analysis scripts runnable on the sample (sample-scale results). Does **not** enable scripts needing the full volume, the specific known anomaly, or the instrument logs. |

## Data preparation
| Script | Manuscript |
| :-- | :-- |
| `export_processed_to_csv.py` | Builds the shipped `Data/*.csv` sample from the internal store. |
| `analysis_full/count_exclusion_flow.py` | CONSORT-style exclusion flow (Methods §2.4; Suppl. S1). |

## Synthetic anomaly battery
| Script | Manuscript |
| :-- | :-- |
| `analysis_full/generate_endpoint_synthetic_anomalies_p5_38.py` | Generates the A–E + benign_offset battery on points 5–38 (Methods §2.6). |
| `analysis_full/synthetic_validity_qc.py` | Validity QC (clean≈real, ΔCalc=0, severity=k×σ̄); Suppl. S7. |

## Method comparison (Results §3.2–3.6)
| Script | Manuscript |
| :-- | :-- |
| `analysis_full/run_paper_method_comparison.py` | Core 11-detector benchmark engine (imports TensorFlow for the DL baselines). |
| `analysis_full/run_method_comparison_p5_38.py` | Final configuration wrapper (points 5–38, 14-item panel). Table 1; Fig. 2–4. |
| `analysis_full/method_comparison_posthoc.py` | Friedman + Nemenyi (Suppl. S5). |
| `analysis_full/method_comparison_stat_tests.py` | Item-paired Wilcoxon + McNemar (Suppl. S5). |

## Clean re-analysis of module-2 items (Results §3.2/§3.7; Suppl. S9b)
| Script | Manuscript |
| :-- | :-- |
| `analysis_full/generate_ca_clean_synthetic.py`, `rerun_ca_clean.py` | Ca on a contemporaneous clean window. |
| `analysis_full/generate_m2_clean_synthetic.py`, `rerun_m2_clean.py` | ALB, IP on a contemporaneous clean window. |

## Robustness
| Script | Manuscript |
| :-- | :-- |
| `analysis_full/load_maintenance_lot_events.py` | Parses the maintenance / reagent-lot logs. |
| `analysis_full/event_robustness_analysis.py`, `event_robustness_figures.py` | Maintenance / lot robustness vs placebo (Results §3.7; Suppl. S8). |
| `analysis_full/inter_analyzer_analysis.py`, `inter_analyzer_figures.py` | Cross-analyzer transfer and pooling (Results §3.8; Suppl. S3). |

## Temporal robustness (Results §3.9; Suppl. S13)
| Script | Manuscript |
| :-- | :-- |
| `analysis_full/run_temporal_robustness_all.py` | 56-month decision-consistency (Cohen's κ) across all items → Fig. 6, Suppl. S11. |
| `analysis_full/run_temporal_robustness_demo.py` | Single-item (CRE) monthly time series. |

## Real anomaly (Results §3.10; Suppl. S9)
| Script | Manuscript |
| :-- | :-- |
| `analysis_full/real_anomaly_cu_validation.py` | The known Cu reagent anomaly; 239×/345× threshold, rank 1/416. Fig. 5. |

## Supplementary analyses
| Script | Manuscript |
| :-- | :-- |
| `analysis_full/run_estimator_selection.py` | Empirical vs robust covariance (Suppl. S2). |
| `analysis_full/run_pointwise_variability.py`, `make_pointwise_variability_figure.py` | Reaction-point window / early-point variability (Suppl. S6, Fig. S4). |
| `analysis_full/run_reaction_points_forpaper.py` | Reaction-point window sweep. |

## Figures
| Script | Manuscript |
| :-- | :-- |
| `figures/make_figures.py` | Method-comparison figures (Fig. 2, 3, 4-source; Fig. S9, S10). |
| `figures/make_fig1_curves.py` | Fig. 1 (calc channel + anomaly battery, multi-severity). |
| `figures/make_fig6_temporal.py` | Fig. 6 (temporal robustness κ). |
| `figures/make_graphical_abstract.py` | Graphical abstract. |
| `figures/compose_fig4_fig5.py` | Composes Fig. 4 and Fig. 5 from source panels. |

Figure generators consume the outputs of the analysis scripts (Parquet / `Results/`
/ figure panels); set `ABDETECT_BASE` accordingly or run after the analyses.
