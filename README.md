# Reaction-Curve Anomaly Detection — Dataset & Code (CCA resubmission)

Photometric **reaction-process (reaction-curve) time-series** data from clinical-chemistry
analyzers, and the accompanying anomaly-detection code, for the *Clinica Chimica Acta* (CCA)
resubmission of our study on detecting abnormal reactions from reaction-process data.

This repository ships a **de-identified, size-limited sample** of the dataset (see
[Data availability](#data-availability)). The full corpus (≈22.8 million curves) is too large to host
on GitHub.

## Repository layout

```
Export_For_GitHub_CCA/
├── Data/                       # Shared CSV sample (tracked)
│   ├── <ITEM>.csv              # one file per analyte (endpoint assays)
│   ├── items_info.csv          # data dictionary: wavelengths, assay type
│   ├── manifest.csv            # per-item exported/total row counts, date range
│   ├── reaction_point_window/  # frozen inputs for the point-window figure
│   │   ├── pointwise_variability.csv        # per-item x point roughness (p50/p99/p99.5/p99.9)
│   │   └── recommended_start_by_item.csv    # per-item recommended start (p99 + p99.5/p99.9 robustness)
│   └── README.md               # column schema & provenance
├── Scripts/
│   ├── demo_mahalanobis.py                  # runnable demo on the CSV sample (no full data needed)
│   ├── SCRIPTS_MANIFEST.md                  # maps every script to a manuscript section/figure
│   ├── export_processed_to_csv.py           # provenance: how the sample CSVs were produced
│   ├── make_pointwise_variability_figure.py # model-free reaction-point-window figure (paper)
│   ├── analysis_full/                       # the analysis code used for the paper (full dataset)
│   └── figures/                             # figure generators (Fig. 1–6, graphical abstract)
├── Figures/                    # paper figures (tracked)
│   └── pointwise_variability_{gray,color}.png
├── Results/                    # generated outputs (not tracked)
├── requirements.txt            # pinned dependencies
├── LICENSE                     # AGPL-3.0
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt
python Scripts/demo_mahalanobis.py --item CRE
```

This runs entirely on the shipped CSV sample and prints, for empirical Mahalanobis and the
classical baselines (PCA, LOF, Isolation Forest, One-Class SVM), the clean specificity and the
recall of the synthetic anomaly battery at severities 1/2/5/8 σ. Empirical Mahalanobis detects
even the subtle 1 σ anomalies that several baselines miss. This reproduces the *qualitative*
headline result; the published numbers use up to 15,000 date-sorted curves/item from the full
dataset (see [Data availability](#data-availability)) and additionally include the deep-learning
baselines (TensorFlow), which are omitted from the demo.

## Code and reproducibility

- `Scripts/analysis_full/` contains the analysis scripts used for the paper. They read the full
  internal dataset (`Data/BigData/processed`, Parquet) and write to `Results/`. The project root is
  taken from the `ABDETECT_BASE` environment variable, defaulting to the repository root.
- To run the core analysis scripts against the **shipped sample**, first convert the CSVs to the
  Parquet layout they expect: `python Scripts/csv_to_parquet.py` (writes `Data/BigData/processed/`).
  This yields sample-scale results; scripts needing the full multi-year volume, the specific known
  anomaly, or the instrument logs still require the full dataset.
- `Scripts/figures/` regenerates the manuscript figures from analysis outputs.
- **`Scripts/SCRIPTS_MANIFEST.md` maps every script to its manuscript section, table, and figure.**
- Environment: Python 3.12+ (tested on 3.14.2) with the pinned versions in `requirements.txt`;
  deep-learning baselines were run under Python 3.12.7 with TensorFlow 2.20.0.

## Reproducing the reaction-point-window figure

The choice to analyse photometric points **5–38** is justified in a **detector-independent** way: the
per-point roughness (|2nd difference| of each curve) shows that air-bubble instability is confined to
points 1–4 and settles by ~point 5 for 20/21 items (the sole exception, LDL-C, is a real early kinetic
feature, not noise). The recommendation is robust to the summary percentile (p99 → p99.5, where p99.5
matches the P99.5 detection operating point).

```bash
python Scripts/make_pointwise_variability_figure.py   # -> Figures/pointwise_variability_{gray,color}.png
```

It reads the frozen inputs in `Data/reaction_point_window/` (computed over the full internal dataset),
so it regenerates the exact paper figure without needing the raw data.

## Data format

Each `Data/<ITEM>.csv` holds one reaction curve per row, in the same column layout as the internal
processed store:

| Column | Meaning |
| :--- | :--- |
| `A_Name` | analyte / test item (e.g., `TP`, `CRE`, `IgG`) |
| `SystemName` | analyzer unit (`LABOSPECT 008 ALPHA #1/#2/#3`) |
| `sample_id` | **de-identified** surrogate key (see [Privacy](#privacy)) |
| `S_Date` | measurement timestamp |
| `Value` | reported concentration / result |
| `RAS1`…`RAS38` | **sub-wavelength** absorbance at photometric points 1–38 |
| `RAM1`…`RAM38` | **main-wavelength** absorbance at photometric points 1–38 |

- **All 38 photometric points are provided.** Points are common to every item (same reaction time
  grid). The reagent sequence is: R1 + sample mixed by ultrasonic stirring → after a fixed interval,
  R2 is added and the reaction proceeds. The reaction onset is around point ~20.
- **Calc (analytical) signal** = `RAM − RAS` (main minus sub), computed by the user. Subtracting the
  sub wavelength cancels benign optical offsets, so anomalies that survive in `Calc` are analytically
  meaningful.
- **Recommended analysis window: points 5–38.** Points 1–4 carry air-bubble noise from the initial
  ultrasonic mixing; the jitter settles by point ~5 (99th-percentile analysis in the study). Points
  1–4 are retained here so the choice of window remains the user's.
- Wavelengths per item are in `Data/items_info.csv`.

## Data availability

- **This repository**: a random sample of up to **2,000 curves per item** for the **21 endpoint-assay
  items** used in the paper, drawn across the full study period (2021–2025). ~42,000 curves, ~15 MB.
  See `Data/manifest.csv` for exact counts.
- **Full dataset** (≈22.8 M curves, all items): available from the authors on reasonable request /
  via a data repository (to be finalized before publication).

## Privacy

`sample_id` is **not** the original sample barcode. Each raw barcode was replaced by a deterministic
surrogate (SHA-256 prefix), consistent across items so that cross-item linkage is preserved while the
barcode itself is removed. Quality-control materials and instrument-flagged (alarmed) samples were
excluded during preprocessing; 2026 data (after an analyzer change) were excluded. See
`Scripts/export_processed_to_csv.py` for the exact procedure (set `ANONYMIZE_SID=False` only if ethics
approval permits publishing raw IDs).

## Ethics

Approved by the Institutional Review Board of Keio University School of Medicine (approval
No. **20231085**); opt-out consent, de-identified routinely generated instrument data.

## License

Code and data in this repository are released under the GNU Affero General Public License v3.0
([LICENSE](LICENSE)).
