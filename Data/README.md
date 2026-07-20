# Data — reaction-curve CSV sample

One CSV per analyte item (endpoint assays). Same column layout as the internal processed store.

## Files
- `<ITEM>.csv` — reaction curves for one item, one curve per row.
- `items_info.csv` — data dictionary: `A_Name`, `main_wavelength_nm`, `sub_wavelength_nm`,
  `assay_type`, `sonication_level`.
- `manifest.csv` — per item: rows exported, rows in full dataset, exported date range.

## Column schema (`<ITEM>.csv`)
| Column | Type | Notes |
| :--- | :--- | :--- |
| `A_Name` | string | analyte / test item |
| `SystemName` | string | analyzer unit (#1/#2/#3) |
| `sample_id` | string | de-identified surrogate key (not the original barcode) |
| `S_Date` | datetime | `YYYY-MM-DD HH:MM:SS` |
| `Value` | float | reported result |
| `RAS1`…`RAS38` | int | sub-wavelength absorbance, points 1–38 |
| `RAM1`…`RAM38` | int | main-wavelength absorbance, points 1–38 |

## Notes
- Analytical signal: `Calc = RAM − RAS`.
- All 38 points included; recommended analysis window is points **5–38** (points 1–4 = bubble noise).
- Sample: up to 2,000 curves/item, random over the full period (seed = 42). See repo `README.md`.
