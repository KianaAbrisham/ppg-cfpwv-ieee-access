# PPG Feature Analysis and cf-PWV Regression

Research code for PPG/SDPPG feature extraction, exploratory age associations and XGBoost
regression of carotid–femoral pulse wave velocity (cf-PWV).

Related paper: **Noninvasive Assessment of Arterial Stiffness Using Photoplethysmography:
Feature Analysis and Machine Learning-Based Estimation of Carotid-Femoral Pulse Wave Velocity**
(IEEE Access, 2025). [DOI](https://doi.org/10.1109/ACCESS.2025.3626252) ·
[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11218839).

This repository requires the original study CSV exports or an independently verified conversion.
They are not included. The input checks and a synthetic regression run have been tested;
**full reproduction of the paper's results has not been established**.

## What the code does

- Extract pulse timing, amplitude, area, shape and SDPPG features.
- Calculate Pearson age correlations and exploratory p-values.
- Evaluate a fixed XGBoost regressor using five subject-wise folds at the radial site.
- Save fold metrics, subject-level out-of-fold predictions, diagnostic figures and permutation importance.
- Reject duplicate or missing subject identities and record expected input-quality exclusions.

## Required input

The related dataset is [PWDB on Zenodo](https://zenodo.org/records/2633175), an in-silico pulse-wave database.
Downloading the source dataset alone does not create the seven project-specific CSV exports below.
This repository does not currently provide a validated source-to-CSV converter or fiducial extractor.

Run from the repository root and place these files in `data/`:

| File | Required content |
|---|---|
| `PWs_Digital_PPG.csv` | One digital pulse waveform per subject |
| `PWs_Radial_PPG.csv` | One radial pulse waveform per subject |
| `PWs_Brachial_PPG.csv` | One brachial pulse waveform per subject |
| `digfeatures.csv` | Digital fiducial values/times, SI, RI and age |
| `radfeature.csv` | Radial fiducial values/times, SI, RI and age |
| `brachfeatures.csv` | Brachial fiducial values/times, SI, RI and age |
| `PWV.csv` | Subject IDs and positive `PWV_cf [m/s]` targets |

Every table must contain a unique, positive integer **`Subject Number`**. Matching is by this ID,
never row position. If an older export lacks IDs, recover its verified identities from the source;
do not add sequential IDs on the assumption that row order is correct.

Waveform tables contain `Subject Number`, optional export-index columns named `Unnamed: ...`,
and **only waveform samples** in the remaining columns, in acquisition order. Remove other
metadata explicitly before use. Samples are assumed to represent one pulse at **500 Hz** (`FS`).
Trailing blank padding is allowed. Leading/internal gaps, infinite samples and constant pulses are excluded
with reasons; removing internal gaps would corrupt timing. Confirm the source sampling rate before use.

Each fiducial table must have `Age`, `Subject Number`, and the following fields, using the matching
prefix `Digital`, `Radial` or `Brachial`:

- `{prefix}_PPGsys_V`, `{prefix}_PPGsys_T`, `{prefix}_PPGdia_V`, `{prefix}_PPGdia_T`.
- `{prefix}_PPGa_V`/`_T`, and the corresponding `b`, `c`, `d`, `e` value/time pairs.
- `{prefix}_SI` and `{prefix}_RI`.

Times must be in **seconds relative to the first waveform sample**, not sample indices. Systolic time
must leave at least two samples in each phase. Fiducials and ages must be finite and ages positive.
Additional target rows are permitted, but every retained radial subject needs one matching target.

## Evaluation and interpretation

The regressor uses 400 trees, learning rate 0.05, maximum depth 5 and a fixed seed. Quantile-stratified
folds are used when feasible, with KFold as a fallback. There is no model tuning or feature selection
inside this script. Age and subject ID are excluded from predictors; age-correlation filtering affects
the exploratory reports, not the regression feature set. At least ten retained subjects are required
to give each of five test folds at least two observations; meaningful research evaluation requires more.

Undefined derived features are represented as NaN for XGBoost's missing-value handling.
Permutation importance measures the change in test-fold R² and may be negative. It is a descriptive
diagnostic, not a separate feature-selection experiment. Fold standard deviations are not confidence
intervals. Repeated tuning requires a separate final test set or nested evaluation.

Correlation p-values are unadjusted for multiple comparisons. The feature named
`Rise–Decay Time Ratio` retains the original **decay/rise** calculation. Amplitude is half the
peak-to-peak range. Legacy geometric-length features mix time and amplitude coordinates and are
scale-dependent descriptors, not physical distances. Signal amplitude units are inherited from the
input; they are not assumed to be volts. These definitions and the original export convention still
need to be checked against the study data before interpreting research results.

## Install and run

Use Python 3.12 and a separate environment:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` in Windows Command Prompt or
`source .venv/bin/activate` on Linux/macOS, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python paper_code.py
```

Tests use artificial fixtures and do not need the study CSVs. The final command does require them.
Outputs go to `outputs/figures/` and `outputs/tables/`; another run overwrites files with the same names.
The script does not save a deployable model or provide inference for new subjects.
See [validation](docs/VALIDATION.md) for the checks actually completed.

## License

MIT — see [LICENSE](LICENSE).
