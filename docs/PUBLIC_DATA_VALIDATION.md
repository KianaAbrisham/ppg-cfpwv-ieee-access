# Public PWDB evaluation — 2026-09-28

The official public dataset was downloaded, verified and converted into the seven required input files. The current repository's feature extraction was evaluated at all three sites, and the radial cf-PWV model completed five-fold cross-validation on 4,333 eligible virtual subjects.

| Metric | Current mean | Current fold SD | Paper mean |
| --- | ---: | ---: | ---: |
| MAE (m/s) | 0.115708 | 0.004594 | 0.115 |
| RMSE (m/s) | 0.180949 | 0.011586 | 0.180 |
| R² | 0.992536 | 0.001009 | 0.993 |

The errors are close to the published values, with small numerical differences. Full-paper numerical reproduction is not claimed. The SD column uses the original code's population convention (`ddof=0`); [run.json](public_data/run.json) also records sample SD (`ddof=1`). Neither is a confidence interval. The training-mean baseline's average RMSE was 2.0998 m/s.

## Data and protocol

The source is PWDB v0.2, Zenodo record 3275625. Digital, radial and brachial waveforms contain 4,374 subjects each. Requiring complete provided fiducials retains 4,328 digital, 4,333 radial and 4,374 brachial subjects; exclusions are recorded by ID and reason.

The radial model uses 60 features. Age and subject ID are excluded from predictors. Five outer folds are stratified by cf-PWV deciles, with shuffle enabled and seed 42. XGBoost uses 400 trees, learning rate 0.05, maximum depth 5, squared-error loss and one CPU worker. No model tuning was performed. Permutation importance uses ten repeats on each test fold and is descriptive, not a feature-selection step.

All source-to-target matching uses explicit subject IDs. Every retained radial subject has exactly one out-of-fold prediction, with no subject overlap between training and test within a fold. Missing derived values use the existing code's XGBoost handling; this run produced no missing derived feature cells among the retained radial subjects.

## Run it

```bash
python tools/prepare_public_pwdb.py --source data/pwdb-source --output data/pwdb --download
python tools/run_public_validation.py --repo . --data data/pwdb/ieee/data --output runs/pwdb-validation
```

The validation runner imports the repository's feature extraction and correlation functions unchanged. It records feature tables, exclusions, fold metrics, predictions, splits, permutation importance and five native `.ubj` models. It reloads each model and requires identical predictions on its entire held-out fold. It uses native Booster serialization because the XGBoost 3.0.2 sklearn wrapper's `save_model` method is incompatible with sklearn 1.8's removal of `_estimator_type`.

The existing `paper_code.py` entry point still generates its full publication-layout figures from the seven files in `data/`; the new runner separates numerical validation from those large figures. Choose a new output directory for each validation run.

## Evidence

- [Source conversion and checksums](public_data/conversion_manifest.json).
- [Run metadata and exact scores](public_data/run.json); [fold metrics](public_data/fold_metrics.csv).
- [Subject-level held-out predictions, compressed CSV](public_data/held_out_predictions.csv.gz).
- [Independent verification](public_data/verification.json); [permutation importance](public_data/permutation_importance.csv).
- Excluded subjects: [digital](public_data/digital_excluded.csv), [radial](public_data/radial_excluded.csv), [brachial](public_data/brachial_excluded.csv).

Execution used Linux CPU, Python 3.12.14, NumPy 2.3.5, Pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0 and XGBoost 3.0.2. Feature code was taken from commit `b042815b116034da3ed1746fa54fd1d1ad22285d`.

These are results on simulated virtual subjects. No clinical validation, external-dataset test or reproduction of every original figure/correlation was established. Some exploratory correlation values differ from the paper; no values were adjusted to force agreement. See [source attribution](PUBLIC_DATA.md) and the [related publication](https://doi.org/10.1109/ACCESS.2025.3626252).
