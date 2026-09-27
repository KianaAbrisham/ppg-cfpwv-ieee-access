# Validation record

Review date: 27 September 2026. Tested on Linux, Python 3.12, CPU, using the core package
versions pinned in `requirements.txt`.

Five regression tests passed:

1. Missing, duplicate, fractional or nonfinite subject IDs are rejected.
2. Waveform extraction retains the first sample, allows trailing padding and rejects internal gaps.
3. Targets align by unique subject IDs even when row order differs; missing/duplicate/invalid targets are rejected.
4. Artificial pulse feature extraction retains a valid subject, records an invalid waveform exclusion,
   computes half peak-to-peak amplitude and maps age by ID.
5. Five-fold XGBoost training on 30 artificial observations saves exactly one held-out prediction per subject
   across five folds. The regression diagnostic plotting code also executes; high-resolution figure file
   writes are suppressed in this test to keep it small.

The original seven study CSV exports are unavailable here. Full source-data conversion, all age-analysis
figures, physiological feature definitions against the original study inputs, numerical paper reproduction,
and clinical/general-population validity have **not** been established. Synthetic results are software
checks only and are not substitutes for the paper's reported metrics.

Changes since the earlier script include explicit IDs instead of positional fallback, one-to-one target
joins, retention of the first waveform sample, rejection of internal missing samples, durable exclusion
tables, out-of-fold subject IDs, corrected area/amplitude labels, and display of negative importance values.
These changes can alter the analyzed cohort or features relative to an older export; validate them against
the study data before comparing scientific results.

```bash
python -m unittest discover -s tests -v
```
