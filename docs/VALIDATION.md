# Validation record

Updated 28 September 2026. The [public-PWDB evaluation](PUBLIC_DATA_VALIDATION.md) is complete. The earlier software checks below were performed on 27 September 2026, on Linux CPU with Python 3.12 and the core package versions pinned in `requirements.txt`.

Five regression tests passed:

1. Missing, duplicate, fractional or nonfinite subject IDs are rejected.
2. Waveform extraction retains the first sample, allows trailing padding and rejects internal gaps.
3. Targets align by unique subject IDs even when row order differs; missing/duplicate/invalid targets are rejected.
4. Artificial pulse feature extraction retains a valid subject, records an invalid waveform exclusion,
   computes half peak-to-peak amplitude and maps age by ID.
5. Five-fold XGBoost training on 30 artificial observations saves exactly one held-out prediction per subject
   across five folds. The regression diagnostic plotting code also executes; high-resolution figure file
   writes are suppressed in this test to keep it small.

The original seven study CSV exports remain unavailable. The verified public-source converter now recreates inputs from the official PWDB release, and a complete five-fold radial evaluation is recorded below. Reproduction of all age-analysis figures, physiological feature definitions against the original study exports, every published numerical result, and clinical/general-population validity remain unverified. The earlier artificial-fixture results above are software checks, separate from the new public-data results.

Changes since the earlier script include explicit IDs instead of positional fallback, one-to-one target
joins, retention of the first waveform sample, rejection of internal missing samples, durable exclusion
tables, out-of-fold subject IDs, corrected area/amplitude labels, and display of negative importance values.
These changes can alter the analyzed cohort or features relative to an older export; validate them against
the study data before comparing scientific results.

```bash
python -m unittest discover -s tests -v
```

## Public-source conversion verified — 2026-09-28

The official PWDB v0.2 waveform archive, haemodynamic targets and provided fiducials were downloaded and verified against publisher checksums. All 4,374 subject IDs were aligned explicitly, all waveform values passed a CSV round-trip check, and the target units and sampling rate were checked against the source documentation. Four additional tests verify shuffled-ID alignment and rejection of duplicate IDs, missing subjects and corrupt cached downloads. See [public data setup](PUBLIC_DATA.md) and the [conversion manifest](public_data/conversion_manifest.json). These checks validate data preparation; they do not establish numerical reproduction of a paper.

## Full public-data evaluation

A full public-PWDB five-fold radial evaluation is now complete, with all-site feature extraction, explicit exclusions and native checkpoint replay. See the [2026-09-28 report](PUBLIC_DATA_VALIDATION.md). This supersedes the earlier synthetic-only validation scope; it does not establish reproduction of every paper result.

## Continuous integration

The [Checks workflow](../.github/workflows/checks.yml) runs the existing five regression tests and four public-data integrity tests on Ubuntu with Python 3.12. These tests use fixtures; CI does not download PWDB or rerun the complete research evaluation.
