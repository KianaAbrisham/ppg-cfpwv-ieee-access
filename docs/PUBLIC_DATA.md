# Public PWDB data

The converter uses **PWDB v0.2.0, Zenodo record 3275625**. PWDB contains simulated pulse waves for 4,374 virtual healthy adults. It is not a patient-recording dataset.

## Download and prepare

After installing the repository requirements, run from the repository root:

```bash
python tools/prepare_public_pwdb.py --source data/pwdb-source --output data/pwdb --download
```

The download is approximately 264 MB: `PWs_csv.zip`, `pwdb_haemod_params.csv`, and `pwdb_pw_indices.csv`. The converter verifies the published MD5 checksums and records SHA-256 hashes. Verified cached downloads are reused. Choose a new output directory for each conversion; existing converted data are protected.

## Verified mapping

| Source | Converted content |
| --- | --- |
| `csv/PWs_Digital_PPG.csv` | Digital waveforms, with the original `Subject Number` |
| `csv/PWs_Radial_PPG.csv` | Radial waveforms, with the original `Subject Number` |
| `csv/PWs_Brachial_PPG.csv` | Brachial waveforms, with the original `Subject Number` |
| `pwdb_haemod_params.csv` | `PWV_cf [m/s]` targets and six `age [years]` categories |
| `pwdb_pw_indices.csv` | Provided site-specific fiducials, SI, RI and `Age` |

Every table must contain each ID from 1 through 4,374 exactly once. Alignment uses these IDs, never assumed row correspondence. Age values in the two official metadata tables must agree. Waveform columns are checked in sample order; subject numbers are not model inputs.

The three sites each contain 4,374 waveforms sampled at **500 Hz**, with **337 to 487 samples** per pulse. Amplitude units are arbitrary. Conversion preserves waveform values and trailing empty cells and verifies a CSV round-trip. It performs no resampling, filtering, normalization or cropping. Neural-network runs use `--length 487` to pad shorter pulses to the longest source pulse, instead of the generic 2,000-sample default.

The six age labels are 25, 35, 45, 55, 65 and 75, with 729 subjects in each category. The cf-PWV targets span 4.5551 to 16.195 m/s. These age labels describe the simulation's age categories, not independently measured biological age.

The provided indices have missing required fiducials for 46 digital and 41 radial subjects. The feature pipeline retains 4,328 digital, 4,333 radial and 4,374 brachial subjects and records exclusions. All 4,374 waveforms per site remain valid for the neural-network pipelines. The converter uses the database's fiducials; it does not claim to redetect them.

## Output locations

- `data/pwdb/deep-learning/{digital,radial,brachial}/signals.csv`: subject IDs and sample columns.
- `data/pwdb/deep-learning/targets_cf.csv`: cf-PWV targets.
- `data/pwdb/deep-learning/targets_age.csv`: age-category labels.
- `data/pwdb/ieee/data/`: the seven input files for the feature-analysis project.
- `data/pwdb/conversion_manifest.json`: source checksums, mapping, units, counts and output hashes.

Data conversion alone does not establish reproduction of a paper's numerical results. Keep sites in separate runs and preserve subject-level train, validation and test separation.

## Source attribution

- Charlton and colleagues, [Pulse Wave Database v0.2](https://doi.org/10.5281/zenodo.3275625).
- [Official PWDB format documentation](https://github.com/peterhcharlton/pwdb/wiki/PWs).
- [Database publication and project](https://peterhcharlton.github.io/pwdb/).

The original data remain with their publisher and retain their original terms. The repository's software license does not relicense PWDB data.
