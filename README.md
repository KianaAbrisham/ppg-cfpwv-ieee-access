# PPG-based cf-PWV Estimation (Feature-based XGBoost)

Reproducible Python code supporting my IEEE Access paper:  
**DOI:** 10.1109/ACCESS.2025.3626252  
**IEEE Xplore:** https://ieeexplore.ieee.org/abstract/document/11218839

## What’s included
- Feature extraction from PPG/SDPPG
- Age correlation analysis (Pearson r, p-values) + plots
- cf-PWV regression using XGBoost with cross-validation
- Diagnostics: predicted vs actual, residuals, Bland–Altman
- Permutation importance

## Repository structure
- `paper_code.py` — main script
- `data/` — input CSV files (not tracked in git)
- `outputs/` — generated figures/tables (not tracked in git)

## Data
The dataset used in the paper is publicly available, but it is not redistributed in this repository.  
Place the required CSV files in `./data/` with the following filenames:

- `PWs_Digital_PPG.csv`
- `PWs_Radial_PPG.csv`
- `PWs_Brachial_PPG.csv`
- `digfeatures.csv`
- `radfeature.csv`
- `brachfeatures.csv`
- `PWV.csv`

## Install
```bash
pip install -r requirements.txt

## Run
```bash
python paper_code.py
