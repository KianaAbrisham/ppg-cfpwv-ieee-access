# PPG-based cf-PWV estimation (feature-based XGBoost)

Code supporting my IEEE Access paper (DOI: 10.1109/ACCESS.2025.3626252).

## What’s included
- Feature extraction from PPG/SDPPG
- Age correlation analysis + plots
- cf-PWV regression using XGBoost with cross-validation
- Diagnostics (predicted vs actual, residuals, Bland–Altman) + permutation importance

## Data
The dataset used in the paper is publicly available, but it is not redistributed in this repository.
Update the file paths at the top of `paper_code.py` to match your local files.

## Install
```bash
pip install -r requirements.txt
