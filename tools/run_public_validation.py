"""Evaluate the current IEEE Access implementation on official public PWDB data.

Imports the repository's numerical feature extraction unchanged; records every
eligible subject, fold, metric and native XGBoost checkpoint. Large publication
layout figures are separate from this numerical validation.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source = a.repo / "paper_code.py"
    spec = importlib.util.spec_from_file_location("paper_code", source)
    paper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper)
    record = {
        "dataset": "PWDB v0.2, Zenodo 3275625",
        "status": "running",
        "paper_reproduction_verified": False,
        "code_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "feature_engine": "Unmodified build_feature_table, map_age, correlate_features and prepare_regression_data from paper_code.py",
        "regression": "5 folds; cf-PWV deciles; seed 42; 400 trees; learning_rate .05; max_depth 5; squared-error loss; n_jobs 1",
        "input_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(a.data.glob("*.csv"))
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    write_json(a.output / "run.json", record)
    sites = {}
    for site, idxname in [
        ("Digital", "digfeatures.csv"),
        ("Radial", "radfeature.csv"),
        ("Brachial", "brachfeatures.csv"),
    ]:
        print(f"Extracting {site}: all 4374 input subjects", flush=True)
        waves = paper._strip_columns(pd.read_csv(a.data / f"PWs_{site}_PPG.csv"))
        idx = paper._strip_columns(pd.read_csv(a.data / idxname))
        features = paper.build_feature_table(waves, idx, site)
        pd.DataFrame(
            features.attrs["exclusions"], columns=["Subject Number", "Reason"]
        ).to_csv(a.output / f"{site.lower()}_excluded.csv", index=False)
        features = paper.map_age(features, idx)
        features.to_csv(a.output / f"{site.lower()}_features.csv", index=False)
        sig, nonsig = paper.correlate_features(features)
        sig.to_csv(
            a.output / f"{site.lower()}_significant_correlations.csv", index=False
        )
        nonsig.to_csv(
            a.output / f"{site.lower()}_nonsignificant_correlations.csv", index=False
        )
        sites[site] = features
        print(f"{site}: {len(features)} retained", flush=True)
    merged = paper.prepare_regression_data(
        sites["Radial"], pd.read_csv(a.data / "PWV.csv")
    )
    X = (
        merged.drop(columns=["Subject Number", "Age", "cf_pwv"])
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )
    y = merged.cf_pwv.reset_index(drop=True)
    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    if bins.value_counts().min() < 5:
        raise ValueError("Insufficient subjects for decile-stratified fivefold CV.")
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows, predictions, splits, importance = [], [], [], []
    for fold, (tr, te) in enumerate(cv.split(X, bins), 1):
        print(f"XGBoost fold {fold}/5", flush=True)
        assert set(tr).isdisjoint(te)
        model = xgboost.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        score = {
            "fold": fold,
            "test_subjects": len(te),
            "mae_m_s": float(mean_absolute_error(y.iloc[te], pred)),
            "rmse_m_s": float(np.sqrt(mean_squared_error(y.iloc[te], pred))),
            "r2": float(r2_score(y.iloc[te], pred)),
            "baseline_rmse_m_s": float(
                np.sqrt(
                    mean_squared_error(y.iloc[te], np.full(len(te), y.iloc[tr].mean()))
                )
            ),
        }
        rows.append(score)
        predictions.extend(
            {
                "subject_id": int(merged.iloc[i]["Subject Number"]),
                "fold": fold,
                "actual_cfpwv_m_s": float(v),
                "predicted_cfpwv_m_s": float(z),
            }
            for i, v, z in zip(te, y.iloc[te], pred)
        )
        splits.append(
            {
                "fold": fold,
                "train": merged.iloc[tr]["Subject Number"].astype(int).tolist(),
                "test": merged.iloc[te]["Subject Number"].astype(int).tolist(),
            }
        )
        # Native serialization avoids the XGBoost 3.0.2 sklearn wrapper's
        # use of _estimator_type, which was removed in sklearn 1.8.
        checkpoint = a.output / f"fold_{fold}.ubj"
        model.get_booster().save_model(checkpoint)
        restored = xgboost.Booster()
        restored.load_model(checkpoint)
        np.testing.assert_array_equal(
            restored.predict(xgboost.DMatrix(X.iloc[te])), pred
        )
        importance.append(
            permutation_importance(
                model, X.iloc[te], y.iloc[te], n_repeats=10, random_state=42, n_jobs=1
            ).importances_mean
        )
        pd.DataFrame(rows).to_csv(a.output / "fold_metrics.csv", index=False)
        pd.DataFrame(predictions).to_csv(
            a.output / "held_out_predictions.csv", index=False
        )
        write_json(a.output / "splits.json", splits)
        print(json.dumps(score), flush=True)
    metrics = {
        key: {
            "mean": float(np.mean([r[key] for r in rows])),
            "sd_ddof0_paper_code": float(np.std([r[key] for r in rows], ddof=0)),
            "sd_ddof1": float(np.std([r[key] for r in rows], ddof=1)),
        }
        for key in ["mae_m_s", "rmse_m_s", "r2", "baseline_rmse_m_s"]
    }
    frame = pd.DataFrame(predictions)
    if frame.subject_id.duplicated().any() or set(frame.subject_id) != set(
        merged["Subject Number"]
    ):
        raise ValueError("OOF coverage failed.")
    diff = frame.predicted_cfpwv_m_s - frame.actual_cfpwv_m_s
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    record.update(
        status="completed",
        seconds=time.time() - started,
        feature_count=X.shape[1],
        retained_subjects={site.lower(): len(v) for site, v in sites.items()},
        missing_derived_feature_cells=int(X.isna().sum().sum()),
        missing_handling="Native XGBoost missing-value handling, as in repository code",
        metrics=metrics,
        checkpoint_validation="Native Booster reload predictions exactly match all original held-out predictions",
        bland_altman={
            "bias_m_s": bias,
            "lower_loa_m_s": bias - 1.96 * sd,
            "upper_loa_m_s": bias + 1.96 * sd,
        },
    )
    write_json(a.output / "run.json", record)
    pd.DataFrame(
        {
            "feature": X.columns,
            "mean_permutation_importance": np.mean(importance, axis=0),
        }
    ).sort_values("mean_permutation_importance", ascending=False).to_csv(
        a.output / "permutation_importance.csv", index=False
    )
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
