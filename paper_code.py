"""
paper_code.py

Reproducible pipeline for:
- PPG/SDPPG feature extraction
- Age correlation analysis (Pearson r, p-values)
- cf-PWV estimation using XGBoost (cross-validation + diagnostics)

Expected local files (NOT tracked in git) in ./data/
- PWs_Digital_PPG.csv
- PWs_Radial_PPG.csv
- PWs_Brachial_PPG.csv
- digfeatures.csv
- radfeature.csv
- brachfeatures.csv
- PWV.csv

Outputs will be saved in ./outputs/figures and ./outputs/tables (NOT tracked in git).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd

try:
    # Newer SciPy
    from scipy.integrate import simpson as _scipy_integrate
except Exception:
    # Older SciPy fallback
    from scipy.integrate import simps as _scipy_integrate

from scipy.stats import pearsonr, skew, kurtosis
from scipy.interpolate import interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.lines import Line2D

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor


# ========================== CONFIG & CONSTANTS ==========================
FS = 500                     # Sampling rate (Hz)
RANDOM_STATE = 42
ALPHA_P = 0.05               # significance threshold for correlations

# PIR parameters (definition agreed)
PIR_WINDOW_MS = 10           # Δt for PIR (ms) — FIXED across subjects

# Internal, descriptive names (stay in CSVs); plots map them to short labels
AMP_COL = "Amplitude (median amplitude; 0.5H)"  # plotted as "Amplitude"
PIR_COL = f"PIR (Peak-to-Instantaneous Ratio; Δt={PIR_WINDOW_MS} ms)"  # plotted as "PIR"

# ---------------------- FILE PATHS (edit as needed) ---------------------
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")

FIG_OUT_DIR = OUT_DIR / "figures"
TAB_OUT_DIR = OUT_DIR / "tables"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
TAB_OUT_DIR.mkdir(parents=True, exist_ok=True)

DIG_PWS_CSV = DATA_DIR / "PWs_Digital_PPG.csv"
RAD_PWS_CSV = DATA_DIR / "PWs_Radial_PPG.csv"
BRACH_PWS_CSV = DATA_DIR / "PWs_Brachial_PPG.csv"

DIG_IDX_CSV = DATA_DIR / "digfeatures.csv"
RAD_IDX_CSV = DATA_DIR / "radfeature.csv"
BRACH_IDX_CSV = DATA_DIR / "brachfeatures.csv"

PWV_CSV = DATA_DIR / "PWV.csv"


# ============================== DISPLAY ==============================
mpl.rcParams.update({
    "font.family": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.default": "it",
})


# ============================== DISPLAY NAMES (canonical) ==============================
DISPLAY_NAME = {
    # PPG features
    "AUC": "Area Under the Curve (AUC)",
    "S-AUC": "Systolic Area Under the Curve (S-AUC)",
    "D-AUC": "Diastolic Area Under the Curve (D-AUC)",
    "AUC Ratio": "AUC Ratio",
    "Rise Time": "Rise Time",
    "Decay Time": "Decay Time",
    "Rise–Decay Time Ratio": "Rise–Decay Time Ratio",
    "Amplitude (median amplitude; 0.5H)": "Amplitude",
    "Upslope Length": "Upslope Length",
    "Downslope Length": "Downslope Length",
    "Upslope": "Upslope",
    "Downslope": "Downslope",
    "Onset-End Slope": "Onset–End Slope",
    "Slope Ratio": "Slope Ratio",
    "Length-Height Ratio": "Length–Height Ratio",
    "Slope Length Ratio": "Slope–Length Ratio",
    "Upslope Length Ratio": "Upslope–Length Ratio",
    "Downslope Length Ratio": "Downslope–Length Ratio",
    "Start Datum Area": "Start Datum Area",
    "End Datum Area": "End Datum Area",
    "Datum Area Ratio": "Datum–Area Ratio",
    "Max Start Datum Diff.": "Maximum Start Datum Difference",
    "Max End Datum Diff.": "Maximum End Datum Difference",
    "Med. Start Datum Diff.": "Median Start Datum Difference",
    "Med. End Datum Diff.": "Median End Datum Difference",
    "Pulse Width": "Pulse Width",
    "Systolic Width": "Systolic Width",
    "Diastolic Width": "Diastolic Width",
    "Width Ratio": "Width Ratio",
    "Variance": "Variance",
    "Skew": "Skewness",
    "Kurtosis": "Kurtosis",
    f"PIR (Peak-to-Instantaneous Ratio; Δt={PIR_WINDOW_MS} ms)": "Peak-to-Instantaneous Ratio (PIR)",
    "SI": "Stiffness Index (SI)",
    "RI": "Reflection Index (RI)",

    # SDPPG amplitudes
    "b/a": r"$b/a$",
    "c/a": r"$c/a$",
    "d/a": r"$d/a$",
    "e/a": r"$e/a$",

    # SDPPG slopes
    "slope_a-b": r"$\mathrm{slope}_{a-b}$",
    "slope_a-c": r"$\mathrm{slope}_{a-c}$",
    "slope_a-d": r"$\mathrm{slope}_{a-d}$",
    "slope_a-e": r"$\mathrm{slope}_{a-e}$",
    "slope_b-c": r"$\mathrm{slope}_{b-c}$",
    "slope_b-d": r"$\mathrm{slope}_{b-d}$",
    "slope_b-e": r"$\mathrm{slope}_{b-e}$",
    "slope_c-e": r"$\mathrm{slope}_{c-e}$",
    "slope_d-e": r"$\mathrm{slope}_{d-e}$",

    # SDPPG intervals
    "t_a-b": r"$t_{a-b}$",
    "t_a-c": r"$t_{a-c}$",
    "t_a-d": r"$t_{a-d}$",
    "t_a-e": r"$t_{a-e}$",
    "t_b-c": r"$t_{b-c}$",
    "t_b-d": r"$t_{b-d}$",
    "t_b-e": r"$t_{b-e}$",
    "t_c-e": r"$t_{c-e}$",
    "t_c-d": r"$t_{c-d}$",
    "t_d-e": r"$t_{d-e}$",

    # SDPPG aging indices
    "AGI_int": r"$\mathrm{AGI}_{\mathrm{int}}$",
    "AGI_mod": r"$\mathrm{AGI}_{\mathrm{mod}}$",
}

USE_SHORT_LABELS = True

DISPLAY_NAME_SHORT = {
    "AUC": "AUC",
    "S-AUC": "S-AUC",
    "D-AUC": "D-AUC",
    "AUC Ratio": "AUC Ratio",
    "Rise Time": "Rise Time",
    "Decay Time": "Decay Time",
    "Rise–Decay Time Ratio": "Rise–Decay Time Ratio",
    "Amplitude (median amplitude; 0.5H)": "Amplitude",
    "Upslope Length": "Upslope Length",
    "Downslope Length": "Downslope Length",
    "Upslope": "Upslope",
    "Downslope": "Downslope",
    "Onset-End Slope": "Onset–End Slope",
    "Slope Ratio": "Slope Ratio",
    "Length-Height Ratio": "Length–Height Ratio",
    "Slope Length Ratio": "Slope–Length Ratio",
    "Upslope Length Ratio": "Upslope–Length Ratio",
    "Downslope Length Ratio": "Downslope–Length Ratio",
    "Start Datum Area": "Start Datum Area",
    "End Datum Area": "End Datum Area",
    "Datum Area Ratio": "Datum–Area Ratio",
    "Max Start Datum Diff.": "Max Start–Datum Diff.",
    "Max End Datum Diff.": "Max End–Datum Diff.",
    "Med. Start Datum Diff.": "Med. Start–Datum Diff.",
    "Med. End Datum Diff.": "Med. End–Datum Diff.",
    "Pulse Width": "Pulse Width",
    "Systolic Width": "Systolic Width",
    "Diastolic Width": "Diastolic Width",
    "Width Ratio": "Width Ratio",
    "Variance": "Variance",
    "Skew": "Skewness",
    "Kurtosis": "Kurtosis",
    f"PIR (Peak-to-Instantaneous Ratio; Δt={PIR_WINDOW_MS} ms)": "PIR",
    "SI": "SI",
    "RI": "RI",

    # TeX unchanged
    "b/a": r"$b/a$", "c/a": r"$c/a$", "d/a": r"$d/a$", "e/a": r"$e/a$",
    "slope_a-b": r"$\mathrm{slope}_{a-b}$",
    "slope_a-c": r"$\mathrm{slope}_{a-c}$",
    "slope_a-d": r"$\mathrm{slope}_{a-d}$",
    "slope_a-e": r"$\mathrm{slope}_{a-e}$",
    "slope_b-c": r"$\mathrm{slope}_{b-c}$",
    "slope_b-d": r"$\mathrm{slope}_{b-d}$",
    "slope_b-e": r"$\mathrm{slope}_{b-e}$",
    "slope_c-e": r"$\mathrm{slope}_{c-e}$",
    "slope_d-e": r"$\mathrm{slope}_{d-e}$",
    "t_a-b": r"$t_{a-b}$", "t_a-c": r"$t_{a-c}$", "t_a-d": r"$t_{a-d}$", "t_a-e": r"$t_{a-e}$",
    "t_b-c": r"$t_{b-c}$", "t_b-d": r"$t_{b-d}$", "t_b-e": r"$t_{b-e}$",
    "t_c-e": r"$t_{c-e}$", "t_c-d": r"$t_{c-d}$", "t_d-e": r"$t_{d-e}$",
    "AGI_int": r"$\mathrm{AGI}_{\mathrm{int}}$", "AGI_mod": r"$\mathrm{AGI}_{\mathrm{mod}}$",
}


def label_for(feature_key: str) -> str:
    if USE_SHORT_LABELS:
        return DISPLAY_NAME_SHORT.get(feature_key, DISPLAY_NAME.get(feature_key, feature_key))
    return DISPLAY_NAME.get(feature_key, feature_key)


# ============================== DISPLAY UNITS (by INTERNAL key) ==============================
DISPLAY_UNITS = {
    "AUC": "Amplitude (V)",
    "S-AUC": "Amplitude (V)",
    "D-AUC": "Amplitude (V)",
    "Start Datum Area": "Amplitude (V)",
    "End Datum Area": "Amplitude (V)",
    "Amplitude (median amplitude; 0.5H)": "Amplitude (V)",

    "Rise Time": "Time (s)",
    "Decay Time": "Time (s)",
    "Rise–Decay Time Ratio": "Ratio (a.u.)",
    "AUC Ratio": "Ratio (a.u.)",

    "Upslope Length": "Length (a.u.)",
    "Downslope Length": "Length (a.u.)",
    "Upslope": "Gradient (V/s)",
    "Downslope": "Gradient (V/s)",
    "Onset-End Slope": "Rate (V/s)",
    "Slope Ratio": "Ratio (a.u.)",
    "Length-Height Ratio": "Ratio (a.u.)",
    "Slope Length Ratio": "Ratio (a.u.)",
    "Upslope Length Ratio": "Ratio (a.u.)",
    "Downslope Length Ratio": "Ratio (a.u.)",
    "Datum Area Ratio": "Ratio (a.u.)",

    "Max Start Datum Diff.": "Amplitude (V)",
    "Max End Datum Diff.": "Amplitude (V)",
    "Med. Start Datum Diff.": "Amplitude (V)",
    "Med. End Datum Diff.": "Amplitude (V)",

    "Pulse Width": "Time (s)",
    "Systolic Width": "Time (s)",
    "Diastolic Width": "Time (s)",
    "Width Ratio": "Ratio (a.u.)",

    "Variance": "Variance (V²)",
    "Skew": "Skewness (std)",
    "Kurtosis": "Kurtosis",

    f"PIR (Peak-to-Instantaneous Ratio; Δt={PIR_WINDOW_MS} ms)": "Ratio (a.u.)",

    "SI": "Velocity (m/s)",
    "RI": "Ratio (a.u.)",

    "b/a": "Ratio (a.u.)",
    "c/a": "Ratio (a.u.)",
    "d/a": "Ratio (a.u.)",
    "e/a": "Ratio (a.u.)",

    "slope_a-b": "Gradient (V/s)",
    "slope_a-c": "Gradient (V/s)",
    "slope_a-d": "Gradient (V/s)",
    "slope_a-e": "Gradient (V/s)",
    "slope_b-c": "Gradient (V/s)",
    "slope_b-d": "Gradient (V/s)",
    "slope_b-e": "Gradient (V/s)",
    "slope_c-e": "Gradient (V/s)",
    "slope_d-e": "Gradient (V/s)",

    "t_a-b": "Time (s)",
    "t_a-c": "Time (s)",
    "t_a-d": "Time (s)",
    "t_a-e": "Time (s)",
    "t_b-c": "Time (s)",
    "t_b-d": "Time (s)",
    "t_b-e": "Time (s)",
    "t_c-e": "Time (s)",
    "t_c-d": "Time (s)",
    "t_d-e": "Time (s)",

    "AGI_int": "AGI (a.u.)",
    "AGI_mod": "AGI (a.u.)",
}


def unit_for(feature_key: str) -> str:
    return DISPLAY_UNITS.get(feature_key, "Value (unit)")


# ---------- Styling for figures ----------
LINE_LW = 6
MARKER_SZ = 11
TITLE_FS = 38
AXLABEL_FS = 34
TICK_FS = 33
LEGEND_FS = 40
POS = "#009E73"  # positive correlation
NEG = "#8B3E8F"  # negative correlation


# ============================== UTILITIES ==============================
def _require_files(paths: Sequence[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        msg = "\n".join(f"  - {p.as_posix()}" for p in missing)
        raise FileNotFoundError(
            "Missing input CSVs. Put the required files in ./data/ (see README) and try again:\n" + msg
        )


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _require_columns(df: pd.DataFrame, cols: Sequence[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = "\n".join(f"  - {c}" for c in missing)
        raise KeyError(f"{df_name} is missing required columns:\n{msg}")


def _integrate(y: np.ndarray, x: np.ndarray) -> float:
    """SciPy integration wrapper (supports both simpson and simps)."""
    try:
        return float(_scipy_integrate(y, x=x))
    except TypeError:
        return float(_scipy_integrate(y, x))


# ============================== HELPERS ==============================
def width_at_level(t, x, level) -> float:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    above = x >= level
    edges = np.where(np.diff(above.astype(int)) != 0)[0]
    if edges.size < 2:
        return np.nan

    i = edges[0]
    denom1 = (x[i + 1] - x[i])
    if denom1 == 0:
        return np.nan
    t1 = t[i] + (level - x[i]) * (t[i + 1] - t[i]) / denom1

    j = edges[-1]
    denom2 = (x[j + 1] - x[j])
    if denom2 == 0:
        return np.nan
    t2 = t[j] + (level - x[j]) * (t[j + 1] - t[j]) / denom2

    return float(t2 - t1)


def calculate_pir(signal: np.ndarray, fs: int = FS, window_ms: int = PIR_WINDOW_MS) -> float:
    """
    Peak-to-Instantaneous Ratio using a fixed time window:
      PIR = |x_peak| / |x(peak - Δt)|  where Δt = window_ms
    """
    x = np.asarray(signal, dtype=float)
    if x.size < 2 or not np.isfinite(x).any():
        return np.nan

    peak_idx = int(np.nanargmax(x))
    delta = int(round(fs * (window_ms / 1000.0)))
    inst_idx = peak_idx - delta
    if inst_idx < 0:
        return np.nan

    num = float(np.abs(x[peak_idx]))
    den = float(np.abs(x[inst_idx]))
    if not np.isfinite(den) or den == 0:
        return np.nan
    return float(num / den)


def get_indices_row(indices_df: pd.DataFrame, subject_number: int) -> pd.Series:
    cols = [c.strip() for c in indices_df.columns]
    if "Subject Number" in cols:
        tmp = indices_df.copy()
        tmp.columns = cols
        row = tmp.loc[tmp["Subject Number"] == subject_number]
        if row.empty:
            raise KeyError(f"Subject {subject_number} not found in indices file.")
        return row.iloc[0]
    # fallback: assume row order matches subject_number starting from 1
    return indices_df.iloc[int(subject_number) - 1]


# =========================== FEATURE BUILDING ===========================
def build_feature_table(pws_df: pd.DataFrame, indices_df: pd.DataFrame, site_prefix: str) -> pd.DataFrame:
    pws_df = _strip_columns(pws_df)
    indices_df = _strip_columns(indices_df)

    _require_columns(pws_df, ["Subject Number"], f"{site_prefix} PWs")
    _require_columns(indices_df, ["Age"], f"{site_prefix} indices")

    rows: List[dict] = []

    # Required fields to declare an index row "complete"
    req = [
        f"{site_prefix}_PPGsys_V", f"{site_prefix}_PPGsys_T",
        f"{site_prefix}_PPGdia_V", f"{site_prefix}_PPGdia_T",
        f"{site_prefix}_PPGa_V",   f"{site_prefix}_PPGa_T",
        f"{site_prefix}_PPGb_V",   f"{site_prefix}_PPGb_T",
        f"{site_prefix}_PPGc_V",   f"{site_prefix}_PPGc_T",
        f"{site_prefix}_PPGd_V",   f"{site_prefix}_PPGd_T",
        f"{site_prefix}_PPGe_V",   f"{site_prefix}_PPGe_T",
        f"{site_prefix}_SI",       f"{site_prefix}_RI",
    ]
    _require_columns(
        indices_df,
        req + (["Subject Number"] if "Subject Number" in indices_df.columns else []),
        f"{site_prefix} indices",
    )

    for subject_number in pws_df["Subject Number"].unique():
        try:
            idx = get_indices_row(indices_df, int(subject_number))

            # Pre-screen completeness
            if any(pd.isna(idx.get(c)) for c in req):
                continue

            # Load waveform (single pulse)
            row_sig = pws_df.loc[pws_df["Subject Number"] == subject_number].iloc[0]
            x = row_sig.iloc[2:].dropna().astype(float).to_numpy()
            if x.size < 5:
                continue
            t = np.arange(x.size, dtype=float) / FS

            # Indices
            sys_v = float(idx[f"{site_prefix}_PPGsys_V"])
            sys_t = float(idx[f"{site_prefix}_PPGsys_T"])
            dia_v = float(idx[f"{site_prefix}_PPGdia_V"])
            dia_t = float(idx[f"{site_prefix}_PPGdia_T"])

            sys_mask = t <= sys_t
            dia_mask = t >= sys_t
            x_sys, t_sys = x[sys_mask], t[sys_mask]
            x_dia, t_dia = x[dia_mask], t[dia_mask]

            # Areas
            s_auc = _integrate(x_sys, t_sys)
            d_auc = _integrate(x_dia, t_dia)
            auc = _integrate(x, t)
            auc_ratio = s_auc / d_auc if np.isfinite(d_auc) and d_auc != 0 else np.nan

            # Timing
            rise_time = float(sys_t - t[0])
            decay_time = float(t[-1] - sys_t)
            rise_decay_ratio = decay_time / rise_time if np.isfinite(rise_time) and rise_time != 0 else np.nan

            # Amplitude & widths
            full_amp = float(np.nanmax(x) - np.nanmin(x))
            half_level = float(np.nanmin(x) + full_amp / 2)
            half_peak_amp = float(full_amp / 2)
            pulse_width = width_at_level(t, x, half_level)

            # Phase-restricted widths
            try:
                f_sys = interp1d(t_sys, x_sys, kind="linear", bounds_error=True)
                sys_candidates = [tt for tt in t_sys if f_sys(tt) >= half_level]
                sys_width = float(sys_candidates[-1] - sys_candidates[0]) if sys_candidates else np.nan
            except Exception:
                sys_width = np.nan

            try:
                f_dia = interp1d(t_dia, x_dia, kind="linear", bounds_error=True)
                dia_candidates = [tt for tt in t_dia if f_dia(tt) >= half_level]
                dia_width = float(dia_candidates[-1] - dia_candidates[0]) if dia_candidates else np.nan
            except Exception:
                dia_width = np.nan

            width_ratio = sys_width / dia_width if np.isfinite(dia_width) and dia_width != 0 else np.nan

            # Slopes/lengths
            l_up = float(np.hypot(sys_t - t[0], sys_v - x[0]))
            l_down = float(np.hypot(t[-1] - sys_t, x[-1] - sys_v))
            slope_up = (sys_v - x[0]) / rise_time if np.isfinite(rise_time) and rise_time != 0 else np.nan
            slope_down = (x[-1] - sys_v) / decay_time if np.isfinite(decay_time) and decay_time != 0 else np.nan
            onset_end_slope = (x[-1] - x[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) != 0 else np.nan
            slope_ratio = slope_up / slope_down if np.isfinite(slope_down) and slope_down != 0 else np.nan
            length_height_ratio = (t[-1] - t[0]) / full_amp if np.isfinite(full_amp) and full_amp != 0 else np.nan
            slope_length_ratio = l_up / l_down if np.isfinite(l_down) and l_down != 0 else np.nan
            total_len = l_up + l_down
            up_len_ratio = l_up / total_len if total_len != 0 else np.nan
            down_len_ratio = l_down / total_len if total_len != 0 else np.nan

            # Datum-line features
            if sys_t != t_sys[0]:
                start_vals = x[0] + (t_sys - t_sys[0]) / (sys_t - t_sys[0]) * (sys_v - x[0])
            else:
                start_vals = np.full_like(x_sys, x[0])
            start_diff = np.abs(start_vals - x_sys)
            start_area = _integrate(start_diff, t_sys)
            max_start_diff = float(np.nanmax(start_diff))
            med_start_diff = float(np.nanmedian(start_diff))

            if (t_dia[-1] - sys_t) != 0:
                end_vals = sys_v + (t_dia - sys_t) / (t_dia[-1] - sys_t) * (x[-1] - sys_v)
            else:
                end_vals = np.full_like(x_dia, sys_v)
            end_diff = np.abs(end_vals - x_dia)
            end_area = _integrate(end_diff, t_dia)
            max_end_diff = float(np.nanmax(end_diff))
            med_end_diff = float(np.nanmedian(end_diff))
            datum_area_ratio = start_area / end_area if np.isfinite(end_area) and end_area != 0 else np.nan

            # Stats & PIR
            var = float(np.nanvar(x))
            skw = float(skew(x, nan_policy="omit"))
            kurtv = float(kurtosis(x, nan_policy="omit"))
            pir = calculate_pir(x)

            # SI, RI
            si = float(idx[f"{site_prefix}_SI"])
            ri = float(idx[f"{site_prefix}_RI"])

            # SDPPG a–e
            aV, bV, cV, dV, eV = (
                float(idx[f"{site_prefix}_PPGa_V"]), float(idx[f"{site_prefix}_PPGb_V"]),
                float(idx[f"{site_prefix}_PPGc_V"]), float(idx[f"{site_prefix}_PPGd_V"]),
                float(idx[f"{site_prefix}_PPGe_V"]),
            )
            aT, bT, cT, dT, eT = (
                float(idx[f"{site_prefix}_PPGa_T"]), float(idx[f"{site_prefix}_PPGb_T"]),
                float(idx[f"{site_prefix}_PPGc_T"]), float(idx[f"{site_prefix}_PPGd_T"]),
                float(idx[f"{site_prefix}_PPGe_T"]),
            )

            def safe_slope(v1, v2, t1, t2):
                return (v2 - v1) / (t2 - t1) if (t2 - t1) != 0 else np.nan

            rows.append({
                "Subject Number": int(subject_number),

                "AUC": auc, "S-AUC": s_auc, "D-AUC": d_auc, "AUC Ratio": auc_ratio,
                "Rise Time": rise_time, "Decay Time": decay_time,
                "Rise–Decay Time Ratio": rise_decay_ratio,

                AMP_COL: half_peak_amp,

                "Upslope Length": l_up, "Downslope Length": l_down,
                "Upslope": slope_up, "Downslope": slope_down, "Onset-End Slope": onset_end_slope,
                "Slope Ratio": slope_ratio,
                "Length-Height Ratio": length_height_ratio,
                "Slope Length Ratio": slope_length_ratio,
                "Upslope Length Ratio": up_len_ratio,
                "Downslope Length Ratio": down_len_ratio,

                "Start Datum Area": start_area, "End Datum Area": end_area,
                "Datum Area Ratio": datum_area_ratio,
                "Max Start Datum Diff.": max_start_diff, "Max End Datum Diff.": max_end_diff,
                "Med. Start Datum Diff.": med_start_diff, "Med. End Datum Diff.": med_end_diff,

                "Pulse Width": pulse_width, "Systolic Width": sys_width, "Diastolic Width": dia_width,
                "Width Ratio": width_ratio,

                "Variance": var, "Skew": skw, "Kurtosis": kurtv,
                PIR_COL: pir,

                "SI": si, "RI": ri,

                "b/a": bV / aV if aV != 0 else np.nan,
                "c/a": cV / aV if aV != 0 else np.nan,
                "d/a": dV / aV if aV != 0 else np.nan,
                "e/a": eV / aV if aV != 0 else np.nan,

                "slope_a-b": safe_slope(aV, bV, aT, bT),
                "slope_a-c": safe_slope(aV, cV, aT, cT),
                "slope_a-d": safe_slope(aV, dV, aT, dT),
                "slope_a-e": safe_slope(aV, eV, aT, eT),
                "slope_b-c": safe_slope(bV, cV, bT, cT),
                "slope_b-d": safe_slope(bV, dV, bT, dT),
                "slope_b-e": safe_slope(bV, eV, bT, eT),
                "slope_c-e": safe_slope(cV, eV, cT, eT),
                "slope_d-e": safe_slope(dV, eV, dT, eT),

                "t_a-b": bT - aT, "t_a-c": cT - aT, "t_a-d": dT - aT, "t_a-e": eT - aT,
                "t_b-c": cT - bT, "t_b-d": dT - bT, "t_b-e": eT - bT,
                "t_c-e": eT - cT, "t_c-d": dT - cT, "t_d-e": eT - dT,

                "AGI_int": (bV - eV) / aV if aV != 0 else np.nan,
                "AGI_mod": (bV - cV - dV) / aV if aV != 0 else np.nan,
            })

        except Exception as e:
            print(f"[{site_prefix}] Subject {subject_number}: {e}")

    return pd.DataFrame(rows)


def map_age(df_features: pd.DataFrame, indices_df: pd.DataFrame) -> pd.DataFrame:
    indices_df = _strip_columns(indices_df)
    if "Age" not in indices_df.columns:
        raise KeyError("Indices file must contain an 'Age' column.")

    df = df_features.copy()
    if "Subject Number" in indices_df.columns:
        age_map = dict(zip(indices_df["Subject Number"], indices_df["Age"]))
    else:
        age_map = dict(zip(np.arange(1, len(indices_df) + 1), indices_df["Age"]))

    df["Age"] = df["Subject Number"].map(age_map)
    return df


# ============================== AGE ANALYSIS ==============================
def correlate_features(
    df: pd.DataFrame,
    alpha: float = ALPHA_P,
    extra_drop: Sequence[str] = ("Subject Number",),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (sig_df, nonsig_df), both sorted by |correlation| desc.
    """
    drop = {"Age", *extra_drop}
    cols = [c for c in df.columns if c not in drop]
    rows = []
    x = df["Age"]

    for f in cols:
        y = df[f]
        m = x.notna() & y.notna()
        if m.sum() < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
            continue
        r, p = pearsonr(x[m], y[m])
        rows.append((f, r, p))

    out = pd.DataFrame(rows, columns=["Feature", "Correlation", "P-Value"])
    out = out.sort_values(by="Correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    sig = out.loc[out["P-Value"] < alpha].reset_index(drop=True)
    nonsig = out.loc[out["P-Value"] >= alpha].reset_index(drop=True)
    return sig, nonsig


def report_removed(nonsig_df: pd.DataFrame, artery_label: str, outdir: Path = TAB_OUT_DIR, alpha: float = ALPHA_P) -> None:
    n = len(nonsig_df)
    print(f"[{artery_label}] Non-significant features removed from bar plot (p >= {alpha}): {n}")
    if n == 0:
        return

    for _, row in nonsig_df.sort_values("P-Value").iterrows():
        feat = row["Feature"]
        pval = row["P-Value"]
        r = row["Correlation"]
        print(f"  {feat}: p={pval:.3g}, r={r:+.2f}")

    fname = f"corr_{artery_label.lower().replace(' ', '_')}_nonsignificant_with_p.txt"
    with open(outdir / fname, "w", encoding="utf-8") as f:
        f.write(f"{artery_label} — Non-significant features (p >= {alpha})\n")
        f.write("Feature\tp-value\tcorrelation\n")
        for _, row in nonsig_df.sort_values("P-Value").iterrows():
            f.write(f"{row['Feature']}\t{row['P-Value']:.6f}\t{row['Correlation']:+.6f}\n")


# ============================== PLOTTING ==============================
def plot_age_trajectories(
    comb_dig: pd.DataFrame,
    comb_rad: pd.DataFrame,
    comb_bra: pd.DataFrame,
    outdir: Path = FIG_OUT_DIR,
) -> None:
    """Features vs. Age line plots across Digital/Radial/Brachial."""
    features = [c for c in comb_dig.columns if c not in ("Subject Number", "Age")]
    ages = sorted(comb_dig["Age"].dropna().unique())

    ncols = 5
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(35, nrows * 4.0))
    axes = axes.flatten()

    styles = [
        dict(marker="o", linestyle="-",  linewidth=LINE_LW, markersize=MARKER_SZ, label="Digital"),
        dict(marker="s", linestyle="--", linewidth=LINE_LW, markersize=MARKER_SZ, label="Radial"),
        dict(marker="^", linestyle=":",  linewidth=LINE_LW, markersize=MARKER_SZ, label="Brachial"),
    ]

    for i, f in enumerate(features):
        ax = axes[i]
        dig = comb_dig.groupby("Age")[f].mean()
        rad = comb_rad.groupby("Age")[f].mean()
        bra = comb_bra.groupby("Age")[f].mean()

        ax.plot(dig.index, dig.values, **styles[0])
        ax.plot(rad.index, rad.values, **styles[1])
        ax.plot(bra.index, bra.values, **styles[2])

        title_txt = label_for(f)
        is_math = isinstance(title_txt, str) and title_txt.startswith("$") and title_txt.endswith("$")
        fs = TITLE_FS + (5 if is_math else 0)

        ax.set_title(title_txt, fontsize=fs, pad=18)
        ax.set_xlabel("Age", fontsize=AXLABEL_FS)
        ax.set_ylabel(unit_for(f), fontsize=AXLABEL_FS, labelpad=8)

        ax.set_xticks(ages)
        ax.tick_params(axis="x", labelsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)

        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    fig.subplots_adjust(top=1.0, bottom=0.11, left=0.05, right=0.995, hspace=1.6, wspace=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.038),
               ncol=3, fontsize=LEGEND_FS, frameon=False)

    fig.savefig(outdir / "Figure_features_vs_age.png", dpi=600, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(outdir / "Figure_features_vs_age.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_corr_bars(
    df_sorted: pd.DataFrame,
    label: str,
    artery_name: str,
    outdir: Path = FIG_OUT_DIR,
    save_basename: Optional[str] = None,
) -> None:
    """Plots ONLY significant features (p < ALPHA_P)."""
    if df_sorted.empty:
        print(f"({label}) {artery_name}: no significant features (p<{ALPHA_P}).")
        return

    r_signed = df_sorted["Correlation"].to_numpy()
    r_abs = np.abs(r_signed)
    feats = df_sorted["Feature"].to_numpy()
    feats_disp = [label_for(name) for name in feats]
    y = np.arange(len(df_sorted))
    colors = np.where(r_signed >= 0, POS, NEG)

    fig, ax = plt.subplots(figsize=(10, 12), dpi=600)
    H = 0.92
    bars = ax.barh(y, r_abs, color=colors, height=H)

    ax.set_yticks(y)
    ax.set_yticklabels(feats_disp, fontsize=12)
    ax.set_ylim(len(df_sorted) - 0.44, -0.56)
    ax.set_xlim(0, r_abs.max() + 0.10)

    for bar, r in zip(bars, r_signed):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{r:+.2f}", ha="left", va="center", fontsize=12, clip_on=False)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.grid(False)
    ax.text(0.5, -0.035, f"({label}) {artery_name}",
            transform=ax.transAxes, ha="center", va="top", fontsize=14)

    fig.subplots_adjust(left=0.30, right=0.98, top=0.96, bottom=0.08)

    if save_basename:
        fig.savefig(outdir / f"{save_basename}.png", dpi=600, bbox_inches="tight", pad_inches=0.1)
        fig.savefig(outdir / f"{save_basename}.pdf", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ============================== MODELING (Radial) ==============================
def train_cf_pwv_model(combined_results_rad: pd.DataFrame, pwv_csv: Path = PWV_CSV) -> None:
    pwv_df = _strip_columns(pd.read_csv(pwv_csv))
    _require_columns(pwv_df, ["Subject Number", "PWV_cf [m/s]"], "PWV.csv")

    merged_df = (
        pd.merge(
            combined_results_rad.copy(),
            pwv_df[["Subject Number", "PWV_cf [m/s]"]],
            on="Subject Number",
            how="inner",
        )
        .rename(columns={"PWV_cf [m/s]": "cf_pwv"})
    )

    X = (
        merged_df
        .drop(columns=["Subject Number", "Age", "cf_pwv"], errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
    )
    y = pd.to_numeric(merged_df["cf_pwv"], errors="coerce")

    mask = y.notna() & np.isfinite(y)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    def make_strat_bins(y_series, n_splits=5, max_bins=10, min_bins=3):
        """Quantile bins for stratified CV (ensures ≥ n_splits samples per bin)."""
        y_series = pd.Series(y_series).reset_index(drop=True)
        for q in range(max_bins, min_bins - 1, -1):
            try:
                bins = pd.qcut(y_series, q=q, labels=False, duplicates="drop")
                vc = pd.Series(bins).value_counts()
                if (vc >= n_splits).all() and vc.index.nunique() >= 2:
                    return bins
            except Exception:
                continue
        return None

    n_splits = 5
    y_bins = make_strat_bins(y, n_splits=n_splits)
    if y_bins is not None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iter = cv.split(X, y_bins)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iter = cv.split(X)

    mae_list, rmse_list, r2_list = [], [], []
    fold_y_test, fold_y_pred, fold_residuals = [], [], []
    perm_importances = []
    per_fold_rows = []

    for i, (tr, te) in enumerate(split_iter, start=1):
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]

        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,  # deterministic
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        per_fold_rows.append({"Fold": i, "MAE": mae, "RMSE": rmse, "R2": r2})

        fold_y_test.append(y_test.values)
        fold_y_pred.append(y_pred)
        fold_residuals.append(y_test.values - y_pred)

        pi = permutation_importance(
            model, X_test, y_test,
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
        )
        perm_importances.append(pi.importances_mean)

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_print = per_fold_df.copy()
    for col in ["MAE", "RMSE", "R2"]:
        per_fold_print[col] = per_fold_print[col].map(lambda v: f"{v:.3f}")

    summary = {
        "Fold": "Mean ± SD",
        "MAE": f"{np.mean(mae_list):.3f} ± {np.std(mae_list):.3f}",
        "RMSE": f"{np.mean(rmse_list):.3f} ± {np.std(rmse_list):.3f}",
        "R2": f"{np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}",
    }

    print("\nPer-fold evaluation (Radial → cf-PWV):")
    print(per_fold_print.to_string(index=False))
    print("\nSummary:")
    print(summary)

    per_fold_df.to_csv(TAB_OUT_DIR / "model_radial_per_fold_metrics.csv", index=False, float_format="%.3f")
    with open(TAB_OUT_DIR / "model_radial_summary.txt", "w", encoding="utf-8") as f:
        f.write("Per-fold evaluation (Radial → cf-PWV)\n")
        f.write(per_fold_df.to_csv(index=False, float_format="%.3f"))
        f.write("\nSummary:\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    # ============================== DIAGNOSTICS PLOTS ==============================
    all_actual = np.concatenate(fold_y_test)
    all_pred = np.concatenate(fold_y_pred)
    pad = 0.5
    xlims_common = (all_actual.min() - pad, all_actual.max() + pad)
    lim_both = (min(xlims_common[0], all_pred.min() - pad),
                max(xlims_common[1], all_pred.max() + pad))

    # Style constants
    fold_colors = ["#56B4E9", "#009E73", "#CC79A7", "#A65628", "#0072B2"]
    fold_markers = ["o"] * 5
    ideal_line_color = "#E69F00"
    edge_color = "k"
    edge_lw = 0.9
    point_size_pred = 44
    alpha_pred = 0.70
    point_size_res = 38
    alpha_res = 0.65

    # A. Predicted vs Actual
    fig, ax = plt.subplots(figsize=(5.8, 5.8), dpi=600)
    ax.plot(lim_both, lim_both, ls="--", c=ideal_line_color, lw=1.8, label="Ideal prediction")
    for i, (yt, yp) in enumerate(zip(fold_y_test, fold_y_pred)):
        ax.scatter(yt, yp, s=point_size_pred, c=fold_colors[i], marker=fold_markers[i],
                   alpha=alpha_pred, edgecolors=edge_color, linewidths=edge_lw, label=f"Fold {i + 1}")
    ax.set_xlim(lim_both)
    ax.set_ylim(lim_both)
    ax.set_xlabel("Actual cf-PWV [m/s]")
    ax.set_ylabel("Predicted cf-PWV [m/s]")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=True, fontsize=13)
    fig.savefig(FIG_OUT_DIR / "predicted_vs_actual_cfPWV.png", dpi=600, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(FIG_OUT_DIR / "predicted_vs_actual_cfPWV.pdf", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # B. Residuals vs Actual  (Residual = Actual − Predicted)
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=600)
    ax.axhline(0.0, c=ideal_line_color, ls="--", lw=1.8, label="Zero residual")
    for i, (yt, yp) in enumerate(zip(fold_y_test, fold_y_pred)):
        res = yt - yp
        ax.scatter(yt, res, s=point_size_res, c=fold_colors[i], marker=fold_markers[i],
                   alpha=alpha_res, edgecolors=edge_color, linewidths=edge_lw, label=f"Fold {i + 1}")
    all_res = np.concatenate(fold_residuals)
    rng = np.max(np.abs(all_res)) * 1.1 if all_res.size else 1.0
    ax.set_ylim(-rng, rng)
    ax.set_xlim(xlims_common)
    ax.set_xlabel("Actual cf-PWV [m/s]")
    ax.set_ylabel("Residual (Actual − Predicted) [m/s]")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=True, fontsize=13)
    fig.savefig(FIG_OUT_DIR / "residuals_vs_actual_cfPWV.png", dpi=600, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(FIG_OUT_DIR / "residuals_vs_actual_cfPWV.pdf", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # C. Bland–Altman (Difference = Predicted − Reference)
    y_true_all = np.concatenate(fold_y_test)
    y_pred_all = np.concatenate(fold_y_pred)
    mean_vals = (y_pred_all + y_true_all) / 2.0
    diff_vals = (y_pred_all - y_true_all)

    bias = float(diff_vals.mean())
    sd = float(diff_vals.std(ddof=1))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    print("\nBland–Altman (Predicted − Reference) summary:")
    print(f"  n               : {diff_vals.size:d}")
    print(f"  Bias (mean diff): {bias:+.3f} m/s")
    print(f"  SD of diff      : {sd:.3f} m/s")
    print(f"  95% LoA         : [{loa_lower:+.3f}, {loa_upper:+.3f}] m/s")
    try:
        r_pb, p_pb = pearsonr(mean_vals, diff_vals)
        print(f"  Proportional bias: r = {r_pb:+.3f}, p = {p_pb:.3g}")
    except Exception:
        r_pb, p_pb = np.nan, np.nan

    ba_df = pd.DataFrame({
        "n": [diff_vals.size],
        "Bias_mean_diff_m_per_s": [bias],
        "SD_diff_m_per_s": [sd],
        "LoA_lower_m_per_s": [loa_lower],
        "LoA_upper_m_per_s": [loa_upper],
        "Proportional_bias_r": [r_pb],
        "Proportional_bias_p": [p_pb],
    })
    ba_df.to_csv(TAB_OUT_DIR / "bland_altman_summary.csv", index=False, float_format="%.3f")
    with open(TAB_OUT_DIR / "bland_altman_summary.txt", "w", encoding="utf-8") as f:
        f.write("Bland–Altman (Predicted − Reference) summary\n")
        f.write(f"n: {diff_vals.size}\n")
        f.write(f"Bias (mean diff): {bias:+.3f} m/s\n")
        f.write(f"SD of diff: {sd:.3f} m/s\n")
        f.write(f"95% LoA: [{loa_lower:+.3f}, {loa_upper:+.3f}] m/s\n")
        if np.isfinite(r_pb):
            f.write(f"Proportional bias: r = {r_pb:+.3f}, p = {p_pb:.3g}\n")

    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=600)
    for i, (yt, yp) in enumerate(zip(fold_y_test, fold_y_pred)):
        m = (yp + yt) / 2.0
        d = (yp - yt)
        ax.scatter(m, d, s=point_size_res, c=fold_colors[i], marker=fold_markers[i],
                   alpha=alpha_res, edgecolors=edge_color, linewidths=edge_lw, label=f"Fold {i + 1}")
    xpad = 0.5
    xlims = (float(mean_vals.min() - xpad), float(mean_vals.max() + xpad))
    ax.set_xlim(xlims)
    ax.fill_between([xlims[0], xlims[1]], loa_lower, loa_upper, color="0.7", alpha=0.12, zorder=0)
    ax.axhline(bias, color=ideal_line_color, lw=1.8, label=f"Bias = {bias:+.3f} m/s")
    ax.axhline(loa_upper, color=ideal_line_color, lw=1.8, ls="--")
    ax.axhline(loa_lower, color=ideal_line_color, lw=1.8, ls="--")
    ax.set_xlabel("Mean of predicted and reference cf-PWV [m/s]")
    ax.set_ylabel("Difference (Predicted − Reference) [m/s]")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=ideal_line_color, lw=1.8, ls="--"))
    labels.append("95% LoA")
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.03),
              ncol=3, frameon=True, fontsize=13)
    fig.savefig(FIG_OUT_DIR / "bland_altman_cfPWV.png", dpi=600, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(FIG_OUT_DIR / "bland_altman_cfPWV.pdf", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # D. Permutation importance (Top-10)
    perm_importances = np.array(perm_importances)
    perm_mean = perm_importances.mean(axis=0)
    fi_df = (
        pd.DataFrame({"Feature": X.columns, "Importance": perm_mean})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    fi_df.to_csv(TAB_OUT_DIR / "permutation_importance_all_features.csv", index=False, float_format="%.6f")

    topk = 10
    fi_top = fi_df.head(topk).iloc[::-1]  # reverse for barh
    fi_labels_disp = [label_for(k) for k in fi_top["Feature"]]

    fig, ax = plt.subplots(figsize=(8.6, 3.5), dpi=600)
    ax.barh(fi_labels_disp, fi_top["Importance"], color="#4C78A8", edgecolor="none", height=0.68)
    ax.set_xlabel("Permutation importance", fontsize=13, labelpad=6)
    ax.set_ylabel("Feature", fontsize=12, labelpad=8)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.set_xlim(0, float(fi_top["Importance"].max()) * 1.06 if len(fi_top) else 1.0)
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.22, top=0.96)
    fig.savefig(FIG_OUT_DIR / "feature_importance_cfPWV.png", dpi=600, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(FIG_OUT_DIR / "feature_importance_cfPWV.pdf", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ============================== MAIN ==============================
def main() -> None:
    _require_files([DIG_PWS_CSV, RAD_PWS_CSV, BRACH_PWS_CSV, DIG_IDX_CSV, RAD_IDX_CSV, BRACH_IDX_CSV, PWV_CSV])

    dfdig = _strip_columns(pd.read_csv(DIG_PWS_CSV))
    dfrad = _strip_columns(pd.read_csv(RAD_PWS_CSV))
    dfbra = _strip_columns(pd.read_csv(BRACH_PWS_CSV))

    idx_dig = _strip_columns(pd.read_csv(DIG_IDX_CSV))
    idx_rad = _strip_columns(pd.read_csv(RAD_IDX_CSV))
    idx_bra = _strip_columns(pd.read_csv(BRACH_IDX_CSV))

    combined_results_dig = build_feature_table(dfdig, idx_dig, "Digital")
    combined_results_rad = build_feature_table(dfrad, idx_rad, "Radial")
    combined_results_brach = build_feature_table(dfbra, idx_bra, "Brachial")

    combined_results_dig = map_age(combined_results_dig, idx_dig)
    combined_results_rad = map_age(combined_results_rad, idx_rad)
    combined_results_brach = map_age(combined_results_brach, idx_bra)

    combined_results_dig.to_csv(TAB_OUT_DIR / "combined_result_dig.csv", index=False)
    combined_results_rad.to_csv(TAB_OUT_DIR / "combined_result_rad.csv", index=False)
    combined_results_brach.to_csv(TAB_OUT_DIR / "combined_result_brach.csv", index=False)

    sig_dig, nonsig_dig = correlate_features(combined_results_dig)
    sig_rad, nonsig_rad = correlate_features(combined_results_rad)
    sig_brach, nonsig_brach = correlate_features(combined_results_brach)

    sig_dig.to_csv(TAB_OUT_DIR / "corr_digital_significant.csv", index=False)
    sig_rad.to_csv(TAB_OUT_DIR / "corr_radial_significant.csv", index=False)
    sig_brach.to_csv(TAB_OUT_DIR / "corr_brachial_significant.csv", index=False)

    nonsig_dig.to_csv(TAB_OUT_DIR / "corr_digital_nonsignificant_removed.csv", index=False)
    nonsig_rad.to_csv(TAB_OUT_DIR / "corr_radial_nonsignificant_removed.csv", index=False)
    nonsig_brach.to_csv(TAB_OUT_DIR / "corr_brachial_nonsignificant_removed.csv", index=False)

    report_removed(nonsig_dig, "Digital")
    report_removed(nonsig_rad, "Radial")
    report_removed(nonsig_brach, "Brachial")

    plot_age_trajectories(combined_results_dig, combined_results_rad, combined_results_brach)
    plot_corr_bars(sig_dig, "a", "Digital", save_basename="corr_digital")
    plot_corr_bars(sig_rad, "b", "Radial", save_basename="corr_radial")
    plot_corr_bars(sig_brach, "c", "Brachial", save_basename="corr_brachial")

    train_cf_pwv_model(combined_results_rad, pwv_csv=PWV_CSV)


if __name__ == "__main__":
    main()
