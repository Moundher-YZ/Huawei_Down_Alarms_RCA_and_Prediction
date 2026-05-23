"""
src/prediction.py
=================
Phase 4 — ML Prediction for Djezzy 5G Alarm Analysis.

Architecture
------------
Feature matrix = rolling window features (precomputed once) + temporal columns.
No transformer, no static rate statistics — nothing that could leak the label.

Rolling features capture all meaningful signal:
  • n_total_Wm       — alarm volume at the site in the past W minutes
  • n_hs_Wm          — HS alarm count at the site in the past W minutes
  • n_unique_Wm      — unique alarm types at the site in the past W minutes
  • rate_Wm          — alarm rate (events / minute)
  • hs_rate_Wm       — fraction of alarms that were HS
  • prec_<name>_Wm   — count of each top precursor alarm in the past W minutes

Temporal columns (hour, day_of_week, month, is_night, is_weekend) are
derived from the timestamp and carry no label information.

Cross-validation uses TimeSeriesSplit so future data never leaks into
past training windows. No Pipeline transformer needed — X is already a
clean numeric matrix when it enters cross_validate().

Public API
----------
  build_rolling_features(df, ...)      → (feat_df, top_precursor_alarms)
  prepare_X_y(df, rolling_feat)        → (X, y)
  get_models(scale_pos_weight)         → dict of classifiers
  cross_validate_all(X, y, ...)        → summary DataFrame
  train_final_model(X, y, ...)         → fitted model
  tune_threshold(y_true, y_proba, ...) → (threshold, fbeta)
  evaluate_model(model, X, y, ...)     → metrics dict
  print_evaluation(metrics)
  compute_shap_values(model, X, ...)   → shap.Explanation
  get_top_shap_features(shap_vals, ..) → DataFrame
  save_model(model, meta, path)
  load_model(path)                     → (model, meta)
  predict_hs(new_df, model, meta, ...) → DataFrame with predictions
  train_category_classifier(X_hs, y)  → (model, LabelEncoder)
"""

from __future__ import annotations

import os
import warnings
import joblib
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MAPPING_RULES: dict[str, str] = {
    "OML Fault"                  : "LOSS-OF-ALL CHANNEL",
    "CSL Fault"                  : "LOSS-OF-ALL CHANNEL",
    "NodeB Unavailable"          : "LOSS-OF-ALL CHANNEL",
    "NE Is Disconnected"         : "Pas De Supervision",
    "GSM Cell out of Service"    : "2G cell down",
    "UMTS Cell Unavailable"      : "3G cell down",
    "UMTS Cell Setup Failed"     : "3G cell down",
    "Cell Unavailable"           : "4G cell down",
    "S1ap Link Down"             : "LOSS-OF-ALL CHANNEL",
    "GSM Local Cell Unusable"    : "2G cell down",
    "NR DU Cell TRP Unavailable" : "5G cell down",
    "NR Cell Unavailable"        : "5G cell down",
    "gNodeB Out of Service"      : "LOSS-OF-ALL CHANNEL",
}
HS_ALARM_NAMES: set[str]  = set(MAPPING_RULES.keys())
WINDOWS_MIN:   list[int]  = [5, 15, 30, 60]
TOP_PRECURSORS: int        = 20
N_JOBS:         int        = -1
TEMPORAL_COLS: list[str]  = ["hour", "day_of_week", "month", "is_night", "is_weekend"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive is_HS, hs_category, and integer temporal columns in-place copy."""
    df = df.copy()

    for col in ["Occurrencetime", "Clearancetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "is_HS" not in df.columns:
        df["is_HS"]       = df["Alarmname"].isin(HS_ALARM_NAMES)
        df["hs_category"] = df["Alarmname"].map(MAPPING_RULES).fillna("Non-HS")

    if "Occurrencetime" in df.columns:
        df["hour"]        = df["Occurrencetime"].dt.hour
        df["day_of_week"] = df["Occurrencetime"].dt.dayofweek  # 0=Mon … 6=Sun
        df["month"]       = df["Occurrencetime"].dt.month
        df["is_night"]    = df["hour"].between(0, 6).astype(int)
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    return df


def _safe(name: str, max_len: int = 30) -> str:
    return name.replace(" ", "_").replace("/", "_")[:max_len]


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROLLING WINDOW FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_one_site(
    site_df       : pd.DataFrame,
    windows_ns    : list[int],
    windows_min   : list[int],
    prec_codes    : np.ndarray,
    top_precursors: list[str],
    col_names     : list[str],
) -> pd.DataFrame:
    """
    Vectorised rolling features for a single site.

    Window = [t - W, t)  — current event is EXCLUDED so the features only
    reflect past activity and cannot encode the current event's own label.
    """
    n = len(site_df)
    if n == 0:
        return pd.DataFrame(columns=col_names)

    site_df      = site_df.sort_values("Occurrencetime")
    orig_indices = site_df.index.values          # original positional index
    times_ns     = site_df["Occurrencetime"].values.astype("int64")
    is_hs_arr    = site_df["is_HS"].values.astype("int32")
    alarm_codes  = prec_codes[site_df.index]

    # prefix[i] = sum of rows [0, i-1]  →  prefix[i] excludes row i itself
    hs_prefix = np.zeros(n + 1, dtype="int32")
    np.cumsum(is_hs_arr, out=hs_prefix[1:])

    P           = len(top_precursors)
    prec_matrix = np.zeros((n + 1, P), dtype="int32")
    for p_idx in range(P):
        indicator = (alarm_codes == p_idx).astype("int32")
        np.cumsum(indicator, out=prec_matrix[1:, p_idx])

    def _unique_counts(left_indices: np.ndarray) -> np.ndarray:
        """Unique alarm types in [left, i) — excludes current event."""
        counts = np.empty(n, dtype="int32")
        freq: dict[int, int] = defaultdict(int)
        left = 0
        for i in range(n):
            new_left = int(left_indices[i])
            while left < new_left:
                c = int(alarm_codes[left])
                freq[c] -= 1
                if freq[c] == 0:
                    del freq[c]
                left += 1
            counts[i] = len(freq)
            # add current event AFTER counting so next event sees it
            freq[int(alarm_codes[i])] += 1
        return counts

    rows    = np.empty((n, len(col_names)), dtype="float32")
    col_idx = 0
    i_idx   = np.arange(n, dtype="int32")

    for w_ns, w_min in zip(windows_ns, windows_min):
        left_ns  = times_ns - w_ns
        left_idx = np.searchsorted(times_ns, left_ns, side="left")

        # Use prefix[i] (not prefix[i+1]) → window [t-W, t) excludes row i
        n_total  = i_idx - left_idx
        n_hs     = hs_prefix[i_idx] - hs_prefix[left_idx]
        rate     = n_total.astype("float32") / w_min
        hs_rate  = np.where(n_total > 0,
                            n_hs / np.maximum(n_total, 1), 0).astype("float32")
        n_unique = _unique_counts(left_idx).astype("float32")

        rows[:, col_idx    ] = n_total
        rows[:, col_idx + 1] = n_hs
        rows[:, col_idx + 2] = n_unique
        rows[:, col_idx + 3] = rate
        rows[:, col_idx + 4] = hs_rate
        col_idx += 5

        prec_counts = prec_matrix[i_idx] - prec_matrix[left_idx]
        rows[:, col_idx : col_idx + P] = prec_counts
        col_idx += P

    return pd.DataFrame(rows, columns=col_names, index=orig_indices)


def build_rolling_features(
    df                   : pd.DataFrame,
    windows_min          : list[int]           = WINDOWS_MIN,
    top_precursor_alarms : Optional[list[str]] = None,
    n_jobs               : int                 = N_JOBS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute rolling window features for every alarm event.

    Call ONCE on the full dataset before any train/test split.
    The window [t-W, t) excludes the current event so no label leaks in.

    Returns
    -------
    feat_df              : DataFrame aligned to df's index
    top_precursor_alarms : list[str] — save this alongside the model
    """
    df = _ensure_columns(df).sort_values("Occurrencetime")
    # Use clean 0-based positional index internally so site_df.index
    # maps directly into prec_codes without any "index" column tricks.
    original_index = df.index          # save caller's index to restore later
    df = df.reset_index(drop=True)     # positional 0..N-1

    if top_precursor_alarms is None:
        top_precursor_alarms = (
            df[~df["is_HS"]]["Alarmname"]
            .value_counts()
            .head(TOP_PRECURSORS)
            .index.tolist()
        )

    alarm_cat  = pd.Categorical(df["Alarmname"], categories=top_precursor_alarms)
    prec_codes = np.where(
        alarm_cat.codes >= 0, alarm_cat.codes, len(top_precursor_alarms)
    ).astype("int32")

    col_names: list[str] = []
    for w in windows_min:
        col_names += [f"n_total_{w}m", f"n_hs_{w}m", f"n_unique_{w}m",
                      f"rate_{w}m", f"hs_rate_{w}m"]
        for prec in top_precursor_alarms:
            col_names.append(f"prec_{_safe(prec)}_{w}m")

    windows_ns  = [w * 60 * 1_000_000_000 for w in windows_min]
    site_groups = {s: grp for s, grp in df.groupby("codesite")}

    print(f"  Rolling features: {len(site_groups):,} sites | "
          f"{len(df):,} events | {len(col_names)} columns")

    results: list[pd.DataFrame] = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_rolling_one_site)(
            sg, windows_ns, windows_min, prec_codes, top_precursor_alarms, col_names,
        )
        for sg in tqdm(site_groups.values(), desc="  Sites", unit="site", ncols=80)
    )

    # AFTER (fixed):
    feat_df = pd.concat(results).sort_index()
    # feat_df is 0..N-1, matching the internally-reset df.
    # Restore caller's original index so it aligns with df_raw.
    feat_df.index = original_index
    return feat_df, top_precursor_alarms


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def prepare_X_y(
    df          : pd.DataFrame,
    rolling_feat: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Assemble the final feature matrix and target vector.

    Features = rolling columns + temporal columns (hour, day_of_week, …).
    No static rate statistics — those leak the label.

    Parameters
    ----------
    df           : raw alarm DataFrame (must have Occurrencetime, is_HS)
    rolling_feat : output of build_rolling_features(), aligned to df's index

    Returns
    -------
    X : numeric DataFrame, no NaNs
    y : binary int array (1 = HS)
    """
    df = _ensure_columns(df)

    temporal = df[[c for c in TEMPORAL_COLS if c in df.columns]].copy()
    X = pd.concat([rolling_feat, temporal], axis=1).fillna(0)
    y = df["is_HS"].astype(int).values

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_models(scale_pos_weight: float = 10.0) -> dict:
    """Return {name: estimator} for all candidate classifiers."""
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_all(
    X                : pd.DataFrame,
    y                : np.ndarray,
    scale_pos_weight : float = 10.0,
    n_splits         : int   = 5,
) -> pd.DataFrame:
    """
    Time-series cross-validation for all models.

    X and y must already be sorted chronologically (prepare_X_y on a
    time-sorted DataFrame guarantees this).

    No SMOTE in CV — class weights handle imbalance without the memory cost.
    SMOTE is applied only when training the final model.

    Returns a summary DataFrame ranked by PR-AUC.
    """
    cv      = TimeSeriesSplit(n_splits=n_splits)
    models  = get_models(scale_pos_weight)
    summary = []

    for name, model in models.items():
        print(f"  CV {name}...")
        scores = cross_validate(
            model, X, y,
            cv      = cv,
            scoring = ["roc_auc", "average_precision", "f1"],
            n_jobs  = 1,
            verbose = 0,
        )
        summary.append({
            "model"        : name,
            "roc_auc_mean" : scores["test_roc_auc"].mean(),
            "roc_auc_std"  : scores["test_roc_auc"].std(),
            "pr_auc_mean"  : scores["test_average_precision"].mean(),
            "pr_auc_std"   : scores["test_average_precision"].std(),
            "f1_mean"      : scores["test_f1"].mean(),
            "f1_std"       : scores["test_f1"].std(),
        })
        print(f"    ROC-AUC={scores['test_roc_auc'].mean():.4f}  "
              f"PR-AUC={scores['test_average_precision'].mean():.4f}  "
              f"F1={scores['test_f1'].mean():.4f}")

    return (pd.DataFrame(summary)
              .sort_values("pr_auc_mean", ascending=False)
              .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(
    X                : pd.DataFrame,
    y                : np.ndarray,
    model_name       : str   = "XGBoost",
    scale_pos_weight : float = 10.0,
    use_smote        : bool  = True,
) -> object:
    """
    Train the chosen model on the full training set.

    SMOTE is safe here — there is no validation fold to leak into.
    Returns the fitted model (not a Pipeline; X is already numeric).
    """
    X_fit, y_fit = X.values.astype("float32"), y

    if use_smote:
        print(f"  Applying SMOTE (sampling_strategy=0.1)...")
        smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
        X_fit, y_fit = smote.fit_resample(X_fit, y_fit)
        print(f"  After SMOTE: {X_fit.shape[0]:,} rows  "
              f"HS={y_fit.sum():,} ({y_fit.mean()*100:.1f}%)")

    model = get_models(scale_pos_weight)[model_name]
    print(f"  Fitting {model_name} on {len(X_fit):,} rows...")
    model.fit(X_fit, y_fit)
    print("  Done.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 8. THRESHOLD TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_threshold(
    y_true  : np.ndarray,
    y_proba : np.ndarray,
    beta    : float = 2.0,
) -> tuple[float, float]:
    """
    Find the decision threshold that maximises F-beta on the validation set.
    beta=2 weights recall twice as heavily as precision.

    Returns (best_threshold, best_fbeta).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f_beta = (
        (1 + beta**2) * precision[:-1] * recall[:-1]
        / np.maximum((beta**2) * precision[:-1] + recall[:-1], 1e-9)
    )
    best_idx = np.argmax(f_beta)
    return float(thresholds[best_idx]), float(f_beta[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model      : object,
    X          : pd.DataFrame,
    y          : np.ndarray,
    threshold  : float = 0.5,
    model_name : str   = "Model",
) -> dict:
    """Full evaluation on a held-out set. Returns a metrics dict."""
    proba = model.predict_proba(X.values.astype("float32"))[:, 1]
    pred  = (proba >= threshold).astype(int)

    report         = classification_report(y, pred,
                                           target_names=["Non-HS", "HS"],
                                           output_dict=True)
    cm             = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model"           : model_name,
        "threshold"       : threshold,
        "roc_auc"         : roc_auc_score(y, proba),
        "pr_auc"          : average_precision_score(y, proba),
        "f1_hs"           : report["HS"]["f1-score"],
        "precision_hs"    : report["HS"]["precision"],
        "recall_hs"       : report["HS"]["recall"],
        "f2_hs"           : fbeta_score(y, pred, beta=2, zero_division=0),
        "accuracy"        : report["accuracy"],
        "tp"              : int(tp),
        "fp"              : int(fp),
        "fn"              : int(fn),
        "tn"              : int(tn),
        "confusion_matrix": cm,
        "proba"           : proba,
    }


def print_evaluation(metrics: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  Evaluation — {metrics['model']}  (threshold={metrics['threshold']:.3f})")
    print(f"{'='*55}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
    print(f"  F1  (HS)  : {metrics['f1_hs']:.4f}")
    print(f"  F2  (HS)  : {metrics['f2_hs']:.4f}")
    print(f"  Precision : {metrics['precision_hs']:.4f}")
    print(f"  Recall    : {metrics['recall_hs']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"\n  Confusion Matrix  "
          f"TP={metrics['tp']}  FP={metrics['fp']}  "
          f"FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SHAP EXPLANATIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model        : object,
    X_sample     : pd.DataFrame,
    model_name   : str = "XGBoost",
    n_background : int = 200,
) -> shap.Explanation:
    """
    Compute SHAP values on a sample of the test set.

    TreeExplainer for tree models (fast, exact).
    KernelExplainer fallback for LogisticRegression.
    """
    X_np = X_sample.values.astype("float32")

    if model_name in ("XGBoost", "LightGBM", "RandomForest"):
        # unwrap sklearn Pipeline if LogisticRegression wrapped it
        clf       = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer(X_np)
    else:
        # LogisticRegression is inside a StandardScaler Pipeline
        background = shap.sample(X_np, min(n_background, len(X_np)))
        explainer  = shap.KernelExplainer(
            model.predict_proba, background, link="logit"
        )
        shap_vals  = explainer.shap_values(X_np, nsamples=100)

    return shap_vals


def get_top_shap_features(
    shap_values  ,
    feature_names: list[str],
    top_n        : int = 20,
) -> pd.DataFrame:
    """Mean |SHAP| per feature, ranked descending."""
    vals = np.abs(shap_values.values if hasattr(shap_values, "values")
                  else np.array(shap_values))
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    mean_shap = vals.mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_shap": mean_shap})
        .sort_values("mean_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    model                : object,
    top_precursor_alarms : list[str],
    feature_names        : list[str],
    threshold            : float,
    output_dir           : str = "../models/",
    name                 : str = "hs_predictor",
) -> None:
    """Save model + all metadata needed for inference."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.joblib")
    joblib.dump({
        "model"               : model,
        "top_precursor_alarms": top_precursor_alarms,
        "feature_names"       : feature_names,
        "threshold"           : threshold,
    }, path)
    print(f"Model saved → {path}")


def load_model(path: str) -> tuple[object, dict]:
    """
    Load a saved model.

    Returns (model, meta) where meta has keys:
      top_precursor_alarms, feature_names, threshold
    """
    payload = joblib.load(path)
    model   = payload.pop("model")
    return model, payload


# ─────────────────────────────────────────────────────────────────────────────
# 12. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_hs(
    new_df               : pd.DataFrame,
    model                : object,
    meta                 : dict,
    precomputed_rolling  : Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Batch prediction on new alarm data.

    Parameters
    ----------
    new_df              : raw alarm DataFrame (same schema as training data)
    model               : fitted model from train_final_model / load_model
    meta                : dict with keys top_precursor_alarms, feature_names, threshold
    precomputed_rolling : pass if rolling features are already computed

    Returns new_df with 'hs_proba' and 'hs_predicted' columns appended.
    """
    if precomputed_rolling is not None:
        rolling = precomputed_rolling
    else:
        rolling, _ = build_rolling_features(
            new_df,
            top_precursor_alarms=meta["top_precursor_alarms"],
        )

    X, _ = prepare_X_y(new_df, rolling)
    X    = X[meta["feature_names"]]   # guarantee column order matches training

    proba = model.predict_proba(X.values.astype("float32"))[:, 1]
    pred  = (proba >= meta["threshold"]).astype(int)

    out               = new_df.copy()
    out["hs_proba"]   = proba
    out["hs_predicted"] = pred
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 13. MULTI-CLASS — HS CATEGORY
# ─────────────────────────────────────────────────────────────────────────────

def train_category_classifier(
    X_hs : pd.DataFrame,
    y_cat: np.ndarray,
) -> object:
    """
    Train a LightGBM multi-class classifier on HS-only events.

    Parameters
    ----------
    X_hs  : feature matrix for HS events only (same rolling + temporal columns)
    y_cat : integer-encoded HS category labels

    Returns fitted LightGBM model.
    """
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    print(f"Training category classifier on {len(X_hs):,} HS events...")
    model.fit(X_hs.values.astype("float32"), y_cat)
    print("  Done.")
    return model
