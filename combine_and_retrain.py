"""
COMBINE VOICED + PVQD AND RETRAIN
===================================
Merges both feature datasets, applies SMOTE on the minority (healthy) class
to balance the data, then retrains the SVM / BiLSTM / Ensemble pipeline.

Run after:
  python pvqd_loader.py          ← extracts PVQD features
  (voiced_features.csv and voiced_temporal.npy already exist from step 2)

Outputs:
  models/  ← overwritten with updated models
  results/ ← updated confusion matrices and cv_results_combined.json
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import json, joblib

from sklearn.utils import resample

# Import everything from step 3 (models, CV routines, plotting)
from step3_model_training import (
    get_global_feature_cols,
    prepare_features,
    MG_PRIORITY_FEATURES,
    cross_validate_svm,
    cross_validate_lstm,
    cross_validate_ensemble,
    train_final_svm,
    train_final_lstm,
    plot_confusion_matrix,
    plot_model_comparison,
    normalise_temporal,
    OUTPUT_DIR,
    RESULTS_DIR,
)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
VOICED_CSV  = "./voiced_features.csv"
VOICED_NPY  = "./voiced_temporal.npy"
PVQD_CSV    = "./pvqd_features.csv"
PVQD_NPY    = "./pvqd_temporal.npy"
# ─────────────────────────────────────────────────────────────────────────────

Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)


def merge_datasets():
    """
    Merge VOICED and PVQD feature DataFrames.
    Aligns columns — fills missing cols with NaN (imputer handles them later).
    """
    voiced = pd.read_csv(VOICED_CSV)
    pvqd   = pd.read_csv(PVQD_CSV)

    # Add source tag if not present
    if "source" not in voiced.columns:
        voiced["source"] = "voiced"
    if "source" not in pvqd.columns:
        pvqd["source"] = "pvqd"

    combined = pd.concat([voiced, pvqd], ignore_index=True, sort=False)
    print(f"\nCombined dataset: {len(combined)} samples")
    print(f"  VOICED : {len(voiced)}")
    print(f"  PVQD   : {len(pvqd)}")
    print(f"\nLabel distribution:")
    print(combined["pathology_category"].value_counts().to_string())
    print(f"\nHealthy      : {(combined['label_binary']==0).sum()}")
    print(f"Pathological : {(combined['label_binary']==1).sum()}")
    return combined


def merge_temporal():
    """Stack temporal tensors from both datasets."""
    voiced_t = np.load(VOICED_NPY)
    pvqd_t   = np.load(PVQD_NPY)
    combined = np.concatenate([voiced_t, pvqd_t], axis=0)
    print(f"\nTemporal shapes: VOICED={voiced_t.shape}  PVQD={pvqd_t.shape}"
          f"  Combined={combined.shape}")
    return combined


def oversample_minority(X_global, X_temp, y):
    """
    Oversample the minority class (healthy) using random duplication with
    small Gaussian noise — simple but avoids SMOTE's k-NN issues on small sets.
    Target: balance healthy vs pathological (1:1).
    """
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_neg >= n_pos:
        print("Classes already balanced — skipping oversampling.")
        return X_global, X_temp, y

    n_to_add = n_pos - n_neg
    neg_idx  = np.where(y == 0)[0]

    # Sample with replacement from healthy class
    chosen = np.random.choice(neg_idx, size=n_to_add, replace=True)

    # Add tiny Gaussian noise to global features (avoids exact duplicates)
    noise_scale = 0.01 * np.nanstd(X_global[neg_idx], axis=0)
    X_aug_g     = X_global[chosen] + np.random.randn(n_to_add, X_global.shape[1]) * noise_scale
    X_aug_t     = X_temp[chosen]   # temporal — keep as-is (Praat features, adding noise changes F0)

    X_global_bal = np.vstack([X_global, X_aug_g])
    X_temp_bal   = np.concatenate([X_temp, X_aug_t], axis=0)
    y_bal        = np.concatenate([y, np.zeros(n_to_add, dtype=y.dtype)])

    # Shuffle
    perm         = np.random.permutation(len(y_bal))
    print(f"\nAfter oversampling:")
    print(f"  Healthy      : {(y_bal==0).sum()}  (was {n_neg})")
    print(f"  Pathological : {(y_bal==1).sum()}")
    return X_global_bal[perm], X_temp_bal[perm], y_bal[perm]


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # ── 1. Merge ──────────────────────────────────────────────────────────────
    df       = merge_datasets()
    X_temp   = merge_temporal()
    y        = df["label_binary"].values

    # Align temporal with df rows (PVQD extraction may have skipped some rows)
    # The temporal npy rows exactly match each CSV row (both skip failed records)
    assert len(df) == len(X_temp), (
        f"Row mismatch: df={len(df)}, temporal={len(X_temp)}\n"
        "Re-run pvqd_loader.py to regenerate aligned files."
    )

    # ── 2. Prepare global features ────────────────────────────────────────────
    feat_cols = get_global_feature_cols(df)
    available_priority = [f for f in MG_PRIORITY_FEATURES if f in feat_cols]
    remaining          = [f for f in feat_cols if f not in available_priority]
    ordered_feats      = available_priority + remaining

    X_global, y_, imputer = prepare_features(df, ordered_feats, "label_binary")

    print(f"\nGlobal features : {X_global.shape[1]}")
    print(f"Temporal shape  : {X_temp.shape}")

    # ── 3. Balance via oversampling ───────────────────────────────────────────
    X_global_bal, X_temp_bal, y_bal = oversample_minority(X_global, X_temp, y)

    # ── 4. Cross-validation on balanced data ─────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING ON COMBINED + BALANCED DATASET")
    print("="*60)

    svm_results      = cross_validate_svm(X_global_bal, y_bal)
    lstm_results     = cross_validate_lstm(X_temp_bal, y_bal)
    ensemble_results = cross_validate_ensemble(X_global_bal, X_temp_bal, y_bal)

    # ── 4b. Compute ROC-derived decision thresholds (Youden's J) ─────────────
    # Methodology: Van Calster et al. (2019). "Three myths about risk thresholds
    # for prediction models." BMC Medicine, 17, 192.
    # https://doi.org/10.1186/s12916-019-1425-3
    #
    # HIGH threshold — Youden's J optimal point (maximises sensitivity+specificity).
    # LOW  threshold — 95%-sensitivity point: lowest probability at which ≥95%
    #                  of true positives are still caught. Anything below this is
    #                  labelled LOW (≤5% pathological cases expected in this bin).
    from sklearn.metrics import roc_curve as _roc_curve
    _oof   = np.array(ensemble_results["oof_probs"])
    _ytrue = np.array(ensemble_results["y_true"])
    fpr_arr, tpr_arr, thresh_arr = _roc_curve(_ytrue, _oof)

    j_scores   = tpr_arr - fpr_arr
    opt_idx    = int(np.argmax(j_scores))
    high_thresh = float(thresh_arr[opt_idx])

    # 95%-sensitivity boundary: lowest threshold where TPR ≥ 0.95
    sens95_candidates = thresh_arr[tpr_arr >= 0.95]
    low_thresh = float(sens95_candidates.max()) if len(sens95_candidates) else high_thresh * 0.5
    low_thresh = min(low_thresh, high_thresh - 0.05)   # enforce a gap

    thresholds_meta = {
        "low":               round(low_thresh,  4),
        "high":              round(high_thresh, 4),
        "youden_j":          round(float(j_scores[opt_idx]), 4),
        "sensitivity_at_high": round(float(tpr_arr[opt_idx]),       4),
        "specificity_at_high": round(float(1 - fpr_arr[opt_idx]),   4),
        "method": (
            "HIGH = Youden's J optimal threshold (max sensitivity+specificity). "
            "LOW = 95%-sensitivity threshold (≤5% of pathologicals below this bound)."
        ),
        "citation": (
            "Van Calster B et al. (2019). Three myths about risk thresholds for "
            "prediction models. BMC Medicine, 17, 192. "
            "https://doi.org/10.1186/s12916-019-1425-3"
        ),
    }
    with open(f"{OUTPUT_DIR}/thresholds.json", "w") as f:
        json.dump(thresholds_meta, f, indent=2)
    print(f"\n✓ ROC thresholds saved → {OUTPUT_DIR}/thresholds.json")
    print(f"  LOW  < {low_thresh:.3f}  (95%-sensitivity boundary)")
    print(f"  HIGH ≥ {high_thresh:.3f}  (Youden's J = {thresholds_meta['youden_j']:.3f},"
          f"  Sens={thresholds_meta['sensitivity_at_high']:.3f},"
          f"  Spec={thresholds_meta['specificity_at_high']:.3f})")

    # ── 5. Confusion matrices ─────────────────────────────────────────────────
    import numpy as _np
    plot_confusion_matrix(
        _np.array(svm_results["conf_mat"]),
        "SVM — Combined Dataset",
        save_path=f"{RESULTS_DIR}/svm_confmat_combined.png"
    )
    plot_confusion_matrix(
        _np.array(lstm_results["conf_mat"]),
        "BiLSTM — Combined Dataset",
        save_path=f"{RESULTS_DIR}/lstm_confmat_combined.png"
    )
    plot_confusion_matrix(
        _np.array(ensemble_results["conf_mat"]),
        "Ensemble — Combined Dataset",
        save_path=f"{RESULTS_DIR}/ensemble_confmat_combined.png"
    )
    plot_model_comparison(svm_results, lstm_results, ensemble_results)

    # ── 6. Save results ───────────────────────────────────────────────────────
    _strip = {"conf_mat", "oof_probs", "y_true"}
    all_results = {
        "dataset":  f"combined_balanced (voiced+pvqd, n={len(y_bal)})",
        "svm":      {k: v for k, v in svm_results.items()  if k not in _strip},
        "lstm":     {k: v for k, v in lstm_results.items() if k not in _strip},
        "ensemble": {k: v for k, v in ensemble_results.items() if k not in _strip},
    }
    with open(f"{RESULTS_DIR}/cv_results_combined.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved → {RESULTS_DIR}/cv_results_combined.json")

    # ── 7. Retrain final models on full balanced data ─────────────────────────
    final_svm  = train_final_svm(X_global_bal, y_bal)
    final_lstm = train_final_lstm(X_temp_bal, y_bal)

    meta = {"feature_cols": ordered_feats, "label_map": {0: "healthy", 1: "pathological"}}
    with open(f"{OUTPUT_DIR}/feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    joblib.dump(imputer, f"{OUTPUT_DIR}/imputer.pkl")

    print(f"\n✓ Final models saved to {OUTPUT_DIR}/")
    print("\n── Final CV Summary ─────────────────────────────────────")
    print(f"  SVM      AUC={svm_results['auc']:.3f}  F1={svm_results['f1']:.3f}  "
          f"BalAcc={svm_results['bal_acc']:.3f}")
    print(f"  BiLSTM   AUC={lstm_results['auc']:.3f}  F1={lstm_results['f1']:.3f}  "
          f"BalAcc={lstm_results['bal_acc']:.3f}")
    print(f"  Ensemble AUC={ensemble_results['auc']:.3f}  F1={ensemble_results['f1']:.3f}  "
          f"BalAcc={ensemble_results['bal_acc']:.3f}")
