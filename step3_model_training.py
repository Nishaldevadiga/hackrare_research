"""
STEP 3: MODEL TRAINING
=======================
Two complementary classifiers:

  Model A — SVM (baseline, global features)
      Fast, interpretable, validated for voice pathology on small datasets.
      Best for: pathological vs healthy binary classification.

  Model B — Bidirectional LSTM (temporal features)
      Learns the WITHIN-RECORDING fatigue trajectory.
      Best for: detecting the "sinking pitch sign" pattern unique to MG.

  Model C — Stacked Ensemble (global + temporal → XGBoost)
      Combines probabilities from A and B.

Evaluation: Stratified 5-fold cross-validation with class weighting.

Install:
    pip install scikit-learn xgboost torch pandas numpy matplotlib seaborn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    balanced_accuracy_score, f1_score
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json


# ─── CONFIG ──────────────────────────────────────────────────────────────────
FEATURES_CSV  = "./voiced_features.csv"
TEMPORAL_NPY  = "./voiced_temporal.npy"
OUTPUT_DIR    = "./models"
RESULTS_DIR   = "./results"

N_FOLDS       = 5
RANDOM_SEED   = 42
LSTM_EPOCHS   = 50
LSTM_LR       = 1e-3
LSTM_BATCH    = 16

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# ─────────────────────────────────────────────────────────────────────────────

Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE PREP
# ══════════════════════════════════════════════════════════════════════════════

# Features most relevant to MG fatigue (from literature review)
MG_PRIORITY_FEATURES = [
    "f0_mean", "f0_std", "f0_range", "f0_slope", "f0_cv",
    "temporal_f0_slope",          # sinking pitch sign
    "f0_early_late_diff",         # pitch drop early→late
    "hnr_mean", "hnr_std",
    "hnr_early_late_diff",        # HNR worsening = breathiness increasing
    "jitter_local", "jitter_rap",
    "shimmer_local", "shimmer_apq3", "shimmer_dda",
    "shim_early_late_diff",       # shimmer worsening over time
    "voiced_fraction", "voiced_fraction_decay",
    "rms_mean", "rms_std", "rms_early_late_diff",
    "cpps_mean", "cpps_min",
    "spec_centroid_mean", "spectral_flux_mean",
    "mfcc_drift_l2",              # vocal tract shape change over time
    "f0_std_growth",              # instability growing = fatigue
]


def prepare_features(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Prepare X, y arrays with imputation for NaN features.
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values

    # Impute NaNs with median per column
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    return X, y, imputer


def get_global_feature_cols(df: pd.DataFrame) -> list:
    """Auto-detect numeric feature columns (exclude metadata)."""
    skip = {"record_id", "dat_path", "hea_path", "info_path",
            "pathology", "pathology_category", "gender", "label_binary",
            "smoking", "alcohol", "coffee", "age", "vhi_score", "rsi_score"}
    return [c for c in df.columns if c not in skip and
            pd.api.types.is_numeric_dtype(df[c])]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL A — SVM PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_svm_pipeline(n_features_select: int = 40) -> Pipeline:
    """
    SVM pipeline with:
      1. StandardScaler
      2. ANOVA F-test feature selection
      3. RBF SVM with class_weight='balanced'
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("select",  SelectKBest(f_classif, k=min(n_features_select, 100))),
        ("svm",     SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED
        ))
    ])


def cross_validate_svm(X: np.ndarray, y: np.ndarray,
                        n_folds: int = N_FOLDS) -> dict:
    """Run stratified k-fold CV for SVM. Returns metrics dict."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    all_probs, all_preds, all_true = [], [], []

    print(f"\n── SVM Cross-Validation ({n_folds}-fold) ──────────────────────")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipe = build_svm_pipeline()
        pipe.fit(X_tr, y_tr)

        probs = pipe.predict_proba(X_val)[:, 1]
        preds = pipe.predict(X_val)

        fold_auc = roc_auc_score(y_val, probs)
        fold_f1  = f1_score(y_val, preds, average="weighted")
        print(f"  Fold {fold}: AUC={fold_auc:.3f}  F1={fold_f1:.3f}")

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_true.extend(y_val)

    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    results = {
        "auc":       roc_auc_score(all_true, all_probs),
        "f1":        f1_score(all_true, all_preds, average="weighted"),
        "bal_acc":   balanced_accuracy_score(all_true, all_preds),
        "conf_mat":  confusion_matrix(all_true, all_preds).tolist(),
        "report":    classification_report(all_true, all_preds, output_dict=True),
        "oof_probs": all_probs.tolist(),
        "y_true":    all_true.tolist(),
    }
    print(f"\n  OVERALL → AUC: {results['auc']:.3f} | "
          f"F1: {results['f1']:.3f} | "
          f"Balanced Acc: {results['bal_acc']:.3f}")
    return results


def train_final_svm(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Train SVM on full dataset and save."""
    pipe = build_svm_pipeline()
    pipe.fit(X, y)
    joblib.dump(pipe, f"{OUTPUT_DIR}/svm_model.pkl")
    print(f"✓ SVM saved → {OUTPUT_DIR}/svm_model.pkl")
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# MODEL B — BIDIRECTIONAL LSTM
# ══════════════════════════════════════════════════════════════════════════════

class FatigueLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal fatigue trajectory classification.
    
    Input shape: (batch, T=10, F=19)
    Output: (batch, 2) — logits for [healthy, pathological]
    
    Architecture:
        BiLSTM → LayerNorm → Attention pooling → Dropout → Linear
    
    The attention pooling weights later frames more heavily if they carry
    more discriminative signal (i.e., late-recording degradation in MG).
    """

    def __init__(self, input_size: int = 19, hidden_size: int = 64,
                 num_layers: int = 2, n_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Temporal attention: learned weights per time step
        self.attention = nn.Linear(hidden_size * 2, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # x: (batch, T, F)
        lstm_out, _ = self.lstm(x)              # (batch, T, hidden*2)
        lstm_out    = self.layer_norm(lstm_out)

        # Attention pooling — learns which time steps matter most
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, T, 1)
        context      = (attn_weights * lstm_out).sum(dim=1)            # (batch, hidden*2)

        return self.classifier(context)         # (batch, n_classes)


def normalise_temporal(X_temp: np.ndarray) -> np.ndarray:
    """
    Normalise temporal feature matrix across all samples.
    X_temp: (N, T, F) → returns same shape, normalised per feature.
    """
    N, T, F = X_temp.shape
    X_flat  = X_temp.reshape(-1, F)

    mean_ = np.nanmean(X_flat, axis=0)
    std_  = np.nanstd(X_flat, axis=0) + 1e-9

    X_norm = (X_temp - mean_) / std_
    X_norm = np.nan_to_num(X_norm, nan=0.0)

    return X_norm, mean_, std_


def train_lstm_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    epochs: int = LSTM_EPOCHS) -> tuple:
    """
    Train LSTM for one CV fold.
    Returns: (model, val_probs, val_preds)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class weights for imbalanced data
    n_pos   = y_tr.sum()
    n_neg   = len(y_tr) - n_pos
    w_pos   = n_neg / (n_pos + 1e-9)
    weights = torch.tensor([1.0, w_pos], dtype=torch.float32).to(device)

    model     = FatigueLSTM(input_size=X_tr.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # DataLoaders
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.LongTensor(y_tr).to(device)
    X_v_t  = torch.FloatTensor(X_val).to(device)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    loader   = DataLoader(train_ds, batch_size=LSTM_BATCH, shuffle=True)

    best_val_auc = 0.0
    best_state   = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation AUC tracking
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_v = model(X_v_t)
                probs_v  = torch.softmax(logits_v, dim=1)[:, 1].cpu().numpy()
            cur_auc = roc_auc_score(y_val, probs_v) if len(np.unique(y_val)) > 1 else 0.5
            if cur_auc > best_val_auc:
                best_val_auc = cur_auc
                best_state   = {k: v.cpu() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final predictions
    model.eval()
    with torch.no_grad():
        logits_v = model(X_v_t)
        probs_v  = torch.softmax(logits_v, dim=1)[:, 1].cpu().numpy()
        preds_v  = logits_v.argmax(dim=1).cpu().numpy()

    return model, probs_v, preds_v


def cross_validate_lstm(X_temp: np.ndarray, y: np.ndarray,
                         n_folds: int = N_FOLDS) -> dict:
    """Run stratified k-fold CV for Bi-LSTM."""
    X_norm, _, _ = normalise_temporal(X_temp)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    all_probs, all_preds, all_true = [], [], []

    print(f"\n── BiLSTM Cross-Validation ({n_folds}-fold) ──────────────────────")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_norm, y), 1):
        X_tr, X_val = X_norm[train_idx], X_norm[val_idx]
        y_tr, y_val = y[train_idx],      y[val_idx]

        _, probs, preds = train_lstm_fold(X_tr, y_tr, X_val, y_val)

        if len(np.unique(y_val)) > 1:
            fold_auc = roc_auc_score(y_val, probs)
        else:
            fold_auc = float("nan")
        fold_f1 = f1_score(y_val, preds, average="weighted")
        print(f"  Fold {fold}: AUC={fold_auc:.3f}  F1={fold_f1:.3f}")

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_true.extend(y_val)

    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    results = {
        "auc":     roc_auc_score(all_true, all_probs),
        "f1":      f1_score(all_true, all_preds, average="weighted"),
        "bal_acc": balanced_accuracy_score(all_true, all_preds),
        "conf_mat": confusion_matrix(all_true, all_preds).tolist(),
        "report":  classification_report(all_true, all_preds, output_dict=True),
    }
    print(f"\n  OVERALL → AUC: {results['auc']:.3f} | "
          f"F1: {results['f1']:.3f} | "
          f"Balanced Acc: {results['bal_acc']:.3f}")
    return results


def train_final_lstm(X_temp: np.ndarray, y: np.ndarray):
    """Train LSTM on full dataset and save."""
    X_norm, mean_, std_ = normalise_temporal(X_temp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, _ = train_lstm_fold(X_norm, y, X_norm, y, epochs=LSTM_EPOCHS * 2)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/lstm_model.pt")
    np.save(f"{OUTPUT_DIR}/temporal_mean.npy", mean_)
    np.save(f"{OUTPUT_DIR}/temporal_std.npy",  std_)
    print(f"✓ LSTM saved → {OUTPUT_DIR}/lstm_model.pt")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MODEL C — STACKED ENSEMBLE (XGBoost on model probabilities)
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_ensemble(X_global: np.ndarray, X_temp: np.ndarray,
                             y: np.ndarray) -> dict:
    """
    Stack SVM + LSTM predictions as meta-features, classify with XGBoost.
    Uses out-of-fold predictions to avoid leakage.
    """
    X_norm, _, _ = normalise_temporal(X_temp)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Collect out-of-fold predictions
    svm_oof  = np.zeros(len(y))
    lstm_oof = np.zeros(len(y))

    print(f"\n── Generating OOF predictions for Ensemble ───────────────────")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_global, y), 1):
        print(f"  Fold {fold}...")

        # SVM
        svm_pipe = build_svm_pipeline()
        svm_pipe.fit(X_global[train_idx], y[train_idx])
        svm_oof[val_idx] = svm_pipe.predict_proba(X_global[val_idx])[:, 1]

        # LSTM
        _, lstm_probs, _ = train_lstm_fold(
            X_norm[train_idx], y[train_idx],
            X_norm[val_idx],   y[val_idx]
        )
        lstm_oof[val_idx] = lstm_probs

    # Meta-features: [svm_prob, lstm_prob, avg_prob]
    meta_X = np.column_stack([svm_oof, lstm_oof, (svm_oof + lstm_oof) / 2])

    # XGBoost on meta-features (nested CV)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=RANDOM_SEED
    )

    ensemble_probs = np.zeros(len(y))
    ensemble_preds = np.zeros(len(y), dtype=int)

    for train_idx, val_idx in skf.split(meta_X, y):
        xgb_model.fit(meta_X[train_idx], y[train_idx])
        ensemble_probs[val_idx] = xgb_model.predict_proba(meta_X[val_idx])[:, 1]
        ensemble_preds[val_idx] = xgb_model.predict(meta_X[val_idx])

    results = {
        "auc":      roc_auc_score(y, ensemble_probs),
        "f1":       f1_score(y, ensemble_preds, average="weighted"),
        "bal_acc":  balanced_accuracy_score(y, ensemble_preds),
        "conf_mat": confusion_matrix(y, ensemble_preds).tolist(),
        "report":   classification_report(y, ensemble_preds, output_dict=True),
        "oof_probs": ensemble_probs.tolist(),
        "y_true":    y.tolist(),
    }
    print(f"\n  ENSEMBLE → AUC: {results['auc']:.3f} | "
          f"F1: {results['f1']:.3f} | "
          f"Balanced Acc: {results['bal_acc']:.3f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(conf_mat: list, title: str, labels=None, save_path: str = None):
    labels = labels or ["Healthy", "Pathological"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        conf_mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_model_comparison(svm_r: dict, lstm_r: dict, ensemble_r: dict):
    """Bar chart comparing AUC, F1, Balanced Accuracy across models."""
    models  = ["SVM", "BiLSTM", "Ensemble"]
    metrics = {
        "ROC-AUC":          [svm_r["auc"],     lstm_r["auc"],     ensemble_r["auc"]],
        "F1 (weighted)":    [svm_r["f1"],      lstm_r["f1"],      ensemble_r["f1"]],
        "Balanced Accuracy":[svm_r["bal_acc"], lstm_r["bal_acc"], ensemble_r["bal_acc"]],
    }

    x    = np.arange(len(models))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i * w, vals, w, label=metric)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — MG Fatigue Detection")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150)
    plt.close()
    print(f"✓ Saved → {RESULTS_DIR}/model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading features...")
    df       = pd.read_csv(FEATURES_CSV)
    X_temp   = np.load(TEMPORAL_NPY)     # (N, T, F)
    y        = df["label_binary"].values  # 0=healthy, 1=pathological

    # Get feature columns
    feat_cols = get_global_feature_cols(df)
    
    # Prioritize MG-relevant features if available
    available_priority = [f for f in MG_PRIORITY_FEATURES if f in feat_cols]
    remaining          = [f for f in feat_cols if f not in available_priority]
    ordered_feats      = available_priority + remaining

    X_global, y_, imputer = prepare_features(df, ordered_feats, "label_binary")
    
    print(f"\nDataset: {len(df)} samples | "
          f"Healthy: {(y==0).sum()} | Pathological: {(y==1).sum()}")
    print(f"Global features: {X_global.shape[1]} | "
          f"Temporal shape: {X_temp.shape}")

    # ── Model A: SVM ──────────────────────────────────────────────────────────
    svm_results  = cross_validate_svm(X_global, y)
    svm_confmat  = np.array(svm_results["conf_mat"])
    plot_confusion_matrix(svm_confmat, "SVM — Confusion Matrix",
                          save_path=f"{RESULTS_DIR}/svm_confmat.png")

    # ── Model B: BiLSTM ───────────────────────────────────────────────────────
    lstm_results = cross_validate_lstm(X_temp, y)
    lstm_confmat = np.array(lstm_results["conf_mat"])
    plot_confusion_matrix(lstm_confmat, "BiLSTM — Confusion Matrix",
                          save_path=f"{RESULTS_DIR}/lstm_confmat.png")

    # ── Model C: Ensemble ─────────────────────────────────────────────────────
    ensemble_results = cross_validate_ensemble(X_global, X_temp, y)
    ens_confmat      = np.array(ensemble_results["conf_mat"])
    plot_confusion_matrix(ens_confmat, "Ensemble — Confusion Matrix",
                          save_path=f"{RESULTS_DIR}/ensemble_confmat.png")

    # ── Comparison plot ───────────────────────────────────────────────────────
    plot_model_comparison(svm_results, lstm_results, ensemble_results)

    # ── Save results ──────────────────────────────────────────────────────────
    all_results = {
        "svm":      {k: v for k, v in svm_results.items()  if k != "conf_mat"},
        "lstm":     {k: v for k, v in lstm_results.items() if k != "conf_mat"},
        "ensemble": {k: v for k, v in ensemble_results.items() if k != "conf_mat"},
    }
    with open(f"{RESULTS_DIR}/cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ All results saved → {RESULTS_DIR}/cv_results.json")

    # ── Train final models on all data ────────────────────────────────────────
    final_svm  = train_final_svm(X_global, y)
    final_lstm = train_final_lstm(X_temp, y)
    
    # Save feature metadata for inference
    meta = {"feature_cols": ordered_feats, "label_map": {0: "healthy", 1: "pathological"}}
    with open(f"{OUTPUT_DIR}/feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    joblib.dump(imputer, f"{OUTPUT_DIR}/imputer.pkl")
    print(f"\n✓ Final models trained and saved to {OUTPUT_DIR}/")
