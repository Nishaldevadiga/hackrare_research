"""
TEST SCRIPT: Run SVM + LSTM inference on real audio files from VOICED dataset.
Picks known healthy and known pathological samples to validate model performance.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np
import pandas as pd
import json
import joblib
import torch
from pathlib import Path

# ─── Setup ────────────────────────────────────────────────────────────────────
os.chdir("/Users/nishu/Desktop/hackrare_research")
sys.path.insert(0, ".")

OUTPUT_DIR = "./models"

# ─── Load metadata to pick test samples with known labels ─────────────────────
meta = pd.read_csv("voiced_metadata.csv")
print("=" * 70)
print("  MODEL VALIDATION TEST — SVM + LSTM on VOICED Audio Files")
print("=" * 70)
print(f"\nDataset: {len(meta)} recordings total")
print(f"Label distribution:\n{meta['label_binary'].value_counts().to_string()}")
print(f"Pathology categories:\n{meta['pathology_category'].value_counts().to_string()}")

# Pick test samples: 2 healthy (label=0) + 3 pathological (label=1, diverse categories)
test_samples = []

# Healthy samples
healthy = meta[meta['label_binary'] == 0].head(3)
for _, row in healthy.iterrows():
    test_samples.append({
        'record_id': row['record_id'],
        'dat_path': row['dat_path'],
        'true_label': 0,
        'pathology': row['pathology'],
        'category': row['pathology_category']
    })

# Pathological: mg_like
mg_like = meta[meta['pathology_category'] == 'mg_like'].head(2)
for _, row in mg_like.iterrows():
    test_samples.append({
        'record_id': row['record_id'],
        'dat_path': row['dat_path'],
        'true_label': 1,
        'pathology': row['pathology'],
        'category': row['pathology_category']
    })

# Pathological: structural
structural = meta[meta['pathology_category'] == 'structural'].head(1)
for _, row in structural.iterrows():
    test_samples.append({
        'record_id': row['record_id'],
        'dat_path': row['dat_path'],
        'true_label': 1,
        'pathology': row['pathology'],
        'category': row['pathology_category']
    })

# Pathological: other_pathology
other = meta[meta['pathology_category'] == 'other_pathology'].head(1)
for _, row in other.iterrows():
    test_samples.append({
        'record_id': row['record_id'],
        'dat_path': row['dat_path'],
        'true_label': 1,
        'pathology': row['pathology'],
        'category': row['pathology_category']
    })

print(f"\n── Test samples selected: {len(test_samples)} ─────────────────────")
for s in test_samples:
    label_str = "HEALTHY" if s['true_label'] == 0 else "PATHOLOGICAL"
    print(f"  {s['record_id']:10s}  label={label_str:13s}  pathology={s['pathology']}  cat={s['category']}")

# ─── Load models ──────────────────────────────────────────────────────────────
print("\n── Loading model artifacts ──────────────────────────────────────")

svm_pipe = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
print(f"  ✓ SVM pipeline loaded: {type(svm_pipe).__name__}")

imputer = joblib.load(f"{OUTPUT_DIR}/imputer.pkl")
print(f"  ✓ Imputer loaded: {type(imputer).__name__}")

with open(f"{OUTPUT_DIR}/feature_meta.json") as f:
    feat_meta = json.load(f)
feat_cols = feat_meta["feature_cols"]
print(f"  ✓ Feature columns: {len(feat_cols)} features")

mean_ = np.load(f"{OUTPUT_DIR}/temporal_mean.npy")
std_  = np.load(f"{OUTPUT_DIR}/temporal_std.npy")
print(f"  ✓ Temporal stats loaded: mean shape={mean_.shape}, std shape={std_.shape}")

from step3_model_training import FatigueLSTM
lstm = FatigueLSTM(input_size=19)
lstm.load_state_dict(torch.load(f"{OUTPUT_DIR}/lstm_model.pt", map_location="cpu"))
lstm.eval()
print(f"  ✓ LSTM model loaded: {sum(p.numel() for p in lstm.parameters())} params")

with open(f"{OUTPUT_DIR}/thresholds.json") as f:
    thresh = json.load(f)
low_thresh, high_thresh = thresh["low"], thresh["high"]
print(f"  ✓ Thresholds: low={low_thresh:.4f}, high={high_thresh:.4f}")

# ─── Import the inference feature extractor ───────────────────────────────────
from step5_inference import extract_features_for_inference

# ─── Run inference on each sample ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  RUNNING INFERENCE")
print("=" * 70)

results = []

for sample in test_samples:
    audio_path = sample['dat_path']
    record_id  = sample['record_id']
    true_label = sample['true_label']
    
    print(f"\n── {record_id} (true={('HEALTHY' if true_label == 0 else 'PATHOLOGICAL')}) ──")
    
    try:
        # 1. Extract features from raw audio
        global_feats, temporal = extract_features_for_inference(audio_path)
        
        # 2. SVM prediction
        x_row = np.array([global_feats.get(c, 0.0) for c in feat_cols],
                         dtype=np.float32).reshape(1, -1)
        x_imp = imputer.transform(x_row)
        svm_prob = float(svm_pipe.predict_proba(x_imp)[0, 1])
        svm_pred = 1 if svm_prob >= high_thresh else 0
        
        # 3. LSTM prediction
        temp_norm = (temporal - mean_) / (std_ + 1e-9)
        temp_norm = np.nan_to_num(temp_norm, nan=0.0)
        x_temp_t  = torch.FloatTensor(temp_norm).unsqueeze(0)
        
        with torch.no_grad():
            logits    = lstm(x_temp_t)
            lstm_prob = float(torch.softmax(logits, dim=1)[0, 1])
        lstm_pred = 1 if lstm_prob >= high_thresh else 0
        
        # 4. Ensemble
        ensemble_prob = (svm_prob + lstm_prob) / 2.0
        risk = "LOW" if ensemble_prob < low_thresh else "MODERATE" if ensemble_prob < high_thresh else "HIGH"
        ensemble_pred = 1 if ensemble_prob >= high_thresh else 0
        
        # 5. Key clinical features
        f0_slope = global_feats.get("temporal_f0_slope", 0.0)
        sinking  = f0_slope < -10
        
        svm_correct  = "✓" if svm_pred == true_label else "✗"
        lstm_correct = "✓" if lstm_pred == true_label else "✗"
        ens_correct  = "✓" if ensemble_pred == true_label else "✗"
        
        print(f"  SVM  → prob={svm_prob:.4f}  pred={'PATHOLOGICAL' if svm_pred else 'HEALTHY':13s} {svm_correct}")
        print(f"  LSTM → prob={lstm_prob:.4f}  pred={'PATHOLOGICAL' if lstm_pred else 'HEALTHY':13s} {lstm_correct}")
        print(f"  ENS  → prob={ensemble_prob:.4f}  pred={'PATHOLOGICAL' if ensemble_pred else 'HEALTHY':13s} {ens_correct}  risk={risk}")
        print(f"  F0 mean={global_feats.get('f0_mean',0):.1f}Hz  slope={f0_slope:.2f}  sinking={'YES ⚠' if sinking else 'No'}")
        print(f"  HNR drift={global_feats.get('hnr_early_late_diff',0):.2f}dB  shimmer growth={global_feats.get('shim_early_late_diff',0):.4f}")
        
        results.append({
            'record_id': record_id,
            'true_label': true_label,
            'category': sample['category'],
            'pathology': sample['pathology'],
            'svm_prob': svm_prob,
            'svm_pred': svm_pred,
            'lstm_prob': lstm_prob,
            'lstm_pred': lstm_pred,
            'ensemble_prob': ensemble_prob,
            'ensemble_pred': ensemble_pred,
            'risk': risk,
            'sinking_pitch': sinking,
            'f0_slope': f0_slope,
        })
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

if results:
    df_res = pd.DataFrame(results)
    
    # Accuracy per model
    for model in ['svm', 'lstm', 'ensemble']:
        correct = (df_res[f'{model}_pred'] == df_res['true_label']).sum()
        total   = len(df_res)
        print(f"  {model.upper():10s} accuracy: {correct}/{total} = {correct/total:.1%}")
    
    print(f"\n{'─'*70}")
    print(f"  {'Record':<12} {'True':>12} {'SVM prob':>10} {'LSTM prob':>10} {'Ens prob':>10} {'Risk':>8} {'Correct':>8}")
    print(f"{'─'*70}")
    for _, r in df_res.iterrows():
        true_str = "HEALTHY" if r['true_label'] == 0 else "PATHOL."
        ens_ok   = "✓" if r['ensemble_pred'] == r['true_label'] else "✗"
        print(f"  {r['record_id']:<12} {true_str:>12} {r['svm_prob']:>10.4f} {r['lstm_prob']:>10.4f} {r['ensemble_prob']:>10.4f} {r['risk']:>8} {ens_ok:>8}")
    
    # Healthy vs pathological prob distributions
    healthy_ens = df_res[df_res['true_label'] == 0]['ensemble_prob']
    patho_ens   = df_res[df_res['true_label'] == 1]['ensemble_prob']
    
    print(f"\n  Healthy samples    → mean ensemble prob: {healthy_ens.mean():.4f} (should be LOW)")
    print(f"  Pathological samples → mean ensemble prob: {patho_ens.mean():.4f} (should be HIGH)")
    
    # Separation check
    if len(healthy_ens) > 0 and len(patho_ens) > 0:
        separation = patho_ens.mean() - healthy_ens.mean()
        print(f"\n  Probability separation (pathol - healthy): {separation:.4f}")
        if separation > 0.2:
            print("  ✓ Good separation between classes")
        elif separation > 0.1:
            print("  ⚠ Moderate separation — model somewhat discriminative")
        else:
            print("  ✗ Poor separation — model may not be discriminating well")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
