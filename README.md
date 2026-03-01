# MG Voice Fatigue Detection Pipeline
### Using the VOICED Dataset to Detect Myasthenia Gravis Vocal Fatigue

---

## Overview

This pipeline detects **neuromuscular vocal fatigue** from the VOICED database
using two complementary approaches:

1. **SVM on global acoustic features** — standard voice pathology classifier
2. **Bidirectional LSTM on temporal features** — captures the MG-specific
   "sinking pitch sign" (F0 declining within a single sustained vowel)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Execution

### Step 0 — Download VOICED

```bash
wget -r -N -c -np https://physionet.org/files/voiced/1.0.0/
```

### Step 1 — Load and parse dataset metadata

```bash
# Edit VOICED_DIR in step1_data_loader.py first
python step1_data_loader.py
```

**Output:** `voiced_metadata.csv`  
Columns: record_id, pathology, pathology_category (healthy / structural / mg_like / other), gender, age, VHI score

---

### Step 1b — Convert .dat → .wav (optional but faster)

Add to step1:
```python
from step2_feature_extraction import convert_dat_to_wav
convert_dat_to_wav("./voiced", "./voiced_wav")
```

---

### Step 2 — Feature extraction

```bash
python step2_feature_extraction.py
```

**Output:**
- `voiced_features.csv` — one row per recording, ~120+ features
- `voiced_temporal.npy` — shape (208, 10, 19): temporal matrices

**Features extracted:**

| Category | Features | Why relevant to MG |
|---|---|---|
| F0 (pitch) | mean, std, range, slope, CV | Sinking pitch sign |
| Temporal F0 | slope, early-late diff, std growth | Core MG fatigue marker |
| Jitter | local, RAP, PPQ5 | Vocal fold irregularity |
| Shimmer | local, APQ3, APQ5, DDA, growth | Amplitude perturbation |
| HNR | mean, std, early-late drift | Breathiness increasing |
| MFCC | 20 coeff × mean/std/delta | Vocal tract shape |
| Spectral | centroid, flux, bandwidth, ZCR | Energy distribution |
| CPPS | mean, std, min | Glottic closure quality |
| Temporal | mfcc_drift_l2, voiced_fraction_decay | Trajectory degradation |

---

### Step 3 — Model training

```bash
python step3_model_training.py
```

**Models trained:**

**Model A — SVM (RBF kernel)**
- StandardScaler → ANOVA feature selection → SVM with class_weight='balanced'
- Validated with stratified 5-fold CV
- Metrics: AUC, weighted F1, balanced accuracy

**Model B — Bidirectional LSTM**
- Input: (batch, T=10 frames, F=19 features)
- Architecture: BiLSTM → LayerNorm → Temporal Attention → Dropout → Linear
- Temporal attention learns to upweight late-recording frames (fatigue phase)
- Training: Adam + CosineAnnealingLR, weighted cross-entropy for imbalance

**Model C — Stacked Ensemble**
- Out-of-fold SVM + LSTM probabilities as meta-features
- XGBoost classifier on top
- Prevents train/test leakage

**Output files:**
```
models/
  svm_model.pkl
  lstm_model.pt
  temporal_mean.npy, temporal_std.npy
  imputer.pkl
  feature_meta.json
results/
  cv_results.json
  svm_confmat.png
  lstm_confmat.png
  ensemble_confmat.png
  model_comparison.png
```

---

### Step 4 — Explainability

```bash
python step4_explainability.py
```

**Generates:**

1. `shap_summary.png` — SHAP beeswarm + bar chart
   - Validates that MG fatigue features (F0 slope, HNR drift) top the list

2. `fatigue_trajectories.png` — Per-sample F0/HNR/energy over 5 seconds
   - Healthy: stable trajectories
   - Pathological: declining F0, rising noise, energy decay

3. `feature_distributions.png` — Violin plots with Mann-Whitney p-values
   - Shows statistical separation between groups

4. `sinking_pitch_heatmap.png` — F0 colour map across all samples over time
   - Visual confirmation of the sinking pitch pattern

---

### Step 5 — Inference on new recording

```bash
python step5_inference.py --audio your_recording.wav
```

**Output:**
```
═══════════════════════════════════════════════════════════
  MG VOICE FATIGUE ANALYSIS REPORT
═══════════════════════════════════════════════════════════
  File             : recording.wav
  SVM Probability  : 0.7231
  LSTM Probability : 0.6847
  Ensemble Score   : 0.7039
  Risk Level       : HIGH
─────────────────────────────────────────────────────────
  F0 Mean          : 187.3 Hz
  F0 Slope (Hz/s)  : -12.45    ← negative = sinking pitch
  HNR Drift        : 3.21 dB   ← HNR getting worse
  Sinking Pitch    : YES ⚠
─────────────────────────────────────────────────────────
  ℹ ⚠ Sinking pitch detected: significant F0 decline within
    the phonation task, consistent with laryngeal muscle
    fatigue as seen in Myasthenia Gravis.
═══════════════════════════════════════════════════════════
```

Also generates: `results/inference_report_<filename>.png`

---

## Key MG Fatigue Features Explained

| Feature | Direction | Clinical Meaning |
|---|---|---|
| `temporal_f0_slope` | Negative | Sinking pitch sign — F0 falls during phonation |
| `f0_early_late_diff` | Positive | F0 higher at start than end of phonation |
| `hnr_early_late_diff` | Positive | Voice gets breathier toward end |
| `shim_early_late_diff` | Positive | Amplitude irregularity grows (muscle weakening) |
| `voiced_fraction_decay` | Positive | Phonation breaks down toward end |
| `f0_std_growth` | > 1.0 | Pitch instability increasing |
| `mfcc_drift_l2` | Large | Vocal tract shape changes over time |
| `cpps_min` | Low | Most breathy frame — glottic incompetence |

---

## Architecture Diagram

```
Audio (.wav / .dat)
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │              FEATURE EXTRACTION (Step 2)            │
  │                                                     │
  │  Global Features (120+)    Temporal Matrix (10×19)  │
  │  ─ F0 statistics           ─ F0 per 500ms window    │
  │  ─ Jitter, Shimmer         ─ HNR per window         │
  │  ─ HNR                     ─ Energy per window      │
  │  ─ MFCCs (20)              ─ MFCCs per window       │
  │  ─ Spectral features       ─ Early/late differences  │
  │  ─ CPPS                    ─ Trajectory slopes       │
  └──────────────────┬──────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    ┌──────────┐         ┌──────────────────┐
    │  SVM     │         │  Bidirectional   │
    │  (RBF)   │         │  LSTM + Attention│
    └────┬─────┘         └────────┬─────────┘
         │    probability         │  probability
         └──────────┬─────────────┘
                    ▼
           ┌────────────────┐
           │   XGBoost      │
           │   Ensemble     │
           └────────┬───────┘
                    ▼
         Fatigue Probability Score
         + Clinical Interpretation
```

---

## Literature Basis

| Feature | Reference |
|---|---|
| Sinking pitch sign | Walker FO (1997). *Neurology* 48:1135 |
| MG F0 statistics | Lavorato et al. (2017). *J Neurol Sci* |
| HNR for vocal fatigue | Gao et al. (2022). *Front Cell Dev Biol* |
| SVM for voice pathology | Cesari et al. (2018). *Comput Electr Eng* |
| BiLSTM for voice disorders | Various (2020–2024) |
| SHAP for clinical AI | Lundberg & Lee (2017). *NeurIPS* |

---

## Disclaimer

This tool is for **research purposes only**. It is not a clinical diagnostic
device and should not be used to diagnose or rule out Myasthenia Gravis.
All findings should be interpreted by qualified medical professionals.
