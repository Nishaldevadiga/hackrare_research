"""
PVQD FEATURE EXTRACTION
========================
Extracts the same feature set as step2_feature_extraction.py from the
Perceptual Voice Qualities Database (PVQD) WAV files.

PVQD specifics:
  - 44100 Hz stereo/mono WAV recordings (resampled → 8000 Hz to match VOICED)
  - Labelled via Demographics.xlsx (Diagnosis column)
  - Long recordings (~27s): we use the first 5 seconds of voiced phonation

Outputs (same schema as VOICED step 2):
  pvqd_features.csv
  pvqd_temporal.npy   shape: (N, 10, 19)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import librosa
from pathlib import Path
from tqdm import tqdm

# ── import the feature-extraction functions from step 2 ──────────────────────
from step2_feature_extraction import (
    extract_f0_features,
    extract_jitter_shimmer,
    extract_hnr,
    extract_mfcc_features,
    extract_spectral_features,
    extract_cpps,
    extract_temporal_features,
    compute_fatigue_trajectory_stats,
    F0_MIN, F0_MAX, N_FRAMES,
)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PVQD_DIR      = "./Perceptual Voice Qualities Database (PVQD)"
AUDIO_DIR     = f"{PVQD_DIR}/Audio Files"
DEMO_XLSX     = f"{PVQD_DIR}/Ratings Spreadsheets/Demographics.xlsx"
TARGET_SR     = 8000          # resample to match VOICED
CLIP_SECONDS  = 5.0           # use first 5s of phonation

PVQD_FEATURES_CSV = "./pvqd_features.csv"
PVQD_TEMPORAL_NPY = "./pvqd_temporal.npy"
# ─────────────────────────────────────────────────────────────────────────────


# ── LABEL MAPPING ─────────────────────────────────────────────────────────────
def classify_pvqd_diagnosis(diagnosis) -> str:
    """Map PVQD Diagnosis column → same category scheme as VOICED."""
    if pd.isna(diagnosis):
        return "unknown"
    d = str(diagnosis).strip()
    if d == "N":
        return "healthy"
    dl = d.lower()
    mg_kw = ["paralysis", "paresis", "atrophy", "parkinson", "tremor",
              "spasmodic", "hypomobility", "bilateral vf"]
    if any(k in dl for k in mg_kw):
        return "mg_like"
    struct_kw = ["lesion", "polyp", "nodule", "edema", "reinke", "scar",
                 "cyst", "granuloma", "leukoplakia", "carcinoma", "cancer",
                 "stenosis", "phonotrauma", "candida", "laryngitis"]
    if any(k in dl for k in struct_kw):
        return "structural"
    return "other_pathology"


def load_pvqd_metadata() -> pd.DataFrame:
    """Load Demographics.xlsx and match to WAV files."""
    demo = pd.read_excel(DEMO_XLSX)
    demo.columns = [c.strip() for c in demo.columns]
    demo["Participant ID"] = demo["Participant ID"].astype(str).str.strip()

    audio_path = Path(AUDIO_DIR)
    # Build mapping: participant_id → wav_path
    # File names: "BL01 ENSS.wav" or "BL01_ENSS.wav" → participant "BL01"
    wav_map = {}
    for wav in audio_path.glob("*.wav"):
        stem = wav.stem                             # e.g. "BL01 ENSS" or "BL01_ENSS"
        pid  = stem.replace("_ENSS", "").replace(" ENSS", "").replace("ENSS","").strip()
        wav_map[pid.upper()] = str(wav)

    rows = []
    for _, row in demo.iterrows():
        raw_pid = row["Participant ID"]
        if pd.isna(raw_pid):
            continue
        pid = str(raw_pid).strip().upper()
        wav = wav_map.get(pid)
        if wav is None:
            continue                                # no audio for this participant
        category = classify_pvqd_diagnosis(row.get("Diagnosis", None))
        if category == "unknown":
            continue                                # skip unlabelled
        rows.append({
            "record_id":          pid,
            "wav_path":           wav,
            "pathology":          str(row.get("Diagnosis", "")).strip(),
            "pathology_category": category,
            "gender":             str(row.get("Gender", "")).strip().upper(),
            "age":                row.get("Age", None),
            "label_binary":       0 if category == "healthy" else 1,
            "source":             "pvqd",
        })

    df = pd.DataFrame(rows)
    print(f"PVQD: {len(df)} usable recordings")
    print(df["pathology_category"].value_counts().to_string())
    return df


# ── AUDIO LOADING ─────────────────────────────────────────────────────────────
def load_pvqd_audio(wav_path: str) -> tuple:
    """
    Load PVQD WAV, resample to TARGET_SR, clip to CLIP_SECONDS.
    Returns (signal_float64, sample_rate).
    """
    signal, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True, duration=CLIP_SECONDS)
    signal = signal.astype(np.float64)
    # Normalise to [-1, 1]
    mx = np.max(np.abs(signal))
    if mx > 0:
        signal /= mx
    return signal, TARGET_SR


# ── FEATURE EXTRACTION LOOP ──────────────────────────────────────────────────
def extract_pvqd_features(df: pd.DataFrame):
    """
    Run the same global + temporal feature extraction as VOICED step 2.
    """
    all_global  = []
    all_temporal = []
    valid_idx   = []

    print(f"\nExtracting features from {len(df)} PVQD recordings...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            signal, sr = load_pvqd_audio(row["wav_path"])

            if len(signal) < sr * 1.0:          # skip if < 1 s after clip
                continue

            snd = parselmouth.Sound(signal, sr)

            global_f = {}
            global_f.update(extract_f0_features(snd))
            global_f.update(extract_jitter_shimmer(snd))
            global_f.update(extract_hnr(snd))
            global_f.update(extract_mfcc_features(signal, sr))
            global_f.update(extract_spectral_features(signal, sr))
            global_f.update(extract_cpps(signal, sr))

            temporal_mat = extract_temporal_features(signal, sr)
            global_f.update(compute_fatigue_trajectory_stats(temporal_mat))

            global_f["record_id"] = row["record_id"]
            all_global.append(global_f)
            all_temporal.append(temporal_mat)
            valid_idx.append(idx)

        except Exception as e:
            print(f"  ✗ {row['record_id']}: {e}")
            continue

    feat_df     = pd.DataFrame(all_global)
    meta_subset = df.iloc[valid_idx].reset_index(drop=True)
    result_df   = pd.concat(
        [meta_subset, feat_df.drop(columns=["record_id"])], axis=1
    )

    result_df.to_csv(PVQD_FEATURES_CSV, index=False)
    np.save(PVQD_TEMPORAL_NPY, np.stack(all_temporal, axis=0))

    print(f"\n✓ PVQD features  → {PVQD_FEATURES_CSV}  {result_df.shape}")
    print(f"✓ PVQD temporal  → {PVQD_TEMPORAL_NPY}   "
          f"shape={np.stack(all_temporal).shape}")
    return result_df


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    meta_df = load_pvqd_metadata()
    feat_df = extract_pvqd_features(meta_df)

    print("\n── PVQD Label Summary ───────────────────────────────")
    print(feat_df["pathology_category"].value_counts().to_string())
    print(f"\nHealthy      : {(feat_df['label_binary']==0).sum()}")
    print(f"Pathological : {(feat_df['label_binary']==1).sum()}")
