"""
STEP 1: DATA LOADER
====================
Reads the VOICED dataset (WFDB format) and metadata from info .txt files.
Outputs a pandas DataFrame with paths, labels, and patient demographics.

Install requirements:
    pip install wfdb numpy pandas

Dataset structure expected:
    /voiced/
        voice001.dat, voice001.hea, voice001-info.txt
        voice002.dat, ...
        ...
"""

import os
import re
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path


# ─── CONFIG ──────────────────────────────────────────────────────────────────
VOICED_DIR = "./voice-icar-federico-ii-database-1.0.0"   # VOICED dataset path
OUTPUT_CSV  = "./voiced_metadata.csv"
SAMPLE_RATE = 8000               # All VOICED recordings are 8000 Hz
# ─────────────────────────────────────────────────────────────────────────────


def parse_info_file(info_path: str) -> dict:
    """Parse a VOICED *-info.txt file into a dictionary.

    VOICED info files use tab as separator (Key:\\tValue) so we split on tab
    first, falling back to colon for other key-value lines.
    """
    info = {}
    with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # Prefer tab separator (VOICED format)
            if "\t" in line:
                parts = line.split("\t", 1)
                key   = parts[0].strip().rstrip(":").lower().replace(" ", "_")
                value = parts[1].strip() if len(parts) > 1 else ""
                if key:
                    info[key] = value
            elif ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().lower().replace(" ", "_")
                if key:
                    info[key] = value.strip()
    return info


# Known neurological / neuromuscular pathologies in VOICED
# (muscle tension dysphonia, functional dysphonia excluded — structural only)
NEUROLOGICAL_KEYWORDS = [
    "myasthenia", "parkinson", "paralysis", "paresis",
    "neurological", "neuromuscular", "laryngeal", "spasmodic"
]

FATIGUE_RELEVANT_KEYWORDS = [
    # MG-like: fatigue, weakness, intermittent
    "myasthenia", "paralysis", "paresis", "hypophonia",
    "functional", "spasmodic dysphonia"
]


def classify_pathology(pathology_str: str) -> str:
    """
    Returns a broad category for the pathology label.

    VOICED dataset uses diagnoses like:
      "healthy", "hyperkinetic dysphonia", "hypokinetic dysphonia",
      "reflux laryngitis", etc.

    Categories:
        'healthy'          - no pathology
        'mg_like'          - hypokinetic / neuromuscular / fatigue-type disorders
        'structural'       - nodules, polyps, edema, cysts
        'other_pathology'  - remaining (e.g. reflux laryngitis, hyperkinetic)
    """
    if not pathology_str or pathology_str.lower().strip() in ["healthy", "none", ""]:
        return "healthy"

    p = pathology_str.lower().strip()

    # Neurological / neuromuscular — MG-like (hypokinetic = weak, fatigue pattern)
    if any(kw in p for kw in NEUROLOGICAL_KEYWORDS):
        return "mg_like"

    # Hypokinetic dysphonia — reduced vocal fold movement, weakness pattern
    if "hypokinetic" in p:
        return "mg_like"

    structural = ["nodule", "polyp", "edema", "reinke", "cyst", "granuloma",
                  "leukoplakia", "papilloma", "dysplasia", "prolapse", "cordite"]
    if any(kw in p for kw in structural):
        return "structural"

    return "other_pathology"


def load_voiced_dataset(voiced_dir: str) -> pd.DataFrame:
    """
    Walk the VOICED directory and build a metadata DataFrame.
    
    Returns columns:
        record_id, dat_path, hea_path, info_path,
        pathology, pathology_category, gender, age,
        vhi_score, rsi_score, smoking, alcohol, coffee
    """
    voiced_path = Path(voiced_dir)
    records = []

    # Find all .hea files (each = one recording)
    hea_files = sorted(voiced_path.glob("voice*.hea"))
    
    if not hea_files:
        raise FileNotFoundError(
            f"No voice*.hea files found in {voiced_dir}. "
            "Check your VOICED_DIR path."
        )

    print(f"Found {len(hea_files)} recordings in {voiced_dir}")

    for hea_file in hea_files:
        record_id = hea_file.stem                    # e.g. "voice001"
        info_file = hea_file.with_name(f"{record_id}-info.txt")
        dat_file  = hea_file.with_suffix(".dat")

        row = {
            "record_id": record_id,
            "dat_path":  str(dat_file),
            "hea_path":  str(hea_file),
            "info_path": str(info_file) if info_file.exists() else None,
        }

        # Parse metadata from info file
        if info_file.exists():
            info = parse_info_file(str(info_file))
            # VOICED uses "Diagnosis:" (tab-separated), mapped to "diagnosis" after normalisation
            row["pathology"]   = info.get("diagnosis", info.get("pathology", "unknown"))
            row["gender"]      = info.get("gender",    "unknown").upper()
            row["age"]         = _safe_int(info.get("age", None))
            # VHI key: "Voice Handicap Index (VHI) Score:" → "voice_handicap_index_(vhi)_score"
            vhi_key = next((k for k in info if "vhi" in k), None)
            rsi_key = next((k for k in info if "rsi" in k), None)
            row["vhi_score"]   = _safe_int(info.get(vhi_key, None) if vhi_key else None)
            row["rsi_score"]   = _safe_int(info.get(rsi_key, None) if rsi_key else None)
            row["smoking"]     = info.get("smoker",    info.get("smoking", "unknown")).lower()
            row["alcohol"]     = info.get("alcohol_consumption", info.get("alcohol", "unknown")).lower()
            row["coffee"]      = info.get("coffee",    "unknown").lower()
        else:
            row.update({
                "pathology": "unknown", "gender": "unknown",
                "age": None, "vhi_score": None, "rsi_score": None,
                "smoking": "unknown", "alcohol": "unknown", "coffee": "unknown"
            })

        row["pathology_category"] = classify_pathology(row["pathology"])
        
        # Binary label: 1 = pathological, 0 = healthy
        row["label_binary"] = 0 if row["pathology_category"] == "healthy" else 1
        
        records.append(row)

    df = pd.DataFrame(records)
    return df


def _safe_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def load_audio(record_path: str) -> tuple[np.ndarray, int]:
    """
    Load a VOICED audio recording using WFDB.
    
    Args:
        record_path: Path to record WITHOUT extension (e.g. './voiced/voice001')
    
    Returns:
        (signal_array, sample_rate)  — signal is 1D float64, normalised to [-1, 1]
    """
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0].astype(np.float64)   # first (only) channel
    fs     = record.fs

    # Normalise to [-1, 1]
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    return signal, fs


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_voiced_dataset(VOICED_DIR)
    
    print("\n── Dataset Summary ──────────────────────────────────")
    print(df["pathology_category"].value_counts().to_string())
    print(f"\nTotal records : {len(df)}")
    print(f"Healthy       : {(df['label_binary'] == 0).sum()}")
    print(f"Pathological  : {(df['label_binary'] == 1).sum()}")
    print(f"MG-like       : {(df['pathology_category'] == 'mg_like').sum()}")
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nMetadata saved → {OUTPUT_CSV}")
    
    # Quick sanity check: load first audio
    first = df.iloc[0]
    record_stem = str(Path(first["dat_path"]).with_suffix(""))
    sig, fs = load_audio(record_stem)
    print(f"\nSample audio: {first['record_id']}")
    print(f"  Duration : {len(sig)/fs:.2f}s   |  SR: {fs} Hz")
    print(f"  Pathology: {first['pathology']}")
    print(f"  Category : {first['pathology_category']}")
