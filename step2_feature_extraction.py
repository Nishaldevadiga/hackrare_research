"""
STEP 2: FEATURE EXTRACTION
============================
Extracts two complementary feature sets from each 5-second /a/ vowel recording:

  A) GLOBAL acoustic features (clinically validated for MG):
       F0 stats, jitter, shimmer, HNR, MFCCs, CPPS, spectral features

  B) TEMPORAL fatigue trajectory features (MG-specific innovation):
       Per-frame F0/energy/HNR tracked over time → captures the
       "sinking pitch sign" — F0 falling within a single sustained vowel

Install requirements:
    pip install parselmouth librosa numpy pandas tqdm

Run after step1_data_loader.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import parselmouth                          # Praat wrapper
from parselmouth.praat import call
import librosa
from pathlib import Path
from tqdm import tqdm
import json


# ─── CONFIG ──────────────────────────────────────────────────────────────────
VOICED_DIR    = "./voice-icar-federico-ii-database-1.0.0"
METADATA_CSV  = "./voiced_metadata.csv"     # Output from Step 1
FEATURES_CSV  = "./voiced_features.csv"
TEMPORAL_NPY  = "./voiced_temporal.npy"     # shape: (N, T, F)

N_MFCC        = 20
N_FRAMES      = 10     # Divide 5s recording into 10 temporal windows (500ms each)
SR            = 8000
F0_MIN        = 75.0   # Hz — minimum F0 for voice analysis
F0_MAX        = 500.0  # Hz — maximum F0
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# PART A — GLOBAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def extract_f0_features(snd) -> dict:
    """
    Extract F0 (pitch) global statistics.
    Uses Praat autocorrelation method — best for modal voice.
    """
    pitch = snd.to_pitch(
        time_step=0.01,
        pitch_floor=F0_MIN,
        pitch_ceiling=F0_MAX
    )
    f0_values = pitch.selected_array["frequency"]
    f0_voiced = f0_values[f0_values > 0]   # exclude unvoiced frames

    if len(f0_voiced) < 5:
        return {k: np.nan for k in [
            "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
            "f0_slope", "f0_cv", "voiced_fraction"
        ]}

    # F0 slope (linear regression) — negative slope = sinking pitch
    times = np.linspace(0, 1, len(f0_voiced))
    slope = np.polyfit(times, f0_voiced, 1)[0]

    return {
        "f0_mean":         float(np.mean(f0_voiced)),
        "f0_std":          float(np.std(f0_voiced)),
        "f0_min":          float(np.min(f0_voiced)),
        "f0_max":          float(np.max(f0_voiced)),
        "f0_range":        float(np.max(f0_voiced) - np.min(f0_voiced)),
        "f0_slope":        float(slope),          # KEY: negative = sinking pitch
        "f0_cv":           float(np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-9)),
        "voiced_fraction": float(len(f0_voiced) / (len(f0_values) + 1e-9)),
    }


def extract_jitter_shimmer(snd) -> dict:
    """
    Extract perturbation measures (jitter, shimmer) via Praat.
    High jitter/shimmer → irregular vocal fold vibration.
    """
    point_process = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)

    try:
        jitter_local   = call(point_process, "Get jitter (local)",         0, 0, 0.0001, 0.02, 1.3)
        jitter_rap     = call(point_process, "Get jitter (rap)",            0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5    = call(point_process, "Get jitter (ppq5)",           0, 0, 0.0001, 0.02, 1.3)
        shimmer_local  = call([snd, point_process], "Get shimmer (local)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3   = call([snd, point_process], "Get shimmer (apq3)",   0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5   = call([snd, point_process], "Get shimmer (apq5)",   0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda    = call([snd, point_process], "Get shimmer (dda)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        return {k: np.nan for k in [
            "jitter_local", "jitter_rap", "jitter_ppq5",
            "shimmer_local", "shimmer_apq3", "shimmer_apq5", "shimmer_dda"
        ]}

    return {
        "jitter_local":  jitter_local,
        "jitter_rap":    jitter_rap,
        "jitter_ppq5":   jitter_ppq5,
        "shimmer_local": shimmer_local,
        "shimmer_apq3":  shimmer_apq3,
        "shimmer_apq5":  shimmer_apq5,
        "shimmer_dda":   shimmer_dda,
    }


def extract_hnr(snd) -> dict:
    """
    Harmonics-to-Noise Ratio.
    Low HNR → breathiness, noise → correlates with vocal fold closure issues in MG.
    """
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
        hnr_mean    = call(harmonicity, "Get mean", 0, 0)
        hnr_std     = call(harmonicity, "Get standard deviation", 0, 0)
        return {"hnr_mean": hnr_mean, "hnr_std": hnr_std}
    except Exception:
        return {"hnr_mean": np.nan, "hnr_std": np.nan}


def extract_mfcc_features(signal: np.ndarray, sr: int) -> dict:
    """
    Extract MFCC global statistics (mean + std per coefficient).
    MFCCs capture vocal tract shape changes — relevant for hypernasality in MG.
    """
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    
    feats = {}
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
        feats[f"mfcc{i+1}_std"]  = float(np.std(mfccs[i]))

    # Delta MFCCs (rate of change — captures temporal dynamics)
    delta  = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_delta_mean"]  = float(np.mean(delta[i]))
        feats[f"mfcc{i+1}_delta2_mean"] = float(np.mean(delta2[i]))

    return feats


def extract_spectral_features(signal: np.ndarray, sr: int) -> dict:
    """
    Spectral features: centroid, bandwidth, flux, rolloff, ZCR.
    """
    centroid   = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    bandwidth  = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    rolloff    = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    zcr        = librosa.feature.zero_crossing_rate(signal)[0]
    rms        = librosa.feature.rms(y=signal)[0]

    # Spectral flux (frame-to-frame change — sensitive to voice instability)
    stft = np.abs(librosa.stft(signal))
    flux = np.sqrt(np.mean(np.diff(stft, axis=1) ** 2, axis=0))

    return {
        "spec_centroid_mean":  float(np.mean(centroid)),
        "spec_centroid_std":   float(np.std(centroid)),
        "spec_bandwidth_mean": float(np.mean(bandwidth)),
        "spec_rolloff_mean":   float(np.mean(rolloff)),
        "zcr_mean":            float(np.mean(zcr)),
        "zcr_std":             float(np.std(zcr)),
        "rms_mean":            float(np.mean(rms)),
        "rms_std":             float(np.std(rms)),
        "spectral_flux_mean":  float(np.mean(flux)),
        "spectral_flux_std":   float(np.std(flux)),
    }


def extract_cpps(signal: np.ndarray, sr: int) -> dict:
    """
    Cepstral Peak Prominence Smoothed (CPPS).
    Strong indicator of breathiness / glottic incompetence — key in MG.
    
    Computed manually: cepstrum peak at quefrency corresponding to F0 range,
    smoothed with a polynomial fit.
    """
    # Compute power spectrum
    frame_len = 2048
    hop       = 512
    stft      = np.abs(librosa.stft(signal, n_fft=frame_len, hop_length=hop)) ** 2
    
    # Log power spectrum → cepstrum via IFFT
    log_spec  = np.log(stft + 1e-9)
    cepstrum  = np.real(np.fft.ifft(log_spec, axis=0))
    
    # Quefrency bins corresponding to F0_MIN–F0_MAX
    q_min = int(sr / F0_MAX)
    q_max = int(sr / F0_MIN)
    q_range = cepstrum[q_min:q_max, :]
    
    # Peak per frame
    cep_peak = np.max(q_range, axis=0)
    
    # Regression line (smoothed) subtracted from peak = CPP
    x = np.arange(len(cep_peak))
    poly = np.polyfit(x, cep_peak, 1)
    smoothed = np.polyval(poly, x)
    cpp_values = cep_peak - smoothed

    return {
        "cpps_mean": float(np.mean(cpp_values)),
        "cpps_std":  float(np.std(cpp_values)),
        "cpps_min":  float(np.min(cpp_values)),   # Worst frame (most breathy)
    }


def extract_global_features(record_path: str) -> dict:
    """
    Master function: extract ALL global features for one recording.
    
    Args:
        record_path: Path without extension (e.g. './voiced/voice001')
    Returns:
        dict of all features
    """
    # Load via Praat (parselmouth)
    snd    = parselmouth.Sound(f"{record_path}.wav")   # or .dat via wfdb+convert
    signal = snd.values[0]
    sr     = int(snd.sampling_frequency)

    feats = {}
    feats.update(extract_f0_features(snd))
    feats.update(extract_jitter_shimmer(snd))
    feats.update(extract_hnr(snd))
    feats.update(extract_mfcc_features(signal, sr))
    feats.update(extract_spectral_features(signal, sr))
    feats.update(extract_cpps(signal, sr))

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# PART B — TEMPORAL FATIGUE TRAJECTORY FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def extract_temporal_features(signal: np.ndarray, sr: int,
                               n_frames: int = N_FRAMES) -> np.ndarray:
    """
    Divide signal into N equal windows and compute per-frame features.
    This captures the WITHIN-RECORDING degradation pattern = MG fatigue signature.
    
    Features per frame:
        [f0_mean, f0_std, hnr, rms_energy, zcr, shimmer_proxy,
         mfcc1..mfcc13]   = 19 features × N_FRAMES
    
    Returns:
        array of shape (n_frames, 19)
    """
    frame_size = len(signal) // n_frames
    n_features = 19
    temporal   = np.zeros((n_frames, n_features))

    # Use parselmouth for F0/HNR per segment
    full_snd = parselmouth.Sound(signal.astype(np.float64), sr)

    for i in range(n_frames):
        start_sec = (i * frame_size) / sr
        end_sec   = ((i + 1) * frame_size) / sr
        seg       = signal[i * frame_size : (i + 1) * frame_size]

        if len(seg) < sr * 0.1:   # skip too-short segments
            continue

        # F0 for this segment
        seg_snd = full_snd.extract_part(
            from_time=start_sec, to_time=end_sec,
            preserve_times=False
        )
        pitch    = seg_snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0_vals  = pitch.selected_array["frequency"]
        f0_v     = f0_vals[f0_vals > 0]
        f0_mean  = float(np.mean(f0_v)) if len(f0_v) > 0 else 0.0
        f0_std   = float(np.std(f0_v))  if len(f0_v) > 1 else 0.0

        # HNR for segment
        try:
            harm   = call(seg_snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
            hnr_v  = call(harm, "Get mean", 0, 0)
            hnr_v  = hnr_v if np.isfinite(hnr_v) else 0.0
        except Exception:
            hnr_v = 0.0

        # Librosa features for segment
        rms   = float(np.sqrt(np.mean(seg ** 2)))
        zcr   = float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2)

        # Shimmer proxy: amplitude variation between consecutive 5ms windows
        win = int(0.005 * sr)
        amps = [np.max(np.abs(seg[j:j+win])) for j in range(0, len(seg)-win, win)]
        shim = float(np.std(amps) / (np.mean(amps) + 1e-9)) if len(amps) > 1 else 0.0

        # MFCCs for segment
        mfccs_seg = librosa.feature.mfcc(y=seg.astype(np.float32), sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs_seg, axis=1)   # shape: (13,)

        temporal[i, :6]  = [f0_mean, f0_std, hnr_v, rms, zcr, shim]
        temporal[i, 6:]  = mfcc_means

    return temporal   # (n_frames, 19)


def compute_fatigue_trajectory_stats(temporal: np.ndarray) -> dict:
    """
    Derive scalar features from the temporal matrix that directly encode
    the fatigue trajectory — these are the MG-specific features.
    
    All features compare EARLY (first 30%) vs LATE (last 30%) frames.
    """
    n = temporal.shape[0]
    n_early = max(1, int(n * 0.3))

    early = temporal[:n_early, :]
    late  = temporal[-n_early:, :]

    # Column indices
    F0_IDX, F0_STD_IDX, HNR_IDX, RMS_IDX, ZCR_IDX, SHIM_IDX = 0, 1, 2, 3, 4, 5

    feats = {}

    # ── F0 SLOPE (KEY: sinking pitch sign) ───────────────────────────────────
    f0_series = temporal[:, F0_IDX]
    f0_voiced = f0_series[f0_series > 0]
    if len(f0_voiced) >= 3:
        t = np.linspace(0, 1, len(f0_voiced))
        feats["temporal_f0_slope"] = float(np.polyfit(t, f0_voiced, 1)[0])
    else:
        feats["temporal_f0_slope"] = 0.0

    # ── EARLY vs LATE COMPARISONS ─────────────────────────────────────────────
    # Negative value = parameter worsening (fatigue) over time
    feats["f0_early_late_diff"]   = float(np.mean(early[:, F0_IDX])  - np.mean(late[:, F0_IDX]))
    feats["hnr_early_late_diff"]  = float(np.mean(early[:, HNR_IDX]) - np.mean(late[:, HNR_IDX]))
    feats["rms_early_late_diff"]  = float(np.mean(early[:, RMS_IDX]) - np.mean(late[:, RMS_IDX]))
    feats["shim_early_late_diff"] = float(np.mean(late[:, SHIM_IDX]) - np.mean(early[:, SHIM_IDX]))  # positive = worsening

    # ── INSTABILITY GROWTH ────────────────────────────────────────────────────
    feats["f0_std_growth"]  = float(np.mean(late[:, F0_STD_IDX]) / (np.mean(early[:, F0_STD_IDX]) + 1e-9))
    feats["zcr_growth"]     = float(np.mean(late[:, ZCR_IDX])    / (np.mean(early[:, ZCR_IDX])    + 1e-9))

    # ── MFCC DRIFT (vocal tract shape change over time) ───────────────────────
    mfcc_early = np.mean(early[:, 6:], axis=0)
    mfcc_late  = np.mean(late[:, 6:],  axis=0)
    feats["mfcc_drift_l2"] = float(np.linalg.norm(mfcc_late - mfcc_early))

    # ── VOICED FRACTION DECAY ─────────────────────────────────────────────────
    voiced_early = np.sum(early[:, F0_IDX] > 0) / n_early
    voiced_late  = np.sum(late[:, F0_IDX]  > 0) / n_early
    feats["voiced_fraction_decay"] = float(voiced_early - voiced_late)

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_features(metadata_csv: str, voiced_dir: str) -> pd.DataFrame:
    """
    Run feature extraction over the entire VOICED dataset.
    Saves:
        voiced_features.csv   — global + trajectory scalar features
        voiced_temporal.npy   — temporal matrices, shape (N, T, F)
    """
    df = pd.read_csv(metadata_csv)
    
    all_global_feats  = []
    all_temporal_mats = []
    valid_indices     = []

    print(f"Extracting features from {len(df)} recordings...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        record_id = row["record_id"]
        record_path = str(Path(voiced_dir) / record_id)

        try:
            # ── Load audio ────────────────────────────────────────────────────
            # Option A: if you've converted .dat → .wav (recommended)
            wav_path = f"{record_path}.wav"
            if Path(wav_path).exists():
                snd    = parselmouth.Sound(wav_path)
            else:
                # Option B: load via wfdb then wrap in parselmouth
                import wfdb
                rec    = wfdb.rdrecord(record_path)
                signal = rec.p_signal[:, 0].astype(np.float64)
                sr_    = rec.fs
                snd    = parselmouth.Sound(signal, sr_)

            signal = snd.values[0]
            sr_    = int(snd.sampling_frequency)

            # ── Global features ───────────────────────────────────────────────
            global_f = {}
            global_f.update(extract_f0_features(snd))
            global_f.update(extract_jitter_shimmer(snd))
            global_f.update(extract_hnr(snd))
            global_f.update(extract_mfcc_features(signal, sr_))
            global_f.update(extract_spectral_features(signal, sr_))
            global_f.update(extract_cpps(signal, sr_))

            # ── Temporal fatigue features ─────────────────────────────────────
            temporal_mat = extract_temporal_features(signal, sr_)
            traj_feats   = compute_fatigue_trajectory_stats(temporal_mat)
            global_f.update(traj_feats)

            global_f["record_id"] = record_id
            all_global_feats.append(global_f)
            all_temporal_mats.append(temporal_mat)
            valid_indices.append(idx)

        except Exception as e:
            print(f"  ✗ {record_id}: {e}")
            continue

    # Combine with metadata
    feat_df     = pd.DataFrame(all_global_feats)
    meta_subset = df.iloc[valid_indices].reset_index(drop=True)
    result_df   = pd.concat([meta_subset, feat_df.drop(columns=["record_id"])], axis=1)

    # Save
    result_df.to_csv(FEATURES_CSV, index=False)
    np.save(TEMPORAL_NPY, np.stack(all_temporal_mats, axis=0))

    print(f"\n✓ Global features  → {FEATURES_CSV}  ({result_df.shape})")
    print(f"✓ Temporal tensors → {TEMPORAL_NPY}   "
          f"shape={np.stack(all_temporal_mats).shape}")

    return result_df


# ─── HELPER: Convert VOICED .dat files to .wav (run once) ────────────────────
def convert_dat_to_wav(voiced_dir: str, output_dir: str = None):
    """
    Convert all VOICED .dat recordings to .wav using wfdb + scipy.
    This makes loading faster for subsequent runs.
    
    Install: pip install wfdb scipy
    """
    import wfdb
    from scipy.io import wavfile

    out_dir = Path(output_dir or voiced_dir)
    out_dir.mkdir(exist_ok=True)

    hea_files = sorted(Path(voiced_dir).glob("voice*.hea"))
    print(f"Converting {len(hea_files)} recordings to WAV...")

    for hea in tqdm(hea_files):
        record_stem = str(hea.with_suffix(""))
        out_wav     = out_dir / f"{hea.stem}.wav"

        if out_wav.exists():
            continue

        try:
            rec    = wfdb.rdrecord(record_stem)
            signal = rec.p_signal[:, 0]
            sr     = rec.fs

            # Normalise to int16
            signal_norm = signal / (np.max(np.abs(signal)) + 1e-9)
            signal_int  = (signal_norm * 32767).astype(np.int16)
            wavfile.write(str(out_wav), sr, signal_int)
        except Exception as e:
            print(f"  ✗ {hea.stem}: {e}")

    print(f"✓ WAV files saved to {out_dir}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Optional: convert .dat → .wav first (only needed once)
    # convert_dat_to_wav(VOICED_DIR)

    result_df = extract_all_features(METADATA_CSV, VOICED_DIR)
    print(f"\nFeature columns: {result_df.shape[1]}")
    print(result_df[["record_id", "pathology_category",
                      "f0_mean", "f0_slope", "temporal_f0_slope",
                      "hnr_mean", "jitter_local", "shimmer_local"]].head(10))
