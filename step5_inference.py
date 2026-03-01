"""
STEP 5: INFERENCE PIPELINE
============================
Predict MG fatigue likelihood from a NEW audio recording.

Usage:
    python step5_inference.py --audio path/to/recording.wav
    python step5_inference.py --audio path/to/voice.dat  (VOICED format)

Output:
    - Fatigue probability score (0–1)
    - Temporal trajectory plot showing within-recording degradation
    - Feature breakdown report
    - Clinical interpretation note

Install:
    pip install parselmouth librosa scikit-learn torch joblib matplotlib numpy
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import joblib
import json
from pathlib import Path


# ─── CONFIG ──────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "./models"
RESULTS_DIR  = "./results"
F0_MIN       = 75.0
F0_MAX       = 500.0
N_FRAMES     = 10
SR_EXPECTED  = 8000
# ─────────────────────────────────────────────────────────────────────────────


CLIP_SECONDS = 5.0   # must match pvqd_loader.py CLIP_SECONDS

def load_audio_file(audio_path: str) -> tuple:
    """
    Load audio from .wav or VOICED .dat format.
    Always resamples to SR_EXPECTED (8000 Hz) and clips to CLIP_SECONDS
    to match the sample rate and duration used during training.
    Returns: (signal_1d_float64, sample_rate)
    """
    import librosa
    path = Path(audio_path)

    if path.suffix.lower() == ".wav":
        signal, sr = librosa.load(str(path), sr=SR_EXPECTED, mono=True,
                                  duration=CLIP_SECONDS)
        signal = signal.astype(np.float64)
    elif path.suffix.lower() == ".dat":
        import wfdb
        record_stem = str(path.with_suffix(""))
        rec    = wfdb.rdrecord(record_stem)
        signal = rec.p_signal[:, 0].astype(np.float64)
        sr     = rec.fs
        # Resample if needed
        if sr != SR_EXPECTED:
            signal = librosa.resample(signal.astype(np.float32),
                                      orig_sr=sr, target_sr=SR_EXPECTED).astype(np.float64)
        # Clip
        max_samples = int(CLIP_SECONDS * SR_EXPECTED)
        signal = signal[:max_samples]
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .wav or .dat")

    # Normalise
    max_v = np.max(np.abs(signal))
    if max_v > 0:
        signal /= max_v

    return signal, SR_EXPECTED


def extract_features_for_inference(audio_path: str) -> tuple:
    """
    Extract both global and temporal features from a single audio file.
    Returns: (global_feature_dict, temporal_matrix (T, F))
    """
    import parselmouth
    from parselmouth.praat import call
    import librosa

    signal, sr = load_audio_file(audio_path)

    # ── Parselmouth sound object ──────────────────────────────────────────────
    snd = parselmouth.Sound(signal, sr)

    # ── F0 features ───────────────────────────────────────────────────────────
    pitch   = snd.to_pitch(time_step=0.01, pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f0_vals = pitch.selected_array["frequency"]
    f0_v    = f0_vals[f0_vals > 0]

    if len(f0_v) >= 3:
        t     = np.linspace(0, 1, len(f0_v))
        slope = float(np.polyfit(t, f0_v, 1)[0])
    else:
        slope = 0.0

    feats = {
        "f0_mean":   float(np.mean(f0_v)) if len(f0_v) > 0 else 0.0,
        "f0_std":    float(np.std(f0_v))  if len(f0_v) > 1 else 0.0,
        "f0_range":  float(np.ptp(f0_v))  if len(f0_v) > 0 else 0.0,
        "f0_slope":  slope,
        "f0_cv":     float(np.std(f0_v) / (np.mean(f0_v) + 1e-9)) if len(f0_v) > 0 else 0.0,
        "voiced_fraction": float(len(f0_v) / (len(f0_vals) + 1e-9)),
    }

    # ── Jitter / Shimmer ──────────────────────────────────────────────────────
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
        feats["jitter_local"]  = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        feats["jitter_rap"]    = call(pp, "Get jitter (rap)",   0, 0, 0.0001, 0.02, 1.3)
        feats["shimmer_local"] = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        feats["shimmer_apq3"]  = call([snd, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        feats["shimmer_dda"]   = call([snd, pp], "Get shimmer (dda)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        for k in ["jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq3", "shimmer_dda"]:
            feats[k] = 0.0

    # ── HNR ───────────────────────────────────────────────────────────────────
    try:
        harm         = call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
        feats["hnr_mean"] = call(harm, "Get mean", 0, 0)
        feats["hnr_std"]  = call(harm, "Get standard deviation", 0, 0)
    except Exception:
        feats["hnr_mean"] = 0.0
        feats["hnr_std"]  = 0.0

    # ── MFCCs ─────────────────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=signal.astype(np.float32), sr=sr, n_mfcc=20)
    for i in range(20):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
        feats[f"mfcc{i+1}_std"]  = float(np.std(mfccs[i]))
    delta = librosa.feature.delta(mfccs)
    for i in range(20):
        feats[f"mfcc{i+1}_delta_mean"] = float(np.mean(delta[i]))

    # ── Spectral features ─────────────────────────────────────────────────────
    feats["spec_centroid_mean"]  = float(np.mean(librosa.feature.spectral_centroid(y=signal.astype(np.float32), sr=sr)[0]))
    feats["zcr_mean"]            = float(np.mean(librosa.feature.zero_crossing_rate(signal)[0]))
    feats["rms_mean"]            = float(np.mean(librosa.feature.rms(y=signal.astype(np.float32))[0]))
    feats["spectral_flux_mean"]  = float(np.mean(np.diff(np.abs(librosa.stft(signal.astype(np.float32))), axis=1) ** 2))

    # ── Temporal trajectory ───────────────────────────────────────────────────
    frame_size = len(signal) // N_FRAMES
    temporal   = np.zeros((N_FRAMES, 19))

    for i in range(N_FRAMES):
        s_sec = (i * frame_size) / sr
        e_sec = ((i + 1) * frame_size) / sr
        seg   = signal[i * frame_size : (i + 1) * frame_size]

        seg_snd = snd.extract_part(from_time=s_sec, to_time=e_sec, preserve_times=False)
        p_seg   = seg_snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0_s    = p_seg.selected_array["frequency"]
        f0_sv   = f0_s[f0_s > 0]

        f0m = float(np.mean(f0_sv)) if len(f0_sv) > 0 else 0.0
        f0s = float(np.std(f0_sv))  if len(f0_sv) > 1 else 0.0

        try:
            h_ = call(seg_snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
            hnr_v = float(call(h_, "Get mean", 0, 0))
            hnr_v = hnr_v if np.isfinite(hnr_v) else 0.0
        except Exception:
            hnr_v = 0.0

        rms_v  = float(np.sqrt(np.mean(seg ** 2)))
        zcr_v  = float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2)
        win    = max(1, int(0.005 * sr))
        amps   = [np.max(np.abs(seg[j:j+win])) for j in range(0, max(1, len(seg)-win), win)]
        shim_v = float(np.std(amps) / (np.mean(amps) + 1e-9)) if len(amps) > 1 else 0.0

        mfcc_seg = librosa.feature.mfcc(y=seg.astype(np.float32), sr=sr, n_mfcc=13)
        mfcc_m   = np.mean(mfcc_seg, axis=1)

        temporal[i, :6] = [f0m, f0s, hnr_v, rms_v, zcr_v, shim_v]
        temporal[i, 6:] = mfcc_m

    # Trajectory features
    n_early = max(1, int(N_FRAMES * 0.3))
    early, late = temporal[:n_early, :], temporal[-n_early:, :]

    f0_series = temporal[:, 0]
    f0_voiced = f0_series[f0_series > 0]
    if len(f0_voiced) >= 3:
        t = np.linspace(0, 1, len(f0_voiced))
        feats["temporal_f0_slope"] = float(np.polyfit(t, f0_voiced, 1)[0])
    else:
        feats["temporal_f0_slope"] = 0.0

    feats["f0_early_late_diff"]    = float(np.mean(early[:, 0]) - np.mean(late[:, 0]))
    feats["hnr_early_late_diff"]   = float(np.mean(early[:, 2]) - np.mean(late[:, 2]))
    feats["rms_early_late_diff"]   = float(np.mean(early[:, 3]) - np.mean(late[:, 3]))
    feats["shim_early_late_diff"]  = float(np.mean(late[:, 5])  - np.mean(early[:, 5]))
    feats["f0_std_growth"]         = float(np.mean(late[:, 1])  / (np.mean(early[:, 1]) + 1e-9))
    feats["zcr_growth"]            = float(np.mean(late[:, 4])  / (np.mean(early[:, 4]) + 1e-9))
    feats["mfcc_drift_l2"]         = float(np.linalg.norm(np.mean(late[:, 6:], 0) - np.mean(early[:, 6:], 0)))
    feats["voiced_fraction_decay"] = float(
        np.sum(early[:, 0] > 0) / n_early - np.sum(late[:, 0] > 0) / n_early
    )
    feats["rms_std"] = float(np.std(temporal[:, 3]))
    feats["zcr_std"] = float(np.std(temporal[:, 4]))

    return feats, temporal


def predict(audio_path: str) -> dict:
    """
    Full inference for a single recording.
    Returns dict with scores, interpretation, and clinical notes.
    """
    # ── Load models ───────────────────────────────────────────────────────────
    svm_pipe = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
    imputer  = joblib.load(f"{OUTPUT_DIR}/imputer.pkl")
    with open(f"{OUTPUT_DIR}/feature_meta.json") as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]

    mean_  = np.load(f"{OUTPUT_DIR}/temporal_mean.npy")
    std_   = np.load(f"{OUTPUT_DIR}/temporal_std.npy")

    from step3_model_training import FatigueLSTM
    lstm = FatigueLSTM(input_size=19)
    lstm.load_state_dict(torch.load(f"{OUTPUT_DIR}/lstm_model.pt", map_location="cpu"))
    lstm.eval()

    # ── Extract features ──────────────────────────────────────────────────────
    print(f"\nAnalysing: {audio_path}")
    global_feats, temporal = extract_features_for_inference(audio_path)

    # ── SVM prediction ────────────────────────────────────────────────────────
    x_row = np.array([global_feats.get(c, 0.0) for c in feat_cols],
                     dtype=np.float32).reshape(1, -1)
    x_imp = imputer.transform(x_row)
    svm_prob = float(svm_pipe.predict_proba(x_imp)[0, 1])

    # ── LSTM prediction ───────────────────────────────────────────────────────
    temp_norm = (temporal - mean_) / (std_ + 1e-9)
    temp_norm = np.nan_to_num(temp_norm, nan=0.0)
    x_temp_t  = torch.FloatTensor(temp_norm).unsqueeze(0)   # (1, T, F)

    with torch.no_grad():
        logits    = lstm(x_temp_t)
        lstm_prob = float(torch.softmax(logits, dim=1)[0, 1])

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ensemble_prob = (svm_prob + lstm_prob) / 2.0

    # ── Load ROC-derived risk thresholds ──────────────────────────────────────
    # Thresholds are computed in combine_and_retrain.py using Youden's J
    # (high boundary) and the 95%-sensitivity point (low boundary).
    # Methodology: Van Calster B et al. (2019). "Three myths about risk
    # thresholds for prediction models." BMC Medicine, 17, 192.
    # https://doi.org/10.1186/s12916-019-1425-3
    thresh_path = Path(OUTPUT_DIR) / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as _tf:
            _thresh = json.load(_tf)
        low_thresh  = _thresh["low"]
        high_thresh = _thresh["high"]
    else:
        # Fallback until combine_and_retrain.py has been re-run
        low_thresh, high_thresh = 0.35, 0.65

    # ── Clinical interpretation ───────────────────────────────────────────────
    f0_slope      = global_feats.get("temporal_f0_slope", 0.0)
    hnr_drift     = global_feats.get("hnr_early_late_diff", 0.0)
    shim_growth   = global_feats.get("shim_early_late_diff", 0.0)
    sinking_pitch = f0_slope < -10   # Hz/normalised time

    risk = "LOW" if ensemble_prob < low_thresh else "MODERATE" if ensemble_prob < high_thresh else "HIGH"

    interpretation = {
        "sinking_pitch_detected": sinking_pitch,
        "f0_slope_hz_per_s":      round(f0_slope / 5, 2),   # approx Hz/s
        "hnr_drift_db":           round(hnr_drift, 2),
        "shimmer_growth":         round(shim_growth, 4),
        "f0_mean_hz":             round(global_feats.get("f0_mean", 0), 1),
        "voiced_fraction":        round(global_feats.get("voiced_fraction", 0), 3),
    }

    result = {
        "file":            audio_path,
        "svm_probability": round(svm_prob, 4),
        "lstm_probability": round(lstm_prob, 4),
        "ensemble_probability": round(ensemble_prob, 4),
        "risk_level":      risk,
        "interpretation":  interpretation,
        "clinical_note": (
            "⚠ Sinking pitch detected: significant F0 decline within the phonation task, "
            "consistent with laryngeal muscle fatigue as seen in Myasthenia Gravis."
            if sinking_pitch else
            "No sinking pitch detected. Vocal fatigue markers within normal limits."
        )
    }

    return result, global_feats, temporal


def plot_inference_report(result: dict, temporal: np.ndarray,
                           global_feats: dict, audio_path: str):
    """Generate a clinical-style single-page report plot."""
    time_axis = np.linspace(0, 5, N_FRAMES)
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_f0   = fig.add_subplot(gs[0, 0])
    ax_hnr  = fig.add_subplot(gs[0, 1])
    ax_shim = fig.add_subplot(gs[0, 2])
    ax_prob = fig.add_subplot(gs[1, 0])
    ax_feat = fig.add_subplot(gs[1, 1:])

    risk_color = {"LOW": "#2ecc71", "MODERATE": "#f39c12", "HIGH": "#e74c3c"}[result["risk_level"]]

    # ── F0 trajectory ─────────────────────────────────────────────────────────
    f0 = np.where(temporal[:, 0] > 0, temporal[:, 0], np.nan)
    ax_f0.plot(time_axis, f0, "o-", color="#3498db", lw=2, ms=6)
    valid = ~np.isnan(f0)
    if valid.sum() >= 3:
        coeffs = np.polyfit(time_axis[valid], f0[valid], 1)
        ax_f0.plot(time_axis, np.polyval(coeffs, time_axis), "--",
                   color=risk_color, lw=2, label=f"slope={coeffs[0]:.1f} Hz/s")
        ax_f0.legend(fontsize=8)
    ax_f0.set(title="F0 Trajectory\n(Sinking Pitch Sign)", xlabel="Time (s)", ylabel="F0 (Hz)")
    ax_f0.grid(True, alpha=0.3)

    # ── HNR trajectory ────────────────────────────────────────────────────────
    ax_hnr.plot(time_axis, temporal[:, 2], "o-", color="#9b59b6", lw=2, ms=6)
    ax_hnr.set(title="HNR Trajectory\n(Breathiness over time)", xlabel="Time (s)", ylabel="HNR (dB)")
    ax_hnr.grid(True, alpha=0.3)

    # ── Shimmer trajectory ────────────────────────────────────────────────────
    ax_shim.plot(time_axis, temporal[:, 5], "o-", color="#e67e22", lw=2, ms=6)
    ax_shim.set(title="Shimmer Trajectory\n(Amplitude irregularity)", xlabel="Time (s)", ylabel="Shimmer")
    ax_shim.grid(True, alpha=0.3)

    # ── Probability gauge ─────────────────────────────────────────────────────
    probs  = [result["svm_probability"], result["lstm_probability"], result["ensemble_probability"]]
    labels = ["SVM", "BiLSTM", "Ensemble"]
    bars   = ax_prob.barh(labels, probs, color=["#3498db", "#9b59b6", risk_color])
    ax_prob.axvline(0.5, color="gray", ls="--", lw=1)
    ax_prob.set_xlim(0, 1)
    ax_prob.set_xlabel("Pathology Probability")
    ax_prob.set_title(f"Model Predictions\nRisk: {result['risk_level']}", color=risk_color, fontweight="bold")
    for bar, prob in zip(bars, probs):
        ax_prob.text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.3f}", va="center", fontsize=10)
    ax_prob.grid(True, alpha=0.2, axis="x")

    # ── Feature summary table ─────────────────────────────────────────────────
    interp = result["interpretation"]
    rows = [
        ["F0 Mean",             f"{interp['f0_mean_hz']:.1f} Hz"],
        ["F0 Slope (Hz/s)",     f"{interp['f0_slope_hz_per_s']:.2f}",
         "⚠ SINKING" if interp["sinking_pitch_detected"] else "OK"],
        ["HNR Drift (dB)",      f"{interp['hnr_drift_db']:.2f}",
         "⚠ WORSENING" if interp['hnr_drift_db'] > 2 else "OK"],
        ["Shimmer Growth",      f"{interp['shimmer_growth']:.4f}",
         "⚠ HIGH" if interp['shimmer_growth'] > 0.05 else "OK"],
        ["Voiced Fraction",     f"{interp['voiced_fraction']:.3f}"],
        ["Ensemble Probability",f"{result['ensemble_probability']:.4f}"],
        ["Risk Level",          result['risk_level']],
    ]

    ax_feat.axis("off")
    table = ax_feat.table(
        cellText=[[r[0], r[1], r[2] if len(r) > 2 else ""] for r in rows],
        colLabels=["Feature", "Value", "Flag"],
        loc="center", cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Colour risk flags
    for i, row in enumerate(rows, 1):
        flag = row[2] if len(row) > 2 else ""
        color = "#ffcccc" if "⚠" in flag else "#ccffcc" if flag == "OK" else "white"
        table[i, 2].set_facecolor(color)

    ax_feat.set_title("Clinical Feature Summary", fontsize=10, fontweight="bold")

    # ── Title ─────────────────────────────────────────────────────────────────
    plt.suptitle(
        f"MG Voice Fatigue Analysis Report\n"
        f"File: {Path(audio_path).name}    |    "
        f"Risk: {result['risk_level']}    |    "
        f"Ensemble Score: {result['ensemble_probability']:.3f}",
        fontsize=12, fontweight="bold", color=risk_color, y=1.01
    )

    save = f"{RESULTS_DIR}/inference_report_{Path(audio_path).stem}.png"
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Inference report saved → {save}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN / CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MG Voice Fatigue Inference — Predict from a single audio file"
    )
    parser.add_argument(
        "--audio", type=str, required=True,
        help="Path to audio file (.wav or VOICED .dat)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip generating the report plot"
    )
    args = parser.parse_args()

    result, global_feats, temporal = predict(args.audio)

    print("\n" + "═" * 60)
    print("  MG VOICE FATIGUE ANALYSIS REPORT")
    print("═" * 60)
    print(f"  File             : {result['file']}")
    print(f"  SVM Probability  : {result['svm_probability']:.4f}")
    print(f"  LSTM Probability : {result['lstm_probability']:.4f}")
    print(f"  Ensemble Score   : {result['ensemble_probability']:.4f}")
    print(f"  Risk Level       : {result['risk_level']}")
    print("─" * 60)
    print(f"  F0 Mean          : {result['interpretation']['f0_mean_hz']:.1f} Hz")
    print(f"  F0 Slope (Hz/s)  : {result['interpretation']['f0_slope_hz_per_s']:.2f}")
    print(f"  HNR Drift        : {result['interpretation']['hnr_drift_db']:.2f} dB")
    print(f"  Sinking Pitch    : {'YES ⚠' if result['interpretation']['sinking_pitch_detected'] else 'No'}")
    print("─" * 60)
    print(f"  ℹ {result['clinical_note']}")
    print("═" * 60)
    print("\nDISCLAIMER: This is a research tool. Not for clinical diagnosis.")

    if not args.no_plot:
        plot_inference_report(result, temporal, global_feats, args.audio)


if __name__ == "__main__":
    main()
