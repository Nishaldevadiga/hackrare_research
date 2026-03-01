"""
STEP 4: EXPLAINABILITY & FATIGUE VISUALIZATION
================================================
Two analysis tracks:

  A) SHAP feature importance for the SVM model
       → Which acoustic features most differentiate pathological from healthy
       → Validates that MG-relevant features (F0 slope, HNR drift) top the list

  B) Temporal fatigue trajectory plots
       → Visualise frame-by-frame F0/HNR/energy for individual recordings
       → Produces the "sinking pitch sign" plot directly from audio
       → Compare healthy vs pathological trajectory shapes

Install:
    pip install shap matplotlib seaborn pandas numpy scikit-learn joblib
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import joblib
from pathlib import Path


# ─── CONFIG ──────────────────────────────────────────────────────────────────
FEATURES_CSV = "./voiced_features.csv"
TEMPORAL_NPY = "./voiced_temporal.npy"
OUTPUT_DIR   = "./models"
RESULTS_DIR  = "./results"

Path(RESULTS_DIR).mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

# Temporal feature names (matches step2 extraction order)
TEMPORAL_FEAT_NAMES = [
    "F0 Mean", "F0 Std", "HNR", "RMS Energy", "ZCR", "Shimmer Proxy",
    "MFCC1", "MFCC2", "MFCC3", "MFCC4", "MFCC5",
    "MFCC6", "MFCC7", "MFCC8", "MFCC9", "MFCC10",
    "MFCC11", "MFCC12", "MFCC13"
]


# ══════════════════════════════════════════════════════════════════════════════
# PART A — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def load_global_features(features_csv: str, output_dir: str) -> tuple:
    """Load feature matrix and model pipeline."""
    import json
    
    df       = pd.read_csv(features_csv)
    pipeline = joblib.load(f"{output_dir}/svm_model.pkl")
    imputer  = joblib.load(f"{output_dir}/imputer.pkl")

    with open(f"{output_dir}/feature_meta.json") as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]

    X = df[feat_cols].values.astype(np.float32)
    X = imputer.transform(X)
    y = df["label_binary"].values

    return X, y, feat_cols, pipeline, df


def compute_shap_values(pipeline, X: np.ndarray,
                         feature_names: list) -> np.ndarray:
    """
    Compute SHAP values for the SVM pipeline.
    Uses KernelExplainer (model-agnostic, works for any sklearn pipeline).
    
    For large datasets, subsample background to 50 samples for speed.
    """
    # Extract the predict_proba function through the full pipeline
    predict_fn = lambda x: pipeline.predict_proba(x)

    background = shap.kmeans(X, 50)   # Summarise background data
    explainer  = shap.KernelExplainer(predict_fn, background)
    
    # Compute SHAP for all samples (takes a few minutes on 208 samples)
    print("Computing SHAP values (this may take 3–5 minutes)...")
    shap_values = explainer.shap_values(X, nsamples=100)
    
    # shap_values[1] = SHAP for class 1 (pathological)
    return shap_values[1]   # (N, n_features)


def plot_shap_summary(shap_vals: np.ndarray, X: np.ndarray,
                       feature_names: list, top_n: int = 30):
    """Beeswarm plot of top N features by mean |SHAP|."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-top_n:][::-1]

    top_shap = shap_vals[:, top_idx]
    top_X    = X[:, top_idx]
    top_names = [feature_names[i] for i in top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # ── Left: Mean |SHAP| bar chart ──────────────────────────────────────────
    ax = axes[0]
    colors = ["#e74c3c" if "f0_slope" in n or "temporal" in n or "drift" in n
              else "#3498db" for n in top_names]
    ax.barh(range(top_n), mean_abs[top_idx][::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Feature Importance (SHAP)")
    ax.axvline(0, color="black", lw=0.5)

    # Highlight MG-specific features
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#e74c3c", label="MG Fatigue Trajectory Features"),
        Patch(facecolor="#3498db", label="Standard Acoustic Features")
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)

    # ── Right: Beeswarm-style SHAP scatter ────────────────────────────────────
    ax = axes[1]
    # Normalise X for colour mapping
    X_norm = (top_X - top_X.min(0)) / (top_X.ptp(0) + 1e-9)

    for row_i, feat_i in enumerate(range(top_n)):
        y_jitter = np.random.uniform(-0.3, 0.3, size=len(shap_vals))
        sc = ax.scatter(
            top_shap[:, feat_i],
            np.full(len(shap_vals), top_n - 1 - row_i) + y_jitter,
            c=X_norm[:, feat_i], cmap="RdBu_r",
            alpha=0.6, s=15, vmin=0, vmax=1
        )

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value → Impact on Pathological Prediction")
    ax.set_title("SHAP Beeswarm (red = high feature value)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Feature Value (normalised)")

    plt.suptitle("SHAP Explainability — MG Voice Fatigue Detection", y=1.01,
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save = f"{RESULTS_DIR}/shap_summary.png"
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ SHAP summary saved → {save}")

    # Print top 10
    print("\n── Top 10 Predictive Features ─────────────────────────────────")
    for i, (name, val) in enumerate(zip(top_names[:10], mean_abs[top_idx[:10]])):
        print(f"  {i+1:2d}. {name:40s}  SHAP={val:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — TEMPORAL FATIGUE TRAJECTORY PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_single_fatigue_trajectory(temporal_row: np.ndarray,
                                    record_id: str,
                                    label: str,
                                    ax_f0, ax_hnr, ax_rms):
    """Plot F0, HNR, and energy trajectories for one recording."""
    n_frames  = temporal_row.shape[0]
    time_axis = np.linspace(0, 5, n_frames)   # 5-second recording

    color = "#e74c3c" if label == "pathological" else "#2ecc71"
    alpha = 0.8
    lw    = 2

    # F0
    f0 = temporal_row[:, 0]
    f0_voiced = np.where(f0 > 0, f0, np.nan)
    ax_f0.plot(time_axis, f0_voiced, color=color, lw=lw, alpha=alpha,
               label=f"{record_id} ({label})")
    
    # Trend line (sinking pitch)
    valid = ~np.isnan(f0_voiced)
    if valid.sum() >= 3:
        t_v = time_axis[valid]
        f_v = f0_voiced[valid]
        coeffs = np.polyfit(t_v, f_v, 1)
        ax_f0.plot(time_axis, np.polyval(coeffs, time_axis),
                   "--", color=color, lw=1, alpha=0.5)

    # HNR
    ax_hnr.plot(time_axis, temporal_row[:, 2], color=color, lw=lw, alpha=alpha)

    # RMS Energy
    ax_rms.plot(time_axis, temporal_row[:, 3], color=color, lw=lw, alpha=alpha)


def plot_fatigue_trajectories(df: pd.DataFrame, X_temp: np.ndarray,
                               n_healthy: int = 3, n_path: int = 3):
    """
    Compare temporal trajectories of healthy vs pathological voices.
    Highlights the sinking pitch sign in pathological recordings.
    """
    healthy_idx = df.index[df["label_binary"] == 0].tolist()[:n_healthy]
    path_idx    = df.index[df["label_binary"] == 1].tolist()[:n_path]

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(3, 1, hspace=0.4)
    
    ax_f0  = fig.add_subplot(gs[0])
    ax_hnr = fig.add_subplot(gs[1])
    ax_rms = fig.add_subplot(gs[2])

    for idx in healthy_idx:
        row = df.iloc[idx]
        plot_single_fatigue_trajectory(
            X_temp[idx], row["record_id"], "healthy",
            ax_f0, ax_hnr, ax_rms
        )

    for idx in path_idx:
        row = df.iloc[idx]
        plot_single_fatigue_trajectory(
            X_temp[idx], row["record_id"], "pathological",
            ax_f0, ax_hnr, ax_rms
        )

    # Style axes
    for ax, ylabel, title in [
        (ax_f0,  "F0 (Hz)",   "Fundamental Frequency Trajectory\n"
                              "Dashed line = trend (negative slope = sinking pitch sign)"),
        (ax_hnr, "HNR (dB)",  "Harmonics-to-Noise Ratio Trajectory\n"
                              "(Decrease = increasing breathiness over time)"),
        (ax_rms, "RMS Energy","Signal Energy Trajectory\n"
                              "(Decay = vocal muscle fatigue)")
    ]:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(loc="upper right", fontsize=8)
        ax.axvline(2.5, color="gray", lw=0.8, ls=":", alpha=0.5,
                   label="Midpoint (2.5s)")
        ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2ecc71", lw=2, label="Healthy"),
        Line2D([0], [0], color="#e74c3c", lw=2, label="Pathological"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=10)

    plt.suptitle("Temporal Voice Fatigue Trajectories — Healthy vs Pathological\n"
                 "(5-second sustained vowel /a/ divided into 10 analysis frames)",
                 y=1.03, fontsize=12, fontweight="bold")

    save = f"{RESULTS_DIR}/fatigue_trajectories.png"
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Fatigue trajectory plot saved → {save}")


def plot_feature_distributions(df: pd.DataFrame):
    """
    Violin plots of the most MG-relevant features: healthy vs pathological.
    """
    features_to_plot = [
        ("temporal_f0_slope", "Temporal F0 Slope\n(Sinking Pitch Sign)"),
        ("f0_slope",          "Global F0 Slope"),
        ("hnr_early_late_diff", "HNR Early→Late Diff\n(Breathiness Increase)"),
        ("f0_std_growth",     "F0 Instability Growth"),
        ("shimmer_local",     "Shimmer (Local)"),
        ("voiced_fraction_decay", "Voiced Fraction Decay"),
        ("cpps_mean",         "CPPS (Cepstral Peak Prominence)"),
        ("mfcc_drift_l2",     "MFCC Drift (L2)"),
    ]

    available = [(f, t) for f, t in features_to_plot if f in df.columns]
    n_plots   = len(available)
    n_cols    = 4
    n_rows    = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    palette = {"healthy": "#2ecc71", "pathological": "#e74c3c"}
    df_plot = df.copy()
    df_plot["Group"] = df_plot["label_binary"].map({0: "healthy", 1: "pathological"})

    for ax, (feat, title) in zip(axes, available):
        data_clean = df_plot.dropna(subset=[feat])
        sns.violinplot(
            data=data_clean, x="Group", y=feat,
            palette=palette, ax=ax, inner="box", cut=0
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Add p-value annotation
        from scipy import stats
        h_vals = data_clean[data_clean["Group"] == "healthy"][feat].values
        p_vals = data_clean[data_clean["Group"] == "pathological"][feat].values
        if len(h_vals) > 2 and len(p_vals) > 2:
            _, pval = stats.mannwhitneyu(h_vals, p_vals, alternative="two-sided")
            stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.set_title(f"{title}\n(p={pval:.3f} {stars})", fontsize=8)

    # Hide unused subplots
    for ax in axes[len(available):]:
        ax.set_visible(False)

    plt.suptitle("MG Fatigue Feature Distributions — Healthy vs Pathological",
                 y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()

    save = f"{RESULTS_DIR}/feature_distributions.png"
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Feature distribution plots saved → {save}")


def plot_sinking_pitch_heatmap(X_temp: np.ndarray, y: np.ndarray,
                                n_samples: int = 40):
    """
    Heatmap of F0 trajectories (one row per sample).
    Separates healthy from pathological — shows the sinking pattern clearly.
    """
    f0_matrix = X_temp[:, :, 0]   # shape: (N, T)

    # Select samples
    h_idx = np.where(y == 0)[0][:n_samples // 2]
    p_idx = np.where(y == 1)[0][:n_samples // 2]
    idx   = np.concatenate([h_idx, p_idx])
    
    f0_sub  = f0_matrix[idx]
    labels  = y[idx]

    # Normalise per-row to [0, 1]
    row_min = f0_sub.min(axis=1, keepdims=True)
    row_max = f0_sub.max(axis=1, keepdims=True)
    f0_norm = (f0_sub - row_min) / (row_max - row_min + 1e-9)

    n_h = len(h_idx)
    n_p = len(p_idx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                             gridspec_kw={"width_ratios": [n_h, n_p]})

    for ax, grp_idx, title in [
        (axes[0], slice(n_h),    f"Healthy (n={n_h})"),
        (axes[1], slice(n_h, n_h + n_p), f"Pathological (n={n_p})")
    ]:
        im = ax.imshow(
            f0_norm[grp_idx].T,
            aspect="auto", origin="lower",
            cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_xlabel("Recording Index")
        ax.set_ylabel("Time Frame (0=onset, 9=end of phonation)")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label="Normalised F0")

    plt.suptitle("F0 Trajectory Heatmap\n"
                 "Green = high F0 | Red = low F0\n"
                 "Sinking pitch (green→red over time frames) = MG fatigue marker",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    save = f"{RESULTS_DIR}/sinking_pitch_heatmap.png"
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Sinking pitch heatmap saved → {save}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data...")
    df     = pd.read_csv(FEATURES_CSV)
    X_temp = np.load(TEMPORAL_NPY)
    y      = df["label_binary"].values

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    print("\n── PART A: SHAP Explainability ──────────────────────────────────")
    X_global, y_, feat_cols, pipeline, _ = load_global_features(
        FEATURES_CSV, OUTPUT_DIR
    )
    shap_vals = compute_shap_values(pipeline, X_global, feat_cols)
    plot_shap_summary(shap_vals, X_global, feat_cols)

    # ── Temporal visualisations ───────────────────────────────────────────────
    print("\n── PART B: Temporal Fatigue Visualisations ─────────────────────")
    plot_fatigue_trajectories(df, X_temp, n_healthy=4, n_path=4)
    plot_feature_distributions(df)
    plot_sinking_pitch_heatmap(X_temp, y, n_samples=40)

    print(f"\n✓ All plots saved to {RESULTS_DIR}/")
    print("\nGenerated files:")
    for p in sorted(Path(RESULTS_DIR).glob("*.png")):
        print(f"  {p}")
