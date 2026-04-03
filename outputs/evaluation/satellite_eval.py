"""
ORBITRON — Satellite Health Visualization Module
Generates: sensor correlation heatmap, PCA projection, reconstruction error histogram,
anomaly timeline, health index degradation, RUL predicted vs actual scatter.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
COLORS = {"primary": "#4FC3F7", "secondary": "#FF8A65", "accent": "#81C784",
           "warn": "#FFD54F", "danger": "#EF5350", "bg": "#0d1117"}


def plot_sensor_correlation(df, sensor_cols, output_dir):
    """Sensor correlation heatmap."""
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    corr = df[sensor_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Satellite Sensor Correlation Matrix", fontsize=16,
                 fontweight="bold", color="white", pad=15)
    ax.tick_params(colors="white", labelsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "satellite_sensor_correlation.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_pca_projection(df, sensor_cols, output_dir):
    """Telemetry PCA 2D projection colored by health index."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    X = df[sensor_cols].fillna(0).values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    color_col = df["health_index"].values if "health_index" in df.columns else \
        df["RUL"].values if "RUL" in df.columns else np.zeros(len(df))

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_col, cmap="RdYlGn",
                          s=8, alpha=0.6, edgecolors="none")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Health Index", fontsize=12, color="white")
    cbar.ax.tick_params(colors="white")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=13, color="white")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=13, color="white")
    ax.set_title("Telemetry PCA Projection", fontsize=15,
                 fontweight="bold", color="white", pad=15)
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "satellite_pca_projection.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_reconstruction_error(recon_errors, threshold, output_dir):
    """Reconstruction error histogram with anomaly threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    ax.hist(recon_errors, bins=60, color=COLORS["primary"], alpha=0.75,
            edgecolor="white", linewidth=0.4, label="Reconstruction Error")
    ax.axvline(x=threshold, color=COLORS["danger"], linewidth=2.5,
               linestyle="--", label=f"Threshold = {threshold:.4f}")

    n_anomalies = (recon_errors > threshold).sum()
    ax.fill_betweenx([0, ax.get_ylim()[1]], threshold, recon_errors.max(),
                      alpha=0.1, color=COLORS["danger"])
    ax.text(threshold * 1.05, ax.get_ylim()[1] * 0.85,
            f"{n_anomalies} anomalies", fontsize=12, color=COLORS["danger"],
            fontweight="bold")

    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=13, color="white")
    ax.set_ylabel("Frequency", fontsize=13, color="white")
    ax.set_title("Autoencoder Reconstruction Error Distribution", fontsize=15,
                 fontweight="bold", color="white", pad=15)
    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "satellite_recon_error_hist.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_anomaly_timeline(df, anomalies, output_dir, engine_id=None):
    """Anomaly detection timeline for a specific engine."""
    if engine_id is None:
        engine_id = df["engine_id"].unique()[0]

    mask = df["engine_id"] == engine_id
    engine_data = df[mask].reset_index(drop=True)
    engine_anomalies = anomalies[mask.values] if len(anomalies) == len(df) else \
        anomalies[:len(engine_data)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(COLORS["bg"])

    # Top: health index with anomaly markers
    ax1.set_facecolor("#111827")
    if "health_index" in engine_data.columns:
        ax1.plot(engine_data["cycle"], engine_data["health_index"],
                 color=COLORS["accent"], linewidth=1.5, label="Health Index")
        ax1.fill_between(engine_data["cycle"], engine_data["health_index"],
                          alpha=0.15, color=COLORS["accent"])

    anomaly_cycles = engine_data["cycle"].values[:len(engine_anomalies)]
    anomaly_mask = engine_anomalies[:len(anomaly_cycles)] == 1
    if anomaly_mask.any():
        hi_vals = engine_data["health_index"].values[:len(engine_anomalies)] if "health_index" in engine_data.columns else np.zeros(len(engine_anomalies))
        ax1.scatter(anomaly_cycles[anomaly_mask], hi_vals[anomaly_mask],
                    c=COLORS["danger"], s=20, zorder=5, label="Anomaly", marker="x")

    ax1.set_ylabel("Health Index", fontsize=12, color="white")
    ax1.set_title(f"Anomaly Detection Timeline — Engine {engine_id}", fontsize=15,
                  fontweight="bold", color="white", pad=15)
    ax1.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax1.tick_params(colors="white")

    # Bottom: anomaly flag bar
    ax2.set_facecolor("#111827")
    anom_vals = engine_anomalies[:len(anomaly_cycles)].astype(float)
    ax2.bar(anomaly_cycles, anom_vals, width=1, color=COLORS["danger"], alpha=0.7)
    ax2.set_ylabel("Anomaly", fontsize=12, color="white")
    ax2.set_xlabel("Cycle", fontsize=12, color="white")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "Anomaly"], fontsize=10, color="white")
    ax2.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "satellite_anomaly_timeline.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_health_degradation(df, output_dir, n_engines=5):
    """Health index degradation curves for multiple engines."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    if "health_index" not in df.columns:
        return None

    engines = df["engine_id"].unique()[:n_engines]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(engines)))

    for idx, eid in enumerate(engines):
        edata = df[df["engine_id"] == eid]
        ax.plot(edata["cycle"], edata["health_index"], color=cmap[idx],
                linewidth=1.8, alpha=0.85, label=f"Engine {eid}")

    ax.axhline(y=0.15, color=COLORS["danger"], linestyle="--", linewidth=1.5,
               alpha=0.7, label="Critical Threshold")
    ax.set_xlabel("Operating Cycle", fontsize=13, color="white")
    ax.set_ylabel("Health Index", fontsize=13, color="white")
    ax.set_title("Engine Health Degradation Curves", fontsize=15,
                 fontweight="bold", color="white", pad=15)
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#333",
              labelcolor="white", loc="upper right")
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "satellite_health_degradation.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_rul_scatter(y_true, y_pred, output_dir):
    """RUL predicted vs actual scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    y_true = np.minimum(np.asarray(y_true), 125)
    y_pred = np.minimum(np.asarray(y_pred), 125)

    ax.scatter(y_true, y_pred, c=COLORS["primary"], s=6, alpha=0.3, edgecolors="none")

    lim_max = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "--", color=COLORS["warn"],
            linewidth=2, label="Perfect Prediction")

    # Error bands
    ax.fill_between([0, lim_max], [0 - 15, lim_max - 15], [0 + 15, lim_max + 15],
                     alpha=0.08, color=COLORS["accent"], label="±15 cycles")

    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.92, f"MAE = {mae:.1f} cycles\nR² = {r2:.3f}",
            transform=ax.transAxes, fontsize=13, color="white",
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.4",
            facecolor="#1a1a2e", edgecolor="#333", alpha=0.9))

    ax.set_xlabel("Actual RUL (cycles)", fontsize=13, color="white")
    ax.set_ylabel("Predicted RUL (cycles)", fontsize=13, color="white")
    ax.set_title("Remaining Useful Life — Predicted vs Actual", fontsize=15,
                 fontweight="bold", color="white", pad=15)
    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "satellite_rul_scatter.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
def plot_sensor_trends(df, sensor_cols, output_dir, n_sensors=4, engine_id=None):
    """Sensor trend plots with smoothing."""
    if engine_id is None:
        engine_id = df["engine_id"].unique()[0]

    mask = df["engine_id"] == engine_id
    engine_data = df[mask].reset_index(drop=True)

    sensors_to_plot = sensor_cols[:n_sensors]
    
    fig, axes = plt.subplots(len(sensors_to_plot), 1, figsize=(12, 3 * len(sensors_to_plot)), sharex=True)
    if len(sensors_to_plot) == 1:
        axes = [axes]
    
    fig.patch.set_facecolor(COLORS["bg"])

    for idx, s in enumerate(sensors_to_plot):
        ax = axes[idx]
        ax.set_facecolor("#111827")
        
        y = engine_data[s].values
        x = engine_data["cycle"].values
        
        # Raw Data
        ax.plot(x, y, color=COLORS["secondary"], alpha=0.3, linewidth=1, label="Raw Data")
        
        # Smooth Data (Rolling Mean)
        window = min(10, max(1, len(y) // 10))
        y_smooth = engine_data[s].rolling(window=window, min_periods=1).mean().values
        ax.plot(x, y_smooth, color=COLORS["primary"], linewidth=2, label=f"Smoothed (Window={window})")

        ax.set_ylabel(s, fontsize=12, color="white")
        ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", loc="best")
        ax.tick_params(colors="white")

    axes[-1].set_xlabel("Cycle", fontsize=13, color="white")
    fig.suptitle(f"Sensor Trends with Smoothing — Engine {engine_id}", fontsize=16,
                 fontweight="bold", color="white", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "satellite_sensor_trends.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def generate_all_satellite_plots(train_df, test_df, sensor_cols, ae_model,
                                  threshold, rul_pred, anomalies,
                                  recon_errors, output_dir):
    """Generate all satellite health visualizations."""
    print("\n[SatelliteViz] Generating satellite health plots...")
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    paths.append(plot_sensor_correlation(train_df, sensor_cols, output_dir))
    paths.append(plot_pca_projection(train_df, sensor_cols, output_dir))

    if recon_errors is not None:
        paths.append(plot_reconstruction_error(recon_errors, threshold, output_dir))

    if anomalies is not None:
        paths.append(plot_anomaly_timeline(test_df, anomalies, output_dir))

    paths.append(plot_health_degradation(train_df, output_dir))
    paths.append(plot_sensor_trends(test_df, sensor_cols, output_dir))

    if rul_pred is not None and "RUL" in test_df.columns:
        y_true_rul = np.minimum(test_df["RUL"].values, 125)
        paths.append(plot_rul_scatter(y_true_rul, rul_pred, output_dir))

    paths = [p for p in paths if p is not None]
    print(f"  Generated {len(paths)} plots in {output_dir}")
    return paths
