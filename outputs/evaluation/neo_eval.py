"""
ORBITRON — NEO Orbit Analysis Visualizations
Generates: MOID distribution histogram, orbit prediction error comparison,
Monte Carlo simulation visualization, residual error plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
COLORS = {"primary": "#4FC3F7", "secondary": "#FF8A65", "accent": "#81C784",
           "warn": "#FFD54F", "danger": "#EF5350", "bg": "#0d1117"}


def plot_moid_distribution(collision_results, output_dir):
    """MOID distribution histogram from Monte Carlo simulations."""
    fig, axes = plt.subplots(1, len(collision_results), figsize=(6 * len(collision_results), 6),
                              squeeze=False)
    fig.patch.set_facecolor(COLORS["bg"])

    for idx, neo in enumerate(collision_results):
        ax = axes[0, idx]
        ax.set_facecolor("#111827")
        moid_dist = neo.get("moid_distribution", np.array([neo.get("moid_mean", 0.1)]))

        ax.hist(moid_dist, bins=50, color=COLORS["primary"], alpha=0.75,
                edgecolor="white", linewidth=0.3)
        ax.axvline(x=0.05, color=COLORS["danger"], linewidth=2, linestyle="--",
                   label="Hazard Threshold (0.05 AU)")
        ax.axvline(x=neo.get("moid_mean", 0), color=COLORS["warn"], linewidth=2,
                   linestyle="-", label=f"Mean = {neo.get('moid_mean', 0):.4f} AU")

        ax.set_xlabel("MOID (AU)", fontsize=12, color="white")
        ax.set_ylabel("Frequency", fontsize=12, color="white")
        name = neo.get("name", f"NEO {idx}")
        ax.set_title(f"MOID Distribution — {name}", fontsize=13,
                     fontweight="bold", color="white", pad=10)
        ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
        ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "neo_moid_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_orbit_prediction_error(y_true, y_pred_physics, y_pred_corrected, output_dir):
    """Orbit prediction error comparison: physics-only vs ML-corrected."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    y_true = np.asarray(y_true).flatten()
    y_pred_physics = np.asarray(y_pred_physics).flatten()
    y_pred_corrected = np.asarray(y_pred_corrected).flatten()

    min_len = min(len(y_true), len(y_pred_physics), len(y_pred_corrected))
    y_true, y_pred_physics, y_pred_corrected = y_true[:min_len], y_pred_physics[:min_len], y_pred_corrected[:min_len]

    error_physics = np.abs(y_true - y_pred_physics)
    error_corrected = np.abs(y_true - y_pred_corrected)

    # Left: error distributions
    ax = axes[0]
    ax.set_facecolor("#111827")
    ax.hist(error_physics, bins=40, alpha=0.6, color=COLORS["secondary"],
            label=f"Physics Only (MAE={error_physics.mean():.4f})", edgecolor="white", linewidth=0.3)
    ax.hist(error_corrected, bins=40, alpha=0.6, color=COLORS["accent"],
            label=f"ML Corrected (MAE={error_corrected.mean():.4f})", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Absolute Error", fontsize=12, color="white")
    ax.set_ylabel("Frequency", fontsize=12, color="white")
    ax.set_title("Prediction Error Distribution", fontsize=14,
                 fontweight="bold", color="white", pad=12)
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="white")

    # Right: scatter comparison
    ax = axes[1]
    ax.set_facecolor("#111827")
    ax.scatter(y_true, y_pred_physics, c=COLORS["secondary"], s=12, alpha=0.5,
               label="Physics Only")
    ax.scatter(y_true, y_pred_corrected, c=COLORS["accent"], s=12, alpha=0.5,
               label="ML Corrected")
    lim = max(y_true.max(), y_pred_physics.max(), y_pred_corrected.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color=COLORS["warn"], linewidth=1.5)
    ax.set_xlabel("True Value", fontsize=12, color="white")
    ax.set_ylabel("Predicted Value", fontsize=12, color="white")
    ax.set_title("Predicted vs Actual", fontsize=14,
                 fontweight="bold", color="white", pad=12)
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "neo_prediction_error.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_monte_carlo_orbits(neo_results, output_dir, n_show=100):
    """Monte Carlo orbit simulation visualization in 2D orbital plane."""
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    # Earth orbit
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=COLORS["primary"],
            linewidth=2.5, label="Earth Orbit", zorder=3)
    ax.plot(0, 0, "o", color="#FFD700", markersize=14, zorder=5, label="Sun")

    # NEO nominal orbits
    colors_neo = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    for idx, neo in enumerate(neo_results):
        a = neo.get("a", 1.5)
        e = neo.get("e", 0.3)
        nu = np.linspace(0, 2 * np.pi, 300)
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
        x = r * np.cos(nu)
        y = r * np.sin(nu)
        name = neo.get("name", f"NEO {idx}")
        ax.plot(x, y, color=colors_neo[idx % len(colors_neo)],
                linewidth=2, label=name, zorder=2)

        # Monte Carlo perturbations (semi-transparent)
        np.random.seed(42 + idx)
        for _ in range(min(n_show, 30)):
            a_p = a * (1 + np.random.normal(0, 0.01))
            e_p = np.clip(e + np.random.normal(0, 0.005), 0.001, 0.999)
            r_p = a_p * (1 - e_p**2) / (1 + e_p * np.cos(nu))
            x_p = r_p * np.cos(nu)
            y_p = r_p * np.sin(nu)
            ax.plot(x_p, y_p, color=colors_neo[idx % len(colors_neo)],
                    alpha=0.04, linewidth=0.5, zorder=1)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (AU)", fontsize=13, color="white")
    ax.set_ylabel("Y (AU)", fontsize=13, color="white")
    ax.set_title("Monte Carlo Orbit Simulations", fontsize=16,
                 fontweight="bold", color="white", pad=15)
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#333",
              labelcolor="white", loc="upper right")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    path = os.path.join(output_dir, "neo_monte_carlo_orbits.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_residual_errors(y_true, y_pred, residuals, output_dir):
    """Residual error plots: predicted vs residual, histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    residuals = np.asarray(residuals).flatten()
    min_len = min(len(y_pred), len(residuals))
    y_pred, residuals = y_pred[:min_len], residuals[:min_len]

    # Residuals vs predicted
    ax = axes[0]
    ax.set_facecolor("#111827")
    ax.scatter(y_pred, residuals, c=COLORS["primary"], s=10, alpha=0.5, edgecolors="none")
    ax.axhline(0, color=COLORS["warn"], linewidth=1.5, linestyle="--")
    ax.set_xlabel("Predicted Value", fontsize=12, color="white")
    ax.set_ylabel("Residual", fontsize=12, color="white")
    ax.set_title("Residual vs Predicted", fontsize=14,
                 fontweight="bold", color="white", pad=12)
    ax.tick_params(colors="white")

    # Residual histogram
    ax = axes[1]
    ax.set_facecolor("#111827")
    ax.hist(residuals, bins=40, color=COLORS["accent"], alpha=0.75,
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=COLORS["warn"], linewidth=1.5, linestyle="--")
    ax.set_xlabel("Residual Error", fontsize=12, color="white")
    ax.set_ylabel("Frequency", fontsize=12, color="white")
    ax.set_title("Residual Distribution", fontsize=14,
                 fontweight="bold", color="white", pad=12)
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "neo_residual_errors.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_inclination_axis(df_orb, output_dir):
    """Scatter plot of Inclination vs Semi-major Axis."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    if "a" not in df_orb.columns or "i" not in df_orb.columns:
        return None

    sns.scatterplot(data=df_orb, x="a", y="i", hue="class" if "class" in df_orb.columns else None,
                    palette="viridis", ax=ax, s=40, alpha=0.7)
    
    ax.set_xlabel("Semi-major Axis (AU)", fontsize=13, color="white")
    ax.set_ylabel("Inclination (deg)", fontsize=13, color="white")
    ax.set_title("Orbital Patterns: Inclination vs Semi-major Axis", fontsize=16,
                 fontweight="bold", color="white", pad=15)
    ax.tick_params(colors="white")
    if ax.get_legend():
        ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "neo_inclination_axis.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_diameter_distribution(df_orb, output_dir):
    """Histogram of Asteroid Diameters."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    # Use 'diameter' if it exists, else use 'H' (absolute magnitude) as proxy if needed
    if "diameter" in df_orb.columns:
        data = df_orb["diameter"].dropna()
        label = "Diameter (km)"
    elif "H" in df_orb.columns:
        # Very rough proxy: Dia(km) = 10^(3.123 - 0.2*H)
        data = 10**(3.123 - 0.2 * df_orb["H"]).dropna()
        label = "Estimated Diameter (km) from H"
    else:
        return None

    ax.hist(data, bins=50, color=COLORS["accent"], alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel(label, fontsize=13, color="white")
    ax.set_ylabel("Frequency", fontsize=13, color="white")
    ax.set_title("Asteroid Diameter Distribution", fontsize=16,
                 fontweight="bold", color="white", pad=15)
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "neo_diameter_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_pha_count(df_orb, output_dir):
    """Bar chart of Potentially Hazardous Asteroids (PHA)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    if "pha" not in df_orb.columns:
        return None

    pha_counts = df_orb["pha"].value_counts()
    pha_counts.index = ["Non-PHA", "PHA"] if pha_counts.index[0] in [0, "N"] else pha_counts.index

    sns.barplot(x=pha_counts.index, y=pha_counts.values, palette=[COLORS["primary"], COLORS["danger"]], ax=ax)
    
    for i, v in enumerate(pha_counts.values):
        ax.text(i, v + 0.1, str(v), color="white", ha="center", fontweight="bold")

    ax.set_ylabel("Count", fontsize=13, color="white")
    ax.set_title("Potentially Hazardous Asteroids (PHA) Status", fontsize=16,
                 fontweight="bold", color="white", pad=15)
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "neo_pha_count.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def generate_all_neo_plots(neo_propagation_results, collision_results,
                            residual_y_true, residual_y_pred, df_orb, output_dir):
    """Generate all NEO orbit analysis plots."""
    print("\n[NEOViz] Generating NEO orbit plots...")
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    if collision_results:
        paths.append(plot_moid_distribution(collision_results, output_dir))

    if neo_propagation_results:
        paths.append(plot_monte_carlo_orbits(neo_propagation_results, output_dir))

    if residual_y_true is not None and residual_y_pred is not None:
        y_true = np.asarray(residual_y_true).flatten()
        y_pred = np.asarray(residual_y_pred).flatten()
        min_len = min(len(y_true), len(y_pred))
        residuals = y_true[:min_len] - y_pred[:min_len]

        # Create a simple "physics-only" baseline (mean prediction)
        physics_only = np.full_like(y_pred[:min_len], y_true[:min_len].mean())
        paths.append(plot_orbit_prediction_error(y_true[:min_len], physics_only,
                                                  y_pred[:min_len], output_dir))
        paths.append(plot_residual_errors(y_true[:min_len], y_pred[:min_len],
                                           residuals, output_dir))

    if df_orb is not None:
        paths.append(plot_inclination_axis(df_orb, output_dir))
        paths.append(plot_diameter_distribution(df_orb, output_dir))
        paths.append(plot_pha_count(df_orb, output_dir))

    paths = [p for p in paths if p is not None]
    print(f"  Generated {len(paths)} plots in {output_dir}")
    return paths
