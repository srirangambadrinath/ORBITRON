"""
ORBITRON — Mission Intelligence Visualizations
Generates: risk contribution pie chart, mission risk radar chart,
risk sensitivity analysis, risk timeline chart.
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


def plot_risk_contribution_pie(launch_risk, satellite_risk, neo_risk, output_dir):
    """Risk contribution pie chart."""
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(COLORS["bg"])

    weighted = [0.4 * launch_risk, 0.35 * satellite_risk, 0.25 * neo_risk]
    total = sum(weighted) or 1
    contributions = [w / total * 100 for w in weighted]

    wedge_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(
        contributions, explode=explode, labels=None, autopct="%1.1f%%",
        colors=wedge_colors, startangle=140, pctdistance=0.75,
        wedgeprops=dict(linewidth=2, edgecolor="#0d1117")
    )
    for t in autotexts:
        t.set_fontsize(14)
        t.set_fontweight("bold")
        t.set_color("white")

    ax.legend(
        [f"Launch Risk ({launch_risk:.3f})",
         f"Satellite Risk ({satellite_risk:.3f})",
         f"NEO Risk ({neo_risk:.3f})"],
        loc="lower center", fontsize=12, facecolor="#1a1a2e",
        edgecolor="#333", labelcolor="white",
        bbox_to_anchor=(0.5, -0.05), ncol=1
    )

    centre_circle = plt.Circle((0, 0), 0.50, fc=COLORS["bg"])
    ax.add_artist(centre_circle)
    mission_risk = 0.4 * launch_risk + 0.35 * satellite_risk + 0.25 * neo_risk
    ax.text(0, 0, f"MRI\n{mission_risk:.3f}", ha="center", va="center",
            fontsize=20, fontweight="bold", color="white")
    ax.set_title("Mission Risk Contribution Breakdown", fontsize=16,
                 fontweight="bold", color="white", pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, "mission_risk_pie.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_risk_radar(launch_risk, satellite_risk, neo_risk, output_dir):
    """Mission risk radar chart showing all risk dimensions."""
    categories = ["Launch\nFailure", "Satellite\nAnomaly", "NEO\nCollision",
                   "System\nReadiness", "Mission\nConfidence"]
    values = [
        launch_risk,
        satellite_risk,
        neo_risk,
        1.0 - max(launch_risk, satellite_risk, neo_risk),  # System readiness
        1.0 - (0.4 * launch_risk + 0.35 * satellite_risk + 0.25 * neo_risk)  # Confidence
    ]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    ax.plot(angles, values, color=COLORS["primary"], linewidth=2.5)
    ax.fill(angles, values, color=COLORS["primary"], alpha=0.2)

    # Threshold ring
    threshold_vals = [0.5] * (N + 1)
    ax.plot(angles, threshold_vals, "--", color=COLORS["danger"], linewidth=1.5,
            alpha=0.6, label="Alert Threshold")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, color="white", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                        fontsize=9, color="#888")
    ax.spines["polar"].set_color("#333")
    ax.grid(color="#333", alpha=0.4)

    ax.set_title("Mission Risk Radar", fontsize=16,
                 fontweight="bold", color="white", pad=30)

    plt.tight_layout()
    path = os.path.join(output_dir, "mission_risk_radar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_risk_sensitivity(launch_risk, satellite_risk, neo_risk, output_dir):
    """Risk sensitivity analysis — how each component affects MRI."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    sweep = np.linspace(0, 1, 100)
    base_mri = 0.4 * launch_risk + 0.35 * satellite_risk + 0.25 * neo_risk

    configs = [
        ("Launch Risk", 0.4, satellite_risk, neo_risk, COLORS["danger"]),
        ("Satellite Risk", 0.35, launch_risk, neo_risk, COLORS["accent"]),
        ("NEO Risk", 0.25, launch_risk, satellite_risk, COLORS["primary"]),
    ]

    for idx, (label, weight, other1, other2, color) in enumerate(configs):
        ax = axes[idx]
        ax.set_facecolor("#111827")

        if idx == 0:
            mri_sweep = weight * sweep + 0.35 * other1 + 0.25 * other2
        elif idx == 1:
            mri_sweep = 0.4 * other1 + weight * sweep + 0.25 * other2
        else:
            mri_sweep = 0.4 * other1 + 0.35 * other2 + weight * sweep

        ax.plot(sweep, mri_sweep, color=color, linewidth=2.5)
        ax.fill_between(sweep, mri_sweep, alpha=0.1, color=color)
        ax.axhline(base_mri, color=COLORS["warn"], linestyle="--", linewidth=1.5,
                   alpha=0.7, label=f"Current MRI = {base_mri:.3f}")

        current_val = [launch_risk, satellite_risk, neo_risk][idx]
        ax.axvline(current_val, color="white", linestyle=":", linewidth=1.2,
                   alpha=0.5, label=f"Current = {current_val:.3f}")

        ax.set_xlabel(label, fontsize=12, color="white")
        ax.set_ylabel("Mission Risk Index", fontsize=12, color="white")
        ax.set_title(f"Sensitivity: {label}", fontsize=13,
                     fontweight="bold", color="white", pad=10)
        ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
        ax.tick_params(colors="white")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "mission_risk_sensitivity.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_risk_timeline(launch_risk, satellite_risk, neo_risk, output_dir):
    """Simulated risk timeline showing how risk evolves over mission phases."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    phases = ["Pre-Launch\nCheck", "Countdown", "Ignition", "Ascent",
              "Stage Sep.", "Orbit\nInsertion", "Cruise", "Mid-Course\nCorrection",
              "Final\nApproach", "Mission\nComplete"]
    n = len(phases)

    # Simulate risk evolution across mission phases
    np.random.seed(42)
    lr = np.clip(np.array([0.7, 0.5, launch_risk * 2, launch_risk * 1.8,
                           launch_risk * 1.2, launch_risk * 0.8, launch_risk * 0.4,
                           launch_risk * 0.3, launch_risk * 0.2, launch_risk * 0.1]), 0, 1)
    sr = np.clip(np.array([0.05, 0.05, 0.1, 0.15, satellite_risk * 0.5,
                           satellite_risk * 0.7, satellite_risk, satellite_risk * 1.1,
                           satellite_risk * 0.9, satellite_risk * 0.6]), 0, 1)
    nr = np.clip(np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.02,
                           neo_risk * 0.5, neo_risk * 0.8, neo_risk, neo_risk * 0.7]), 0, 1)

    mri = 0.4 * lr + 0.35 * sr + 0.25 * nr
    x = np.arange(n)

    ax.fill_between(x, 0, lr, alpha=0.2, color="#FF6B6B")
    ax.fill_between(x, 0, sr, alpha=0.2, color="#4ECDC4")
    ax.fill_between(x, 0, nr, alpha=0.2, color="#45B7D1")

    ax.plot(x, lr, "o-", color="#FF6B6B", linewidth=2, markersize=7, label="Launch Risk")
    ax.plot(x, sr, "s-", color="#4ECDC4", linewidth=2, markersize=7, label="Satellite Risk")
    ax.plot(x, nr, "^-", color="#45B7D1", linewidth=2, markersize=7, label="NEO Risk")
    ax.plot(x, mri, "D-", color=COLORS["warn"], linewidth=3, markersize=8,
            label="Mission Risk Index", zorder=5)

    ax.axhline(0.5, color=COLORS["danger"], linestyle="--", linewidth=1.5,
               alpha=0.5, label="Alert Level")

    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=10, color="white")
    ax.set_ylabel("Risk Score", fontsize=13, color="white")
    ax.set_title("Mission Risk Timeline Across Phases", fontsize=16,
                 fontweight="bold", color="white", pad=15)
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#333",
              labelcolor="white", loc="upper right", ncol=2)
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "mission_risk_timeline.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def generate_all_mission_plots(launch_risk, satellite_risk, neo_risk, output_dir):
    """Generate all mission intelligence visualizations."""
    print("\n[MissionViz] Generating mission intelligence plots...")
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    paths.append(plot_risk_contribution_pie(launch_risk, satellite_risk, neo_risk, output_dir))
    paths.append(plot_risk_radar(launch_risk, satellite_risk, neo_risk, output_dir))
    paths.append(plot_risk_sensitivity(launch_risk, satellite_risk, neo_risk, output_dir))
    paths.append(plot_risk_timeline(launch_risk, satellite_risk, neo_risk, output_dir))

    print(f"  Generated {len(paths)} plots in {output_dir}")
    return paths
