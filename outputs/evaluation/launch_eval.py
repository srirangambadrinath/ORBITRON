"""
ORBITRON — Launch Failure Evaluation Visualizations
Generates: confusion matrix, ROC curve, precision-recall curve,
prediction probability distribution, learning curve, feature importance.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.calibration import calibration_curve

# Seaborn dark research style
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
COLORS = {"primary": "#4FC3F7", "secondary": "#FF8A65", "accent": "#81C784",
           "warn": "#FFD54F", "danger": "#EF5350", "bg": "#0d1117"}


def full_evaluation(y_true, y_pred, y_prob, output_dir):
    """Compute all classification metrics and return as dict."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = metrics["accuracy"]

    print("\n[LaunchEval] Classification Metrics:")
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v:.4f}")
    return metrics


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"],
                linewidths=2, linecolor="#1a1a2e", ax=ax,
                annot_kws={"size": 18, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=13, color="white")
    ax.set_ylabel("Actual", fontsize=13, color="white")
    ax.set_title("Launch Failure — Confusion Matrix", fontsize=15,
                 fontweight="bold", color="white", pad=15)
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "launch_confusion_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_roc_curve(y_true, y_prob, output_dir):
    """ROC curve with AUC."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=COLORS["primary"], linewidth=2.5,
                label=f"Ensemble (AUC = {auc_val:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])
    except Exception:
        auc_val = 0.5
        ax.text(0.5, 0.5, "Insufficient data\nfor ROC curve",
                ha="center", va="center", fontsize=14, color="white",
                transform=ax.transAxes)

    ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=1.2, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=13, color="white")
    ax.set_ylabel("True Positive Rate", fontsize=13, color="white")
    ax.set_title("Launch Failure — ROC Curve", fontsize=15,
                 fontweight="bold", color="white", pad=15)
    ax.legend(fontsize=11, loc="lower right", facecolor="#1a1a2e",
              edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="white")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    path = os.path.join(output_dir, "launch_roc_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_precision_recall_curve(y_true, y_prob, output_dir):
    """Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, color=COLORS["accent"], linewidth=2.5,
                label=f"Ensemble (AP = {ap:.3f})")
        ax.fill_between(rec, prec, alpha=0.15, color=COLORS["accent"])
    except Exception:
        ax.text(0.5, 0.5, "Insufficient data\nfor PR curve",
                ha="center", va="center", fontsize=14, color="white",
                transform=ax.transAxes)

    ax.set_xlabel("Recall", fontsize=13, color="white")
    ax.set_ylabel("Precision", fontsize=13, color="white")
    ax.set_title("Launch Failure — Precision-Recall Curve", fontsize=15,
                 fontweight="bold", color="white", pad=15)
    ax.legend(fontsize=11, loc="upper right", facecolor="#1a1a2e",
              edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "launch_precision_recall.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_probability_distribution(y_true, y_prob, output_dir):
    """Prediction probability distribution by class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    mask_0 = y_true == 0
    mask_1 = y_true == 1

    if mask_0.sum() > 0:
        ax.hist(y_prob[mask_0], bins=20, alpha=0.7, color=COLORS["primary"],
                label="No Failure", edgecolor="white", linewidth=0.5)
    if mask_1.sum() > 0:
        ax.hist(y_prob[mask_1], bins=20, alpha=0.7, color=COLORS["danger"],
                label="Failure", edgecolor="white", linewidth=0.5)

    ax.axvline(x=0.5, color=COLORS["warn"], linestyle="--", linewidth=2,
               label="Decision Threshold")
    ax.set_xlabel("Predicted Probability", fontsize=13, color="white")
    ax.set_ylabel("Count", fontsize=13, color="white")
    ax.set_title("Launch Failure — Prediction Probability Distribution",
                 fontsize=15, fontweight="bold", color="white", pad=15)
    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "launch_prob_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_learning_curve(model, X, y, output_dir):
    """Learning curve showing train/val scores vs training size."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    n_samples = len(y)
    n_classes_min = min(np.bincount(y.astype(int)))
    n_folds = min(3, n_classes_min, n_samples)
    n_folds = max(n_folds, 2)

    try:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        train_sizes = np.linspace(0.3, 1.0, 5)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes,
            scoring="accuracy", n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax.fill_between(train_sizes_abs, train_mean - train_std,
                         train_mean + train_std, alpha=0.15, color=COLORS["primary"])
        ax.fill_between(train_sizes_abs, val_mean - val_std,
                         val_mean + val_std, alpha=0.15, color=COLORS["secondary"])
        ax.plot(train_sizes_abs, train_mean, "o-", color=COLORS["primary"],
                linewidth=2, markersize=6, label="Training Score")
        ax.plot(train_sizes_abs, val_mean, "s-", color=COLORS["secondary"],
                linewidth=2, markersize=6, label="Validation Score")
    except Exception as e:
        ax.text(0.5, 0.5, f"Learning curve unavailable\n({e})",
                ha="center", va="center", fontsize=12, color="white",
                transform=ax.transAxes)

    ax.set_xlabel("Training Set Size", fontsize=13, color="white")
    ax.set_ylabel("Accuracy", fontsize=13, color="white")
    ax.set_title("Launch Failure — Learning Curve", fontsize=15,
                 fontweight="bold", color="white", pad=15)
    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "launch_learning_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_dataset_distribution(y_train, y_test, output_dir):
    """Bar chart showing class distribution in train and test sets."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    train_counts = np.bincount(y_train.astype(int), minlength=2)
    test_counts = np.bincount(y_test.astype(int), minlength=2)

    x = np.arange(2)
    width = 0.35

    ax.bar(x - width/2, train_counts, width, label='Train', color=COLORS["primary"], edgecolor="white")
    ax.bar(x + width/2, test_counts, width, label='Test', color=COLORS["accent"], edgecolor="white")

    ax.set_ylabel('Number of Samples', fontsize=13, color="white")
    ax.set_title('Launch Dataset Class Distribution', fontsize=15, fontweight="bold", color="white", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['No Failure (0)', 'Failure (1)'], fontsize=12, color="white")
    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="white")

    for i in range(2):
        ax.text(i - width/2, train_counts[i] + 0.5, str(train_counts[i]), ha='center', va='bottom', color="white", fontweight="bold")
        ax.text(i + width/2, test_counts[i] + 0.5, str(test_counts[i]), ha='center', va='bottom', color="white", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "launch_class_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_feature_correlation(X_train, feature_names, output_dir):
    """Correlation heatmap of features."""
    import pandas as pd
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    if isinstance(X_train, np.ndarray):
        df = pd.DataFrame(X_train, columns=feature_names[:X_train.shape[1]])
    else:
        df = X_train

    corr = df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax,
                annot=False) # Annot off for large feature sets to prevent clutter

    ax.set_title("Launch Failure — Feature Correlation Heatmap", fontsize=15,
                 fontweight="bold", color="white", pad=15)
    plt.setp(ax.get_xticklabels(), color="white", rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), color="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "launch_feature_correlation.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_calibration_curve(y_true, y_prob, output_dir):
    """Calibration curve (reliability diagram)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    ax.plot(prob_pred, prob_true, "s-", color=COLORS["primary"], linewidth=2, label="Ensemble")
    ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=1.2, label="Perfectly Calibrated")

    ax.set_xlabel("Mean Predicted Probability", fontsize=13, color="white")
    ax.set_ylabel("Fraction of Positives", fontsize=13, color="white")
    ax.set_title("Launch Failure — Calibration Curve", fontsize=15,
                 fontweight="bold", color="white", pad=15)
    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "launch_calibration_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names, output_dir):
    """Feature importance bar chart from XGBoost."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor("#111827")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        n = len(importances)
        names = feature_names[:n] if len(feature_names) >= n else \
            feature_names + [f"f{i}" for i in range(len(feature_names), n)]
        indices = np.argsort(importances)

        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(indices)))
        ax.barh(range(len(indices)), importances[indices], color=colors,
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([names[i] for i in indices], fontsize=11, color="white")
    else:
        ax.text(0.5, 0.5, "Feature importances not available",
                ha="center", va="center", fontsize=14, color="white",
                transform=ax.transAxes)

    ax.set_xlabel("Importance (Gain)", fontsize=13, color="white")
    ax.set_title("Launch Failure — XGBoost Feature Importance", fontsize=15,
                 fontweight="bold", color="white", pad=15)
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "launch_feature_importance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return path


def generate_all_launch_plots(xgb_model, X_train, y_train, X_test, y_test,
                               xgb_probs, lstm_probs, feature_names, output_dir):
    """Generate all launch failure evaluation plots."""
    print("\n[LaunchEval] Generating evaluation plots...")
    os.makedirs(output_dir, exist_ok=True)

    # Ensemble predictions
    ensemble_probs = 0.6 * np.asarray(xgb_probs).flatten() + \
                     0.4 * np.asarray(lstm_probs).flatten()
    min_len = min(len(ensemble_probs), len(y_test))
    ensemble_probs = ensemble_probs[:min_len]
    y_test_use = np.asarray(y_test).flatten()[:min_len]
    y_pred = (ensemble_probs > 0.5).astype(int)

    metrics = full_evaluation(y_test_use, y_pred, ensemble_probs, output_dir)

    paths = []
    paths.append(plot_confusion_matrix(y_test_use, y_pred, output_dir))
    paths.append(plot_roc_curve(y_test_use, ensemble_probs, output_dir))
    paths.append(plot_precision_recall_curve(y_test_use, ensemble_probs, output_dir))
    paths.append(plot_probability_distribution(y_test_use, ensemble_probs, output_dir))
    paths.append(plot_learning_curve(xgb_model, X_train, y_train, output_dir))
    paths.append(plot_feature_importance(xgb_model, feature_names, output_dir))
    
    # New plots requested by user
    paths.append(plot_dataset_distribution(y_train, y_test_use, output_dir))
    paths.append(plot_feature_correlation(X_train, feature_names, output_dir))
    paths.append(plot_calibration_curve(y_test_use, ensemble_probs, output_dir))

    print(f"  Generated {len(paths)} plots in {output_dir}")
    return metrics, paths
