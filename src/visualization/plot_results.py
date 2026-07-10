"""Plot generation with matplotlib / seaborn.

Produces four figures:
  * macro-F1 by method
  * per-pathology macro-F1 by method
  * error count by method
  * confusion matrix for one selected pathology (default: pneumonia)

Plots are written to ``experiments/plots`` and copied to
``docs/assets/plots`` for the GitHub Pages dashboard.
"""
from __future__ import annotations

import os
import shutil

import matplotlib

matplotlib.use("Agg")  # headless backend, safe on any laptop / CI

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from utils import LABELS, ensure_dir  # noqa: E402

sns.set_theme(style="whitegrid")


def plot_macro_f1(results: pd.DataFrame, out_path: str) -> None:
    df = results[results["pathology"] == "macro_avg"].copy()
    df["macro_f1"] = df["macro_f1"].astype(float)
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x="method", y="macro_f1", hue="method",
                     palette="viridis", legend=False)
    ax.set_title("Macro-F1 by method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_per_pathology_f1(results: pd.DataFrame, out_path: str) -> None:
    df = results[results["pathology"] != "macro_avg"].copy()
    df["macro_f1"] = df["macro_f1"].astype(float)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x="pathology", y="macro_f1", hue="method",
                     palette="viridis")
    ax.set_title("Per-pathology macro-F1 by method")
    ax.set_xlabel("Pathology")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20)
    plt.legend(title="Method", loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_error_counts(error_df: pd.DataFrame, out_path: str) -> None:
    if error_df.empty:
        counts = pd.DataFrame({"method": [], "count": []})
    else:
        counts = error_df.groupby("method").size().reset_index(name="count")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=counts, x="method", y="count", hue="method",
                     palette="rocket", legend=False)
    ax.set_title("Error count by method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Number of incorrect predictions")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_confusion_matrix(
    predictions: pd.DataFrame, pathology: str, out_path: str
) -> None:
    methods = list(predictions["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.2))
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        sub = predictions[
            (predictions["method"] == method)
            & (predictions["pathology"] == pathology)
        ]
        matrix = confusion_matrix(
            sub["gold_label"], sub["predicted_label"], labels=LABELS
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=LABELS,
            yticklabels=LABELS,
            ax=ax,
        )
        ax.set_title(f"{method}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle(f"Confusion matrix: {pathology}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def generate_all_plots(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    error_df: pd.DataFrame,
    plots_dir: str,
    confusion_pathology: str = "pneumonia",
) -> None:
    ensure_dir(plots_dir)
    plot_macro_f1(results, os.path.join(plots_dir, "macro_f1_by_method.png"))
    plot_per_pathology_f1(
        results, os.path.join(plots_dir, "per_pathology_f1.png")
    )
    plot_error_counts(
        error_df, os.path.join(plots_dir, "error_count_by_method.png")
    )
    plot_confusion_matrix(
        predictions,
        confusion_pathology,
        os.path.join(plots_dir, f"confusion_matrix_{confusion_pathology}.png"),
    )


def copy_plots(src_dir: str, dst_dir: str) -> None:
    """Copy generated PNG plots to the docs assets folder."""
    ensure_dir(dst_dir)
    for name in os.listdir(src_dir):
        if name.endswith(".png"):
            shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name))
