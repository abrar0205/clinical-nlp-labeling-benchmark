"""Evaluation metrics computed with scikit-learn.

Labels are multi-class (positive / negative / uncertain / not_mentioned), so we
report per-pathology macro precision / recall / F1 (averaged over the four label
classes) plus accuracy. We then aggregate a macro-F1 across the four
pathologies, and report an exact-match rate per report.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils import LABELS, PATHOLOGIES


def compute_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy results table with one row per (method, pathology).

    A ``macro_avg`` row per method aggregates macro-F1 across pathologies and
    reports the overall accuracy across all pathology predictions.
    """
    rows = []
    for method in predictions["method"].unique():
        method_df = predictions[predictions["method"] == method]
        per_pathology_f1 = []

        for pathology in PATHOLOGIES:
            sub = method_df[method_df["pathology"] == pathology]
            y_true = sub["gold_label"].tolist()
            y_pred = sub["predicted_label"].tolist()
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=LABELS, average="macro", zero_division=0
            )
            accuracy = accuracy_score(y_true, y_pred)
            rows.append(
                {
                    "method": method,
                    "pathology": pathology,
                    "accuracy": round(accuracy, 4),
                    "macro_precision": round(precision, 4),
                    "macro_recall": round(recall, 4),
                    "macro_f1": round(f1, 4),
                    "support": len(sub),
                }
            )
            per_pathology_f1.append(f1)

        macro_f1 = sum(per_pathology_f1) / len(per_pathology_f1)
        overall_accuracy = accuracy_score(
            method_df["gold_label"], method_df["predicted_label"]
        )
        rows.append(
            {
                "method": method,
                "pathology": "macro_avg",
                "accuracy": round(overall_accuracy, 4),
                "macro_precision": "",
                "macro_recall": "",
                "macro_f1": round(macro_f1, 4),
                "support": len(method_df),
            }
        )

    return pd.DataFrame(rows)


def compute_exact_match(predictions: pd.DataFrame) -> Dict[str, float]:
    """Fraction of reports for which every pathology label is correct."""
    rates: Dict[str, float] = {}
    for method in predictions["method"].unique():
        method_df = predictions[predictions["method"] == method]
        per_report_all_correct = method_df.groupby("report_id")["correct"].all()
        rates[method] = round(float(per_report_all_correct.mean()), 4)
    return rates
