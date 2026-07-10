"""Rule-based error typing for incorrect predictions.

Each incorrect (method, pathology, report) prediction is assigned a coarse
error type so that failure modes can be counted and inspected. The logic is
deliberately simple and ordered; the first matching rule wins.
"""
from __future__ import annotations

import pandas as pd


def classify_error(gold: str, pred: str, parse_error: bool = False) -> str:
    """Return a coarse error type for one incorrect prediction."""
    if parse_error:
        return "llm_parse_error"
    if pred == "positive" and gold == "negative":
        return "false_positive_negation_error"
    if gold == "uncertain" and pred != "uncertain":
        return "uncertainty_misclassified"
    if gold == "positive" and pred in ("not_mentioned", "negative"):
        return "false_negative_missed_concept"
    if gold == "not_mentioned" and pred != "not_mentioned":
        return "not_mentioned_confusion"
    return "other"


def build_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build the error-analysis table for all incorrect predictions."""
    rows = []
    for _, row in predictions.iterrows():
        if bool(row["correct"]):
            continue
        parse_error = bool(row.get("parse_error", False))
        error_type = classify_error(
            row["gold_label"], row["predicted_label"], parse_error
        )
        rows.append(
            {
                "report_id": row["report_id"],
                "method": row["method"],
                "pathology": row["pathology"],
                "report_text": row["report_text"],
                "gold_label": row["gold_label"],
                "predicted_label": row["predicted_label"],
                "error_type": error_type,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "report_id",
            "method",
            "pathology",
            "report_text",
            "gold_label",
            "predicted_label",
            "error_type",
        ],
    )
