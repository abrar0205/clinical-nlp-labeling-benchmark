"""Entry point for the radiology report labeling mini-benchmark.

Usage
-----
    python src/main.py --config configs/demo_mock.yaml
    python src/main.py --config configs/demo_ollama_gemma.yaml

The pipeline loads the config and data, runs every labeler, evaluates the
predictions, performs error analysis, generates plots, copies the plots to the
docs assets folder, and prints a clean terminal summary.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import pandas as pd

# Make the local ``src`` package importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.error_analysis import build_error_analysis  # noqa: E402
from evaluation.metrics import compute_exact_match, compute_results  # noqa: E402
from labelers.keyword_labeler import KeywordLabeler  # noqa: E402
from labelers.llm_labeler import LLMLabeler  # noqa: E402
from labelers.negation_rule_labeler import NegationRuleLabeler  # noqa: E402
from utils import PATHOLOGIES, ensure_dir, load_config  # noqa: E402
from visualization.plot_results import (  # noqa: E402
    copy_plots,
    generate_all_plots,
)

PRED_COLUMNS = [
    "report_id",
    "method",
    "pathology",
    "gold_label",
    "predicted_label",
    "correct",
    "report_text",
]


def build_labelers(config: Dict) -> List:
    """Instantiate the labelers requested by the config."""
    labelers: List = [KeywordLabeler(), NegationRuleLabeler()]
    llm_cfg = config.get("llm", {}) or {}
    if llm_cfg.get("enabled", True):
        labelers.append(
            LLMLabeler(
                mode=llm_cfg.get("mode", "mock"),
                model=llm_cfg.get("model", "gemma3:1b"),
                endpoint=llm_cfg.get(
                    "endpoint", "http://localhost:11434/api/generate"
                ),
                fallback_to_mock=llm_cfg.get("fallback_to_mock", True),
            )
        )
    return labelers


def run_predictions(labelers: List, df: pd.DataFrame) -> pd.DataFrame:
    """Run every labeler over every report and return a tidy predictions table."""
    rows = []
    for _, report in df.iterrows():
        report_id = report["report_id"]
        report_text = report["report_text"]
        for labeler in labelers:
            preds = labeler.label_report(report_text)
            parse_error = bool(preds.get("parse_error", False))
            for pathology in PATHOLOGIES:
                gold = report[f"{pathology}_gold"]
                pred = preds[pathology]
                rows.append(
                    {
                        "report_id": report_id,
                        "method": labeler.name,
                        "pathology": pathology,
                        "gold_label": gold,
                        "predicted_label": pred,
                        "correct": gold == pred,
                        "report_text": report_text,
                        "parse_error": parse_error,
                    }
                )
    return pd.DataFrame(rows)


def print_summary(
    results: pd.DataFrame, error_df: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    exact = compute_exact_match(predictions)
    macro = results[results["pathology"] == "macro_avg"]
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print("\nMacro-F1 by method:")
    for _, row in macro.iterrows():
        method = row["method"]
        print(
            f"  {method:<14} "
            f"macro-F1={float(row['macro_f1']):.3f}  "
            f"overall-acc={float(row['accuracy']):.3f}  "
            f"exact-match={exact.get(method, 0):.3f}"
        )

    print("\nError counts by method:")
    if error_df.empty:
        print("  (no errors)")
    else:
        for method, count in error_df.groupby("method").size().items():
            print(f"  {method:<14} {count} incorrect predictions")
        print("\nError types (all methods):")
        for etype, count in (
            error_df.groupby("error_type").size().sort_values(ascending=False).items()
        ):
            print(f"  {etype:<34} {count}")

    print("\nDone. Outputs written to experiments/ and docs/assets/plots/.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Radiology report labeling mini-benchmark"
    )
    parser.add_argument(
        "--config", required=True, help="Path to a YAML config file."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = config["data_path"]
    output_dir = config.get("output_dir", "experiments")
    plots_dir = os.path.join(output_dir, "plots")
    docs_plots_dir = os.path.join("docs", "assets", "plots")

    ensure_dir(output_dir)
    ensure_dir(plots_dir)
    ensure_dir(docs_plots_dir)

    if not os.path.exists(data_path):
        print(f"ERROR: data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} reports from {data_path}")

    labelers = build_labelers(config)
    print("Methods:", ", ".join(labeler.name for labeler in labelers))

    predictions = run_predictions(labelers, df)
    predictions[PRED_COLUMNS].to_csv(
        os.path.join(output_dir, "predictions.csv"), index=False
    )

    results = compute_results(predictions)
    results.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    error_df = build_error_analysis(predictions)
    error_df.to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)

    confusion_pathology = (config.get("evaluation", {}) or {}).get(
        "confusion_pathology", "pneumonia"
    )
    generate_all_plots(results, predictions, error_df, plots_dir, confusion_pathology)
    copy_plots(plots_dir, docs_plots_dir)

    print_summary(results, error_df, predictions)


if __name__ == "__main__":
    main()
