"""Entry point for the radiology report labeling mini-benchmark.

Usage
-----
    python src/main.py --config configs/demo_mock.yaml
    python src/main.py --config configs/demo_hf_local.yaml

The pipeline loads the config and data, runs every labeler, evaluates the
predictions, performs error analysis, generates plots, writes run metadata, and
prints a clean terminal summary. All outputs are written under ``experiments/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
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
from visualization.plot_results import generate_all_plots  # noqa: E402

PRED_COLUMNS = [
    "report_id",
    "method",
    "pathology",
    "gold_label",
    "predicted_label",
    "correct",
    "report_text",
]


def build_labelers(config: Dict) -> tuple:
    """Instantiate the labelers requested by the config.

    Returns ``(labelers, llm)`` where ``llm`` is the resolved LLMLabeler (or
    ``None`` if no LLM ran). The LLM is only included after resolving whether the
    requested backend is actually available, so method names are honest.
    """
    labelers: List = [KeywordLabeler(), NegationRuleLabeler()]
    llm: LLMLabeler | None = None
    llm_cfg = config.get("llm", {}) or {}

    if llm_cfg.get("enabled", True):
        candidate = LLMLabeler(
            mode=llm_cfg.get("mode", "mock"),
            model=llm_cfg.get("model", "Qwen/Qwen2.5-0.5B-Instruct"),
            fallback_to_mock=llm_cfg.get("fallback_to_mock", True),
            max_new_tokens=llm_cfg.get("max_new_tokens", 160),
            cache_dir=llm_cfg.get("cache_dir"),
        )
        name, active = candidate.resolve()
        if active:
            if candidate.mode == "huggingface" and candidate._use_real:
                print(f"LLM: Hugging Face model loaded locally -> method '{name}'")
            elif candidate.mode == "huggingface":
                print(
                    "LLM: Hugging Face model could not be loaded; using mock "
                    f"fallback -> method '{name}'"
                )
            else:
                print(f"LLM: mock mode -> method '{name}'")
            labelers.append(candidate)
            llm = candidate
        else:
            print(
                "LLM: Hugging Face local inference was requested, but the model "
                "could not be loaded and fallback_to_mock is false.\n"
                f"     Model: {candidate.model}\n"
                f"     Error: {candidate._load_error}\n"
                "     Try: python src/check_huggingface.py --model "
                f"{candidate.model}\n"
                "     Or use configs/demo_mock.yaml for the quick reproducible demo."
            )
            sys.exit(1)
    return labelers, llm


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
            f"  {method:<18} "
            f"macro-F1={float(row['macro_f1']):.3f}  "
            f"overall-acc={float(row['accuracy']):.3f}  "
            f"exact-match={exact.get(method, 0):.3f}"
        )

    print("\nError counts by method:")
    if error_df.empty:
        print("  (no errors)")
    else:
        for method, count in error_df.groupby("method").size().items():
            print(f"  {method:<18} {count} incorrect predictions")
        print("\nError types (all methods):")
        for etype, count in (
            error_df.groupby("error_type").size().sort_values(ascending=False).items()
        ):
            print(f"  {etype:<34} {count}")


def write_run_metadata(
    path: str,
    args_config: str,
    data_path: str,
    num_reports: int,
    methods: List[str],
    llm: "LLMLabeler | None",
    output_files: List[str],
) -> Dict:
    """Write experiments/run_metadata.json describing exactly what ran."""
    llm_cfg_present = llm is not None
    real_hf_used = bool(llm_cfg_present and llm._use_real and llm.real_calls > 0)
    fallback_mock_used = bool(
        llm_cfg_present and (llm.name == "llm_hf_fallback_mock" or llm.fallback_calls > 0)
    )
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": args_config,
        "data_path": data_path,
        "num_reports": num_reports,
        "methods_run": methods,
        "llm_mode": llm.mode if llm_cfg_present else None,
        "llm_model": llm.model if llm_cfg_present else None,
        "fallback_to_mock": llm.fallback_to_mock if llm_cfg_present else None,
        "cache_dir": llm.cache_dir if llm_cfg_present else None,
        "real_huggingface_used": real_hf_used,
        "fallback_mock_used": fallback_mock_used,
        "real_hf_calls": llm.real_calls if llm_cfg_present else 0,
        "fallback_calls": llm.fallback_calls if llm_cfg_present else 0,
        "output_files": output_files,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


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

    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    if not os.path.exists(data_path):
        print(f"ERROR: data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} reports from {data_path}")

    labelers, llm = build_labelers(config)
    print("Methods:", ", ".join(labeler.name for labeler in labelers))

    predictions = run_predictions(labelers, df)
    predictions_path = os.path.join(output_dir, "predictions.csv")
    results_path = os.path.join(output_dir, "results.csv")
    error_path = os.path.join(output_dir, "error_analysis.csv")
    metadata_path = os.path.join(output_dir, "run_metadata.json")

    predictions[PRED_COLUMNS].to_csv(predictions_path, index=False)

    results = compute_results(predictions)
    results.to_csv(results_path, index=False)

    error_df = build_error_analysis(predictions)
    error_df.to_csv(error_path, index=False)

    confusion_pathology = (config.get("evaluation", {}) or {}).get(
        "confusion_pathology", "pneumonia"
    )
    generate_all_plots(results, predictions, error_df, plots_dir, confusion_pathology)

    output_files = [
        predictions_path,
        results_path,
        error_path,
        metadata_path,
        plots_dir,
    ]
    metadata = write_run_metadata(
        metadata_path,
        args.config,
        data_path,
        len(df),
        [labeler.name for labeler in labelers],
        llm,
        output_files,
    )

    print_summary(results, error_df, predictions)

    print("\nLLM run type:")
    if llm is None:
        print("  no LLM labeler ran")
    else:
        print(f"  method name             : {llm.name}")
        print(f"  real Hugging Face used  : {metadata['real_huggingface_used']}")
        print(f"  fallback mock used      : {metadata['fallback_mock_used']}")

    print("\nOutputs written to:")
    print(f"  {predictions_path}")
    print(f"  {results_path}")
    print(f"  {error_path}")
    print(f"  {metadata_path}")
    print(f"  {plots_dir}/ (plots)")
    print("\nTip: check experiments/run_metadata.json to confirm the run type.")


if __name__ == "__main__":
    main()
