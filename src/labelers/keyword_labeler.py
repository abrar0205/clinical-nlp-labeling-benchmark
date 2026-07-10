"""Keyword baseline labeler.

Simple presence/absence matching:
  * If any keyword for a pathology appears in the report -> ``positive``.
  * Otherwise -> ``not_mentioned``.

This method has no notion of negation or uncertainty and is therefore expected
to fail on negated ("no pneumothorax") and uncertain ("possible effusion")
statements. That failure is intentional: it establishes a naive baseline that
the negation-aware and LLM labelers can be compared against.
"""
from __future__ import annotations

from typing import Dict, List

from preprocessing.clean_reports import clean_text
from utils import PATHOLOGIES

# Keyword dictionary per pathology. Kept small and transparent on purpose.
KEYWORDS: Dict[str, List[str]] = {
    "pneumonia": ["pneumonia", "consolidation", "infiltrate", "airspace opacity"],
    "pleural_effusion": ["pleural effusion", "effusion"],
    "pneumothorax": ["pneumothorax"],
    "cardiomegaly": [
        "cardiomegaly",
        "enlarged cardiac silhouette",
        "enlarged heart",
        "cardiomediastinal enlargement",
    ],
}


class KeywordLabeler:
    """Label a report using pure keyword presence."""

    name = "keyword"

    def __init__(self) -> None:
        self.keywords = KEYWORDS

    def label_report(self, report_text: str) -> Dict[str, str]:
        text = clean_text(report_text)
        result: Dict[str, str] = {}
        for pathology in PATHOLOGIES:
            found = any(kw in text for kw in self.keywords[pathology])
            result[pathology] = "positive" if found else "not_mentioned"
        return result
