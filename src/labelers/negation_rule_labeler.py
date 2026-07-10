"""Negation- and uncertainty-aware rule-based labeler.

Transparent, window-based logic (NegEx-inspired but deliberately minimal):

  1. Split the report into short clauses on sentence terminators so that a
     negation in one sentence does not leak into another
     (e.g. "Stable cardiomegaly. No pleural effusion.").
  2. For every pathology keyword found inside a clause, inspect a small
     character window around that keyword.
        * a negation phrase in the window   -> ``negative``
        * an uncertainty phrase in the window -> ``uncertain``
        * otherwise                          -> ``positive``
  3. If a pathology is mentioned in several clauses, combine the candidate
     labels with a simple priority (positive > uncertain > negative).
  4. If a pathology keyword never appears -> ``not_mentioned``.

This is intentionally simple and readable rather than perfect; a few hard
cases (out-of-vocabulary phrasing such as "not present" or "none seen") are
left uncaught on purpose so the benchmark can surface them in error analysis.
"""
from __future__ import annotations

import re
from typing import Dict, List

from labelers.keyword_labeler import KEYWORDS
from preprocessing.clean_reports import clean_text
from utils import PATHOLOGIES

# Negation cues. "no " keeps the trailing space so it does not match words such
# as "normal" or "none".
NEGATION_PHRASES: List[str] = [
    "no evidence of",
    "negative for",
    "no ",
    "without",
    "absence of",
    "free of",
    "ruled out",
    "not seen",
]

# Uncertainty / hedging cues.
UNCERTAINTY_PHRASES: List[str] = [
    "possible",
    "possibly",
    "cannot exclude",
    "cannot be excluded",
    "may represent",
    "may reflect",
    "suggestive of",
    "suspicious for",
    "questionable",
    "equivocal",
]

# Characters inspected on each side of a matched keyword.
WINDOW = 45

# Priority when the same pathology is mentioned in more than one clause.
_PRIORITY = {"positive": 3, "uncertain": 2, "negative": 1, "not_mentioned": 0}


def _split_clauses(text: str) -> List[str]:
    """Split cleaned text into clauses on sentence terminators."""
    parts = re.split(r"[.;\n]", text)
    return [part.strip() for part in parts if part.strip()]


def _context_label(clause: str, keyword: str) -> str:
    """Decide a label for a single keyword within one clause."""
    label = "positive"
    start = 0
    while True:
        idx = clause.find(keyword, start)
        if idx == -1:
            break
        window_start = max(0, idx - WINDOW)
        window_end = min(len(clause), idx + len(keyword) + WINDOW)
        window = clause[window_start:window_end]
        # Negation takes precedence over uncertainty within a window.
        if any(phrase in window for phrase in NEGATION_PHRASES):
            return "negative"
        if any(phrase in window for phrase in UNCERTAINTY_PHRASES):
            label = "uncertain"
        start = idx + len(keyword)
    return label


class NegationRuleLabeler:
    """Rule-based labeler aware of negation and uncertainty."""

    name = "negation_rule"

    def __init__(self) -> None:
        self.keywords = KEYWORDS

    def label_report(self, report_text: str) -> Dict[str, str]:
        text = clean_text(report_text)
        clauses = _split_clauses(text)
        result: Dict[str, str] = {p: "not_mentioned" for p in PATHOLOGIES}
        for pathology in PATHOLOGIES:
            for clause in clauses:
                for keyword in self.keywords[pathology]:
                    if keyword in clause:
                        candidate = _context_label(clause, keyword)
                        if _PRIORITY[candidate] > _PRIORITY[result[pathology]]:
                            result[pathology] = candidate
        return result
