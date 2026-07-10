"""Light-touch preprocessing for radiology-style report snippets.

The cleaning is intentionally conservative. We lowercase and normalise
whitespace, but we deliberately keep clinical negation / uncertainty phrasing
intact (e.g. "no", "without", "cannot be excluded") so the downstream labelers
can still reason about it. Over-aggressive cleaning (stop-word removal,
stemming, punctuation stripping) would destroy exactly the signal we care
about, so we avoid it here.
"""
from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Return a lightly cleaned version of a report snippet.

    Steps:
      * lowercase for case-insensitive matching
      * collapse runs of whitespace into single spaces
      * trim leading/trailing whitespace
    """
    if text is None:
        return ""
    cleaned = str(text).lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
