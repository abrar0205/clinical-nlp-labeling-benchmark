"""LLM-based labeler with a mock mode and an Ollama/Gemma mode.

Modes
-----
* ``mock``   : uses the negation-aware rule labeler internally so the whole
               project is runnable with no external dependencies. Deterministic
               (no randomness).
* ``ollama`` : calls a locally running Ollama server (default Gemma) via the
               HTTP API. The model is asked to return JSON only; the response is
               parsed robustly and falls back gracefully on malformed output.

No paid APIs, no API keys, and no external hosted services are used.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Tuple

import requests

from labelers.negation_rule_labeler import NegationRuleLabeler
from utils import LABELS, PATHOLOGIES

# The report text is injected in place of <REPORT_TEXT>. Using .replace() (rather
# than str.format) avoids having to escape the JSON braces in the template.
PROMPT_TEMPLATE = """You are labeling a chest X-ray radiology report for research evaluation.
For each condition, assign exactly one label from:
positive, negative, uncertain, not_mentioned.

Conditions:
pneumonia, pleural_effusion, pneumothorax, cardiomegaly

Definitions:
positive = condition is present
negative = condition is explicitly absent or negated
uncertain = report expresses uncertainty, possibility, or cannot exclude
not_mentioned = condition is not discussed

Return only valid JSON with this schema:
{
  "pneumonia": "...",
  "pleural_effusion": "...",
  "pneumothorax": "...",
  "cardiomegaly": "..."
}

Report:
<REPORT_TEXT>"""


def build_prompt(report_text: str) -> str:
    """Insert the report text into the prompt template."""
    return PROMPT_TEMPLATE.replace("<REPORT_TEXT>", str(report_text))


class LLMLabeler:
    """Label reports with an LLM (mock or local Ollama/Gemma)."""

    def __init__(
        self,
        mode: str = "mock",
        model: str = "gemma3:1b",
        endpoint: str = "http://localhost:11434/api/generate",
        fallback_to_mock: bool = True,
        timeout: int = 60,
    ) -> None:
        self.mode = mode
        self.model = model
        self.endpoint = endpoint
        self.fallback_to_mock = fallback_to_mock
        self.timeout = timeout
        self._mock_backend = NegationRuleLabeler()
        # Counters used for honest run metadata.
        self.parse_errors = 0
        self.real_calls = 0
        self.fallback_calls = 0
        self.available: bool | None = None
        # ``_use_real`` decides whether label_report actually calls Ollama.
        self._use_real = mode == "ollama"
        # Method name is provisional until resolve() is called.
        self.name = "llm_mock" if mode == "mock" else "llm_gemma_ollama"

    # ------------------------------------------------------------------ setup
    def check_available(self) -> bool:
        """Return True if the local Ollama server responds on its base URL."""
        base = self.endpoint.split("/api/")[0]
        try:
            response = requests.get(base, timeout=5)
            self.available = response.status_code == 200
        except Exception:
            self.available = False
        return bool(self.available)

    def resolve(self) -> Tuple[str, bool]:
        """Decide the effective method name and whether to skip this labeler.

        Returns ``(name, active)``. ``active=False`` means the labeler should be
        skipped entirely (ollama requested, unreachable, and no fallback).

        Naming is honest about what actually ran:
          * mock mode                         -> ``llm_mock``
          * ollama reachable                  -> ``llm_gemma_ollama``
          * ollama unreachable + fallback     -> ``llm_gemma_fallback_mock``
        """
        if self.mode == "mock":
            self.name = "llm_mock"
            self._use_real = False
            return self.name, True

        # ollama mode
        if self.check_available():
            self.name = "llm_gemma_ollama"
            self._use_real = True
            return self.name, True

        if self.fallback_to_mock:
            self.name = "llm_gemma_fallback_mock"
            self._use_real = False
            return self.name, True

        # Unreachable and no fallback requested: caller should skip.
        self.name = "llm_gemma_ollama"
        self._use_real = False
        return self.name, False

    # ------------------------------------------------------------------ public
    def label_report(self, report_text: str) -> Dict[str, str]:
        if not self._use_real:
            return self._mock_label(report_text)

        # Real Ollama/Gemma call.
        try:
            raw = self._call_ollama(report_text)
        except Exception as exc:  # network / server errors
            if self.fallback_to_mock:
                self.fallback_calls += 1
                print(
                    f"[LLMLabeler] Ollama call failed ({exc}); "
                    "mock fallback for this report."
                )
                return self._mock_label(report_text)
            self.parse_errors += 1
            return self._default_labels(parse_error=True)

        self.real_calls += 1
        labels, parse_error = self._parse_response(raw)
        if parse_error:
            self.parse_errors += 1
        return labels

    # ----------------------------------------------------------------- helpers
    def _mock_label(self, report_text: str) -> Dict[str, str]:
        labels = dict(self._mock_backend.label_report(report_text))
        labels["parse_error"] = False
        return labels

    def _call_ollama(self, report_text: str) -> str:
        payload = {
            "model": self.model,
            "prompt": build_prompt(report_text),
            "stream": False,
            "format": "json",
        }
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    def _parse_response(self, raw: str) -> Tuple[Dict[str, str], bool]:
        """Robustly parse the model output into a validated label dict."""
        text = str(raw).strip()
        # Strip markdown code fences if present.
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        # Extract the first {...} block in case the model added prose.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._default_labels(parse_error=True), True

        labels: Dict[str, str] = {}
        for pathology in PATHOLOGIES:
            value = str(parsed.get(pathology, "not_mentioned")).strip().lower()
            labels[pathology] = value if value in LABELS else "not_mentioned"
        labels["parse_error"] = False
        return labels, False

    def _default_labels(self, parse_error: bool = False) -> Dict[str, str]:
        labels: Dict[str, str] = {p: "not_mentioned" for p in PATHOLOGIES}
        labels["parse_error"] = parse_error
        return labels
