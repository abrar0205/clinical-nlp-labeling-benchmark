"""LLM-based labeler with mock mode and Hugging Face local inference.

Modes
-----
* ``mock``        : uses the negation-aware rule labeler internally so the whole
                    project is runnable with no model download. Deterministic.
* ``huggingface`` : downloads/loads a local Hugging Face model with
                    ``transformers`` and prompts it to return JSON labels.

No hosted APIs, no API keys, and no clinical data are used.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Tuple

from labelers.negation_rule_labeler import NegationRuleLabeler
from utils import LABELS, PATHOLOGIES

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
    """Label reports with mock rules or a local Hugging Face model."""

    def __init__(
        self,
        mode: str = "mock",
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        fallback_to_mock: bool = True,
        max_new_tokens: int = 160,
        cache_dir: str | None = None,
    ) -> None:
        self.mode = mode
        self.model = model
        self.fallback_to_mock = fallback_to_mock
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir

        self._mock_backend = NegationRuleLabeler()
        self._tokenizer = None
        self._model_obj = None
        self._device = "cpu"
        self._load_error: str | None = None

        # Counters used for honest run metadata.
        self.parse_errors = 0
        self.real_calls = 0
        self.fallback_calls = 0
        self._use_real = mode == "huggingface"
        self.name = "llm_mock" if mode == "mock" else "llm_hf_local"

    # ------------------------------------------------------------------ setup
    def resolve(self) -> Tuple[str, bool]:
        """Decide the effective method name and whether to run this labeler.

        Returns ``(name, active)``. ``active=False`` means the caller should stop
        or skip this labeler depending on the config.

        Naming is honest about what actually ran:
          * mock mode                            -> ``llm_mock``
          * Hugging Face model loaded locally    -> ``llm_hf_local``
          * HF load failed + fallback requested  -> ``llm_hf_fallback_mock``
        """
        if self.mode == "mock":
            self.name = "llm_mock"
            self._use_real = False
            return self.name, True

        if self.mode != "huggingface":
            self._load_error = f"Unsupported LLM mode: {self.mode}"
            if self.fallback_to_mock:
                self.name = "llm_hf_fallback_mock"
                self._use_real = False
                return self.name, True
            return self.name, False

        try:
            self._load_hf_model()
        except Exception as exc:  # noqa: BLE001 - keep message user-friendly
            self._load_error = str(exc)
            if self.fallback_to_mock:
                self.name = "llm_hf_fallback_mock"
                self._use_real = False
                return self.name, True
            self.name = "llm_hf_local"
            self._use_real = False
            return self.name, False

        self.name = "llm_hf_local"
        self._use_real = True
        return self.name, True

    def _load_hf_model(self) -> None:
        """Load tokenizer and model from Hugging Face into local memory/cache."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model_obj = AutoModelForCausalLM.from_pretrained(
            self.model,
            cache_dir=self.cache_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._model_obj.to(self._device)
        self._model_obj.eval()

    # ------------------------------------------------------------------ public
    def label_report(self, report_text: str) -> Dict[str, str]:
        if not self._use_real:
            return self._mock_label(report_text)

        try:
            raw = self._call_huggingface(report_text)
        except Exception as exc:  # noqa: BLE001 - user-facing fallback path
            if self.fallback_to_mock:
                self.fallback_calls += 1
                print(
                    f"[LLMLabeler] Hugging Face generation failed ({exc}); "
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

    def _call_huggingface(self, report_text: str) -> str:
        """Run local prompt-based inference with the loaded HF model."""
        import torch

        if self._tokenizer is None or self._model_obj is None:
            self._load_hf_model()

        assert self._tokenizer is not None
        assert self._model_obj is not None

        prompt = build_prompt(report_text)

        # Prefer chat template for instruction-tuned chat models; fall back to
        # plain text prompt if the tokenizer does not define one.
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted = prompt

        inputs = self._tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]

        pad_token_id = self._tokenizer.eos_token_id
        with torch.no_grad():
            generated = self._model_obj.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        new_tokens = generated[0][input_length:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _parse_response(self, raw: str) -> Tuple[Dict[str, str], bool]:
        """Robustly parse model output into a validated label dictionary."""
        text = str(raw).strip()
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

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
