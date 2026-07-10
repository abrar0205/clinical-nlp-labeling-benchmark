"""Quick local Hugging Face model loading and generation check.

Usage
-----
    python src/check_huggingface.py
    python src/check_huggingface.py --model Qwen/Qwen2.5-0.5B-Instruct

This downloads/loads the requested model locally through transformers and sends a
small JSON-only prompt. Setup problems print a short message instead of a long
traceback.
"""
from __future__ import annotations

import argparse
import json
import sys

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def check(model_name: str, cache_dir: str | None = None) -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print("Missing Hugging Face dependencies.")
        print("Run: pip install -r requirements.txt")
        print(f"Details: {exc}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model locally: {model_name}")
    print(f"Device: {device}")
    print("This may download model files on the first run.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001 - user-facing setup script
        print("Could not load the Hugging Face model locally.")
        print("Check your internet connection, model name, and available disk/RAM.")
        print(f"Model: {model_name}")
        print(f"Details: {exc}")
        return 1

    prompt = 'Return only this JSON: {"status": "ok"}'
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as exc:  # noqa: BLE001 - user-facing setup script
        print("Model loaded, but generation failed.")
        print(f"Details: {exc}")
        return 1

    print("Model generated a response successfully.")
    print(f"Raw response: {text!r}")
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        parsed = json.loads(text[start:end]) if start != -1 and end > start else None
        print(f"Parsed JSON reply: {parsed}")
    except Exception:
        print("The response was not strict JSON, but local inference is working.")

    print("SUCCESS: your local Hugging Face inference setup is working.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test local Hugging Face model loading/inference."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--cache-dir", default=None, help="Optional local HF cache dir")
    args = parser.parse_args()
    sys.exit(check(args.model, args.cache_dir))


if __name__ == "__main__":
    main()
