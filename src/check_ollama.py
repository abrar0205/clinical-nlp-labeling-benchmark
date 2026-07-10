"""Quick connectivity test for a local Ollama / Gemma setup.

Usage
-----
    python src/check_ollama.py
    python src/check_ollama.py --model gemma3:4b

It sends a tiny JSON-only prompt to the local Ollama server and reports clearly
whether Ollama is reachable and whether the requested model responds. Normal
setup problems (server down, model not downloaded) print a short, friendly
message instead of a long traceback.
"""
from __future__ import annotations

import argparse
import json
import sys

import requests

DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:1b"


def base_url(endpoint: str) -> str:
    return endpoint.split("/api/")[0]


def check(model: str, endpoint: str) -> int:
    server = base_url(endpoint)

    # 1) Is the Ollama server reachable at all?
    try:
        ping = requests.get(server, timeout=5)
        ping.raise_for_status()
    except requests.exceptions.RequestException:
        print(
            f"Ollama is not reachable at {server}. "
            "Start Ollama and run: ollama run gemma3:1b"
        )
        return 1

    print(f"Ollama server is reachable at {server}.")

    # 2) Does the requested model respond to a tiny JSON prompt?
    payload = {
        "model": model,
        "prompt": 'Return only this JSON: {"status": "ok"}',
        "stream": False,
        "format": "json",
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=60)
    except requests.exceptions.RequestException as exc:
        print(f"Could not complete a generation request: {exc}")
        return 1

    if response.status_code == 404:
        print(
            f"Model '{model}' may not be downloaded yet. "
            f"Run: ollama run {model}"
        )
        return 1

    if response.status_code != 200:
        # Ollama returns 500 with an error body when a model is missing.
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        if "not found" in str(detail).lower() or "no such model" in str(detail).lower():
            print(
                f"Model '{model}' may not be downloaded yet. "
                f"Run: ollama run {model}"
            )
        else:
            print(f"Ollama returned HTTP {response.status_code}: {detail}")
        return 1

    raw = response.json().get("response", "")
    print(f"Model '{model}' responded successfully.")
    try:
        parsed = json.loads(raw)
        print(f"Parsed JSON reply: {parsed}")
    except (ValueError, TypeError):
        print(f"Raw reply (not strict JSON, but the model responded): {raw!r}")

    print("SUCCESS: your local Ollama/Gemma setup is working.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test a local Ollama / Gemma connection."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag.")
    parser.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT, help="Ollama generate endpoint."
    )
    args = parser.parse_args()
    sys.exit(check(args.model, args.endpoint))


if __name__ == "__main__":
    main()
