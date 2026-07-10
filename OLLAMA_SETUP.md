# Running Gemma locally with Ollama

This project can label reports with a real local LLM (Gemma) through
[Ollama](https://ollama.com). It is **optional** — the project runs fully in
mock mode without it. No paid APIs and no API keys are used; everything runs on
your own machine.

Recommended model first: **`gemma3:1b`**. It is lighter and easier to run on a
normal laptop. `gemma3:4b` is stronger but needs more memory; try it only after
`gemma3:1b` works.

---

## Windows (primary instructions)

1. **Download Ollama** from <https://ollama.com/download>.
2. **Install** the downloaded installer (default options are fine).
3. Open **PowerShell** or **Command Prompt**.
4. **Check the installation:**

   ```powershell
   ollama --version
   ```

5. **Download and run Gemma** (this pulls the model on first run and then keeps
   an interactive session open):

   ```powershell
   ollama run gemma3:1b
   ```

6. **Keep Ollama running.** Installing Ollama starts a background server on
   `http://localhost:11434`. Leave the `ollama run` window open, or ensure the
   Ollama app is running in the tray.

7. In **another terminal**, from the **repository root**, test the connection:

   ```powershell
   python src/check_ollama.py
   ```

   You should see `SUCCESS: your local Ollama/Gemma setup is working.`

8. **Run the real Gemma benchmark:**

   ```powershell
   python src/main.py --config configs/demo_ollama_gemma.yaml
   ```

### Optional stronger model

```powershell
ollama run gemma3:4b
python src/check_ollama.py --model gemma3:4b
```

To benchmark with `gemma3:4b`, set `model: gemma3:4b` in
`configs/demo_ollama_gemma.yaml` (or `..._fallback.yaml`) and rerun `main.py`.

---

## macOS

```bash
# Install (Homebrew) or download from https://ollama.com/download
brew install ollama
ollama --version
ollama run gemma3:1b          # keep this running

# in another terminal, from the repo root:
python src/check_ollama.py
python src/main.py --config configs/demo_ollama_gemma.yaml
```

## Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
ollama run gemma3:1b          # keep this running

# in another terminal, from the repo root:
python src/check_ollama.py
python src/main.py --config configs/demo_ollama_gemma.yaml
```

---

## Troubleshooting

- **"Ollama is not reachable ..."** — the server is not running. Start it with
  `ollama run gemma3:1b` (Windows: also check the Ollama tray app is running).
- **"Model may not be downloaded yet ..."** — run `ollama run gemma3:1b` once to
  pull the model, then retry.
- **The strict config fails clearly instead of mocking.**
  `configs/demo_ollama_gemma.yaml` uses `fallback_to_mock: false` on purpose, so
  it never silently produces mock results. If you want a forgiving run that
  falls back to mock, use `configs/demo_ollama_gemma_fallback.yaml`.

## Verifying the run was real Gemma

After a run, open `experiments/run_metadata.json` and check:

- `"real_ollama_gemma_used": true`
- `"fallback_mock_used": false`
- `"methods_run"` contains `"llm_gemma_ollama"`

If instead you see `"llm_gemma_fallback_mock"` or `"llm_mock"`, the LLM results
came from the mock labeler, not real Gemma.
