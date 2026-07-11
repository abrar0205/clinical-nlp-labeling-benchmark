# Running the Hugging Face model locally

This project can run an instruction model locally with Hugging Face `transformers`.

You do not need:

- a hosted API,
- an API key,
- Ollama,
- real clinical data.

The model is downloaded once into your Hugging Face cache and reused from there in later runs.

Default model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

This model was chosen because it is small enough for a first local CPU run on a normal laptop. It is not expected to be clinically strong; it is used here to test the full local inference pipeline.

---

## What local inference means

In this repo, local inference means:

```text
report text
→ prompt
→ local Hugging Face model
→ generated JSON labels
→ evaluation against gold labels
```

This is not fine-tuning. Fine-tuning would mean training/adapting a model on labeled data. This project only uses a pretrained instruction model and asks it to generate labels.

---

## Windows setup

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the mock pipeline first. This does not download a model:

```powershell
python src/main.py --config configs/demo_mock.yaml
```

Then test local Hugging Face loading:

```powershell
python src/check_huggingface.py
```

The first run may take time because it downloads the model files. Later runs should load from cache.

Run the real local Hugging Face benchmark:

```powershell
python src/main.py --config configs/demo_hf_local.yaml
```

---

## macOS / Linux setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/main.py --config configs/demo_mock.yaml
python src/check_huggingface.py
python src/main.py --config configs/demo_hf_local.yaml
```

---

## Where the model is stored

The model is not stored inside this repository.

Typical Windows cache location:

```text
C:\Users\<your-user>\.cache\huggingface\hub
```

Typical macOS/Linux cache location:

```text
~/.cache/huggingface/hub
```

You normally do not need to edit or move these files manually.

---

## Strict vs fallback configs

Strict real model run:

```bash
python src/main.py --config configs/demo_hf_local.yaml
```

This config uses:

```yaml
fallback_to_mock: false
```

So if the model cannot be loaded, the run stops instead of silently creating mock results.

Forgiving run:

```bash
python src/main.py --config configs/demo_hf_local_fallback.yaml
```

This falls back to mock mode if the model cannot be loaded. Results from this mode are labeled:

```text
llm_hf_fallback_mock
```

They should not be treated as real Hugging Face model results.

---

## How to verify a real Hugging Face run

After running the benchmark, open:

```text
experiments/run_metadata.json
```

For a real local model run, check:

```json
"real_huggingface_used": true,
"fallback_mock_used": false,
"real_hf_calls": 30
```

Also check `experiments/results.csv`. It should contain:

```text
llm_hf_local
```

If you see `llm_mock` or `llm_hf_fallback_mock`, the LLM result was not produced by the real local model.

---

## Common notes

- A warning about unauthenticated Hugging Face requests is usually fine for this small model.
- A Windows symlink warning is also usually fine; caching still works, but may use slightly more disk space.
- CPU inference can be slow. That is normal for a laptop without an NVIDIA GPU.
- Start with the default 0.5B model before trying larger models.
