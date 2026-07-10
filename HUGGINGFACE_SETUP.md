# Running a Hugging Face model locally

This project can label synthetic radiology-style reports with a local Hugging Face instruction model through `transformers`. This is optional — the project runs fully in mock mode without downloading any model.

No hosted API, no API key, and no clinical data are used. The model is downloaded to your local Hugging Face cache on first use and then loaded from cache in later runs.

Recommended first model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Why this model first: it is small enough to test on a normal laptop and is suitable for a lightweight local-inference demo. You can later try larger instruction models if your machine has enough RAM/VRAM.

---

## What “local inference” means

Local inference means:

```text
Download model from Hugging Face Hub
→ load it locally with transformers
→ send a prompt
→ generate labels
```

This is not fine-tuning. Fine-tuning would mean training/adapting the model on your own labeled data. This project only performs prompt-based inference.

---

## Windows setup

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check that the Hugging Face model can be downloaded/loaded locally:

```powershell
python src/check_huggingface.py
```

Run the real local Hugging Face benchmark:

```powershell
python src/main.py --config configs/demo_hf_local.yaml
```

If your machine is slow, the first run may take a while because the model is downloaded and loaded locally.

---

## macOS / Linux setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/check_huggingface.py
python src/main.py --config configs/demo_hf_local.yaml
```

---

## Optional: choose another model

You can pass another model to the check script:

```bash
python src/check_huggingface.py --model Qwen/Qwen2.5-0.5B-Instruct
```

You can also edit the model in:

```text
configs/demo_hf_local.yaml
```

Example:

```yaml
llm:
  mode: huggingface
  model: Qwen/Qwen2.5-0.5B-Instruct
```

---

## Where models are cached

By default, Hugging Face stores downloaded model files in your user cache directory, not inside this repository.

Typical Windows cache location:

```text
C:\Users\<your-user>\.cache\huggingface\hub
```

Typical macOS/Linux cache location:

```text
~/.cache/huggingface/hub
```

You normally do not need to manage this manually.

---

## Strict vs fallback configs

Strict real model run:

```bash
python src/main.py --config configs/demo_hf_local.yaml
```

This uses:

```yaml
fallback_to_mock: false
```

So if the model cannot be loaded, the run stops clearly and does not silently create mock results.

Forgiving run:

```bash
python src/main.py --config configs/demo_hf_local_fallback.yaml
```

This falls back to mock mode if the model cannot be loaded. Results from this mode are labeled:

```text
llm_hf_fallback_mock
```

They are not mislabeled as real Hugging Face results.

---

## Verifying the run was real Hugging Face inference

After a run, open:

```text
experiments/run_metadata.json
```

For real local Hugging Face inference, check:

```json
"real_huggingface_used": true,
"fallback_mock_used": false,
"methods_run": ["keyword", "negation_rule", "llm_hf_local"]
```

If you see `llm_mock` or `llm_hf_fallback_mock`, the LLM result is not a real local model result.
