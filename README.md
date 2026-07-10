# Clinical NLP Radiology Report Labeling Mini-Benchmark
### Keyword, Negation-Aware, and Hugging Face Local Inference

A small, honest, reproducible benchmark comparing three approaches to extracting pathology labels from chest X-ray radiology-style reports: a **keyword baseline**, a **negation-aware rule-based labeler**, and an optional **Hugging Face instruction-model labeler** run locally with `transformers`. A **mock LLM mode** lets the whole project run immediately with no model download.

> **This project is a lightweight preparation benchmark and is not clinically validated. It is not a clinical tool.** It uses only synthetic demo reports.

---

## 1. Project overview

Chest X-ray classifiers are often trained on pathology labels extracted from free-text radiology reports. If those labels are wrong, every downstream model inherits the error. Extracting labels reliably is hard precisely because reports are full of **negation** ("no pleural effusion") and **uncertainty** ("effusion cannot be excluded").

This repository reproduces that *core challenge* at a small scale: it compares labeling strategies on four pathologies and four label classes, and produces reproducible result files, error analysis, run metadata, and plots — all saved locally under `experiments/`.

**Core research question:** How do keyword-based, negation-aware, and LLM-based labeling approaches differ when extracting pathology labels from chest X-ray reports containing negation, uncertainty, and clinical context?

- **Pathologies:** pneumonia · pleural effusion · pneumothorax · cardiomegaly
- **Label space:** `positive` · `negative` · `uncertain` · `not_mentioned`

## 2. Methods

| Method | Idea | Handles negation? | Handles uncertainty? |
|---|---|:---:|:---:|
| **keyword baseline** | Pathology word present → `positive`, else `not_mentioned` | ❌ | ❌ |
| **negation-aware rule-based** | Clause splitting + window-based negation/uncertainty cues | ✅ | ✅ |
| **Hugging Face local model** | Local instruction model prompted to return JSON labels | ✅ (prompt-based) | ✅ (prompt-based) |
| **mock LLM** | Reuses the rule-based labeler for reproducibility without model download | ✅ | ✅ |

The LLM labeler can run **locally** using Hugging Face `transformers`; no hosted API and no API key are required. Because not everyone wants to download a model immediately, **mock** mode keeps the pipeline fully runnable out of the box.

## 3. Dataset

The included dataset is **fully synthetic**: 30 artificial chest-X-ray-style report snippets written for this project. They cover clear positives, clear negations, uncertainty phrasing, not-mentioned cases, mixed multi-pathology reports, and deliberately difficult cases where keyword matching fails.

- File: [`data/synthetic_demo_reports.csv`](data/synthetic_demo_reports.csv)
- Columns: `report_id`, `report_text`, `pneumonia_gold`, `pleural_effusion_gold`, `pneumothorax_gold`, `cardiomegaly_gold`
- Gold values: `positive`, `negative`, `uncertain`, `not_mentioned`

You can point the pipeline at your own local CSV (same columns) via `data_path` in a config file.

> **No real MIMIC-CXR / MIMIC-CXR-JPG data is included or redistributed.** No real clinical reports are used. Keep any private data under `data/private/` (git-ignored).

## 4. Quick run without model download

Requires Python 3.9+ (developed on 3.13). Runs on a normal laptop.

```bash
pip install -r requirements.txt
python src/main.py --config configs/demo_mock.yaml
```

This runs the keyword, negation-aware, and **mock** LLM labelers and writes all outputs to `experiments/`.

## 5. Run real Hugging Face local inference

See [`HUGGINGFACE_SETUP.md`](HUGGINGFACE_SETUP.md) for full Windows / macOS / Linux steps.

Default local model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

In short:

```bash
pip install -r requirements.txt
python src/check_huggingface.py
python src/main.py --config configs/demo_hf_local.yaml
```

The first local inference run may download the model into your Hugging Face cache. Later runs load it from cache.

Honesty notes:

- **Mock mode is not a real LLM result.**
- **Real Hugging Face local inference** appears with the method name **`llm_hf_local`**.
- **Fallback** results appear as **`llm_hf_fallback_mock`** and are never mislabeled as real model output.
- The strict config `configs/demo_hf_local.yaml` uses `fallback_to_mock: false`, so if the model cannot be loaded it **stops clearly** instead of silently mocking.
- Use `configs/demo_hf_local_fallback.yaml` for a forgiving run.
- Always check [`experiments/run_metadata.json`](experiments) to confirm the run type.

## 6. Outputs

Every run generates, under `experiments/`:

```text
experiments/predictions.csv      # per (report, method, pathology) predictions
experiments/results.csv          # accuracy + macro precision/recall/F1
experiments/error_analysis.csv   # typed errors for incorrect predictions
experiments/run_metadata.json    # exactly what ran (mock vs real HF model)
experiments/plots/               # macro-F1, per-pathology F1, errors, confusion matrix
```

Plots are written **only** to `experiments/plots/`.

## 7. How to commit local Hugging Face results

After running real local Hugging Face inference, you can commit the generated artifacts so others can see your results:

```bash
git add experiments/results.csv experiments/predictions.csv \
        experiments/error_analysis.csv experiments/run_metadata.json \
        experiments/plots/
git commit -m "Add local Hugging Face benchmark results"
git push
```

`run_metadata.json` documents that the results came from real local inference (`"real_huggingface_used": true`).

## 8. Repository layout

```text
clinical-nlp-labeling-benchmark/
├── README.md
├── HUGGINGFACE_SETUP.md
├── requirements.txt
├── configs/
│   ├── demo_mock.yaml
│   ├── demo_hf_local.yaml            # strict: fails if model cannot load
│   └── demo_hf_local_fallback.yaml   # forgiving: falls back to mock
├── data/
│   └── synthetic_demo_reports.csv
├── src/
│   ├── main.py
│   ├── check_huggingface.py
│   ├── utils.py
│   ├── preprocessing/clean_reports.py
│   ├── labelers/{keyword_labeler,negation_rule_labeler,llm_labeler}.py
│   ├── evaluation/{metrics,error_analysis}.py
│   └── visualization/plot_results.py
├── experiments/            # generated outputs (results, predictions, plots, metadata)
└── report/mini_report.md
```

## 9. Limitations

- Tiny **synthetic** dataset (30 snippets); numbers are illustrative only.
- Not clinically validated; **not** a clinical tool.
- The rule-based labeler uses small, fixed cue lists and simple windows; it misses out-of-vocabulary phrasing and findings expressed without a keyword.
- LLM output may vary between models and environments.
- **No real MIMIC-CXR data is included or redistributed.**

## 10. Future work

- Evaluate other local Hugging Face instruction models on larger, properly annotated report sets.
- Study how label quality affects **downstream chest X-ray classifier training**.
- Move toward a MIMIC-CXR / MIMIC-CXR-JPG labeling benchmark under appropriate data-use agreements (never redistributing the data).
- Fine-tuning could be explored only after access to a properly annotated dataset; this project currently uses prompt-based local inference, not fine-tuning.

See [`report/mini_report.md`](report/mini_report.md) for the academic write-up.
