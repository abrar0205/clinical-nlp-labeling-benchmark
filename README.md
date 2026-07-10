# Clinical NLP Radiology Report Labeling Mini-Benchmark
### Keyword, Negation-Aware, and Gemma-Based Labeling

A small, honest, reproducible benchmark comparing three approaches to extracting
pathology labels from chest X-ray radiology-style reports: a **keyword
baseline**, a **negation-aware rule-based labeler**, and an optional
**Gemma-based LLM labeler** run locally via [Ollama](https://ollama.com). A
**mock LLM mode** lets the whole project run immediately with no external setup.

> **This project is a lightweight preparation benchmark and is not clinically
> validated. It is not a clinical tool.** It uses only synthetic demo reports.

---

## 1. Project overview

Chest X-ray classifiers are trained on pathology labels that are themselves
extracted from free-text radiology reports (as in CheXpert / MIMIC-CXR). If
those labels are wrong, every downstream model inherits the error. Extracting
labels reliably is hard precisely because reports are full of **negation** ("no
pleural effusion") and **uncertainty** ("effusion cannot be excluded").

This repository reproduces that *core challenge* at a small scale: it compares
labeling strategies on four pathologies and four label classes, and produces
reproducible result files, error analysis, run metadata, and plots — all saved
locally under `experiments/`.

**Core research question:** How do keyword-based, negation-aware, and LLM-based
labeling approaches differ when extracting pathology labels from chest X-ray
reports containing negation, uncertainty, and clinical context?

- **Pathologies:** pneumonia · pleural effusion · pneumothorax · cardiomegaly
- **Label space:** `positive` · `negative` · `uncertain` · `not_mentioned`

## 2. Methods

| Method | Idea | Handles negation? | Handles uncertainty? |
|---|---|:---:|:---:|
| **keyword baseline** | Pathology word present → `positive`, else `not_mentioned` | ❌ | ❌ |
| **negation-aware rule-based** | Clause splitting + window-based negation/uncertainty cues | ✅ | ✅ |
| **Gemma via Ollama (optional)** | Local LLM prompted to return JSON labels | ✅ (learned) | ✅ (learned) |
| **mock LLM** | Reuses the rule-based labeler for reproducibility without any LLM | ✅ | ✅ |

The LLM labeler runs **locally** through Ollama and needs no paid API and no API
keys. Because not everyone has Ollama installed, a **mock** mode keeps the
pipeline fully runnable out of the box.

## 3. Dataset

The included dataset is **fully synthetic**: 30 artificial chest-X-ray-style
report snippets written for this project. They cover clear positives, clear
negations, uncertainty phrasing, not-mentioned cases, mixed multi-pathology
reports, and deliberately difficult cases where keyword matching fails.

- File: [`data/synthetic_demo_reports.csv`](data/synthetic_demo_reports.csv)
- Columns: `report_id`, `report_text`, `pneumonia_gold`,
  `pleural_effusion_gold`, `pneumothorax_gold`, `cardiomegaly_gold`
- Gold values: `positive`, `negative`, `uncertain`, `not_mentioned`

You can point the pipeline at your own local CSV (same columns) via `data_path`
in a config file.

> **No real MIMIC-CXR / MIMIC-CXR-JPG data is included or redistributed.** No
> real clinical reports are used. Keep any private data under `data/private/`
> (git-ignored).

## 4. Quick run without Ollama (default)

Requires Python 3.9+ (developed on 3.13). Runs on a normal laptop.

```bash
pip install -r requirements.txt
python src/main.py --config configs/demo_mock.yaml
```

This runs the keyword, negation-aware, and **mock** LLM labelers and writes all
outputs to `experiments/`.

## 5. Run real Gemma locally with Ollama (optional)

See [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md) for full Windows / macOS / Linux steps.
In short:

```bash
ollama run gemma3:1b                       # keep this running
python src/check_ollama.py                 # verify the connection
python src/main.py --config configs/demo_ollama_gemma.yaml
```

Honesty notes:

- **Mock mode is not a real LLM result.**
- **Real Gemma** results appear with the method name **`llm_gemma_ollama`**.
- **Fallback** results (Ollama requested but unreachable, forgiving config)
  appear as **`llm_gemma_fallback_mock`** — never mislabeled as real Gemma.
- The strict config `configs/demo_ollama_gemma.yaml` uses
  `fallback_to_mock: false`, so if Ollama is unavailable it **stops clearly**
  instead of silently mocking. Use
  `configs/demo_ollama_gemma_fallback.yaml` for a forgiving run.
- Always check [`experiments/run_metadata.json`](experiments) to confirm the
  run type.

## 6. Outputs

Every run generates, under `experiments/`:

```
experiments/predictions.csv      # per (report, method, pathology) predictions
experiments/results.csv          # accuracy + macro precision/recall/F1
experiments/error_analysis.csv   # typed errors for incorrect predictions
experiments/run_metadata.json    # exactly what ran (mock vs real Gemma)
experiments/plots/               # macro-F1, per-pathology F1, errors, confusion matrix
```

Plots are written **only** to `experiments/plots/`.

## 7. How to commit local Gemma results

After running real Gemma locally, you can commit the generated artifacts so
others can see your results:

```bash
git add experiments/results.csv experiments/predictions.csv \
        experiments/error_analysis.csv experiments/run_metadata.json \
        experiments/plots/
git commit -m "Add local Gemma (gemma3:1b) benchmark results"
git push
```

`run_metadata.json` documents that the results came from real Gemma
(`"real_ollama_gemma_used": true`).

## 8. Repository layout

```
clinical-nlp-labeling-benchmark/
├── README.md
├── OLLAMA_SETUP.md
├── requirements.txt
├── configs/
│   ├── demo_mock.yaml
│   ├── demo_ollama_gemma.yaml            # strict: fails if Ollama down
│   └── demo_ollama_gemma_fallback.yaml   # forgiving: falls back to mock
├── data/
│   └── synthetic_demo_reports.csv
├── src/
│   ├── main.py
│   ├── check_ollama.py
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
- The rule-based labeler uses small, fixed cue lists and simple windows; it
  misses out-of-vocabulary phrasing (e.g. "not present", "none seen") and
  findings expressed without a keyword (e.g. "the heart is enlarged").
- LLM output may vary between runs and models.
- **No real MIMIC-CXR data is included or redistributed.**

## 10. Future work

- Evaluate real Gemma (and other local models) via Ollama on larger, properly
  annotated report sets.
- Study how label quality affects **downstream chest X-ray classifier training**.
- Move toward a MIMIC-CXR / MIMIC-CXR-JPG labeling benchmark under appropriate
  data-use agreements (never redistributing the data).

See [`report/mini_report.md`](report/mini_report.md) for the academic write-up.
