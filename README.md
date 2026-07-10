# Radiology Report Labeling Mini-Benchmark
### Keyword, Negation-Aware, and Gemma-Based Labeling

A small, honest, runnable benchmark comparing three approaches to extracting
pathology labels from chest X-ray radiology-style reports: a **keyword
baseline**, a **negation-aware rule-based labeler**, and a **Gemma-based LLM
labeler** (via local Ollama, with a mock mode so the project runs immediately).

> **This project is a lightweight preparation benchmark and is not clinically
> validated.** It is intended as a research-preparation exercise, not a clinical
> tool.

---

## Motivation

Automated labeling of radiology reports is the backbone of large-scale medical
imaging research: chest X-ray classifiers are typically trained on labels that
were themselves extracted from free-text reports (as in CheXpert / MIMIC-CXR).
If those labels are wrong, every downstream model inherits the error.

### Why automated radiology report labeling matters
- Manual annotation of hundreds of thousands of reports is infeasible.
- Label quality directly bounds the quality of any downstream imaging model.
- Reproducible benchmarks let us compare labeling methods honestly.

### Why negation and uncertainty are difficult
Radiologists constantly write about findings that are **absent** or
**uncertain**:
- *"No evidence of pneumothorax or pleural effusion."* — the pathology words are
  present, but the finding is **negative**.
- *"Pleural effusion cannot be excluded."* — **uncertain**, not positive.
- *"Right lower lobe opacity may represent pneumonia."* — **uncertain**.

A naive keyword matcher labels all of these **positive** because the pathology
word appears. Handling negation, uncertainty, and clinical context is exactly
the hard part.

---

## Core research question

> How do keyword-based, negation-aware, and LLM-based labeling approaches differ
> when extracting pathology labels from chest X-ray radiology-style reports
> containing negation, uncertainty, and clinical context?

**Pathologies:** pneumonia · pleural effusion · pneumothorax · cardiomegaly
**Label space:** `positive` · `negative` · `uncertain` · `not_mentioned`

---

## Methods compared

| Method | Idea | Handles negation? | Handles uncertainty? |
|---|---|:---:|:---:|
| **Keyword baseline** | Pathology word present → `positive`, else `not_mentioned` | ❌ | ❌ |
| **Negation-aware rule-based** | Clause splitting + window-based negation/uncertainty cues | ✅ | ✅ |
| **Gemma LLM (Ollama)** | Prompt the model to return JSON labels | ✅ (learned) | ✅ (learned) |

### Why Gemma via Ollama is optional
The LLM labeler runs **locally** through [Ollama](https://ollama.com) and needs
no paid API and no API keys. Because not everyone has Ollama installed, the
default mode is **`mock`**: the LLM labeler reuses the negation-aware labeler so
the whole pipeline runs out of the box. Switch to real Gemma only if you have
Ollama running.

---

## Dataset

The public demo dataset is **fully synthetic**: 30 artificial chest-X-ray-style
report snippets written for this project. They cover clear positives, clear
negations, uncertainty phrasing, not-mentioned cases, mixed multi-pathology
reports, and deliberately difficult cases where keyword matching fails.

- File: [`data/synthetic_demo_reports.csv`](data/synthetic_demo_reports.csv)
- Columns: `report_id`, `report_text`, `pneumonia_gold`,
  `pleural_effusion_gold`, `pneumothorax_gold`, `cardiomegaly_gold`
- Gold values: `positive`, `negative`, `uncertain`, `not_mentioned`

You can point the pipeline at your own local CSV (same columns) via
`data_path` in a config file.

> **Real MIMIC-CXR / MIMIC-CXR-JPG data is NOT included and is NOT
> redistributed here.** No real clinical reports are used. Keep any private
> data under `data/private/` (git-ignored).

---

## Repository layout

```
radiology-report-labeling-benchmark/
├── README.md
├── requirements.txt
├── configs/
│   ├── demo_mock.yaml
│   └── demo_ollama_gemma.yaml
├── data/
│   └── synthetic_demo_reports.csv
├── src/
│   ├── main.py
│   ├── utils.py
│   ├── preprocessing/clean_reports.py
│   ├── labelers/{keyword_labeler,negation_rule_labeler,llm_labeler}.py
│   ├── evaluation/{metrics,error_analysis}.py
│   └── visualization/plot_results.py
├── experiments/            # generated: results, predictions, error analysis, plots
├── docs/                   # GitHub Pages dashboard (index.html, style.css, assets/plots)
└── report/mini_report.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ (developed on 3.13). Runs on a normal laptop.

## How to run (demo / mock mode)

From the repository root:

```bash
python src/main.py --config configs/demo_mock.yaml
```

This will:
1. load the config and synthetic data,
2. run the keyword, negation-aware, and mock-LLM labelers,
3. write `experiments/predictions.csv`, `experiments/results.csv`, and
   `experiments/error_analysis.csv`,
4. generate plots in `experiments/plots/` and copy them to
   `docs/assets/plots/`,
5. print a clean summary to the terminal.

## How to run (optional Ollama / Gemma mode)

First, in a separate terminal, start a Gemma model with Ollama:

```bash
ollama run gemma3:1b
# or
ollama run gemma3:4b
```

Then run:

```bash
python src/main.py --config configs/demo_ollama_gemma.yaml
```

The LLM labeler sends each report with a prompt that asks for **JSON only** and
parses it robustly (stripping code fences, extracting the first JSON object, and
validating labels). If Ollama is unavailable, the run prints a helpful message
and falls back to mock mode (`fallback_to_mock: true`).

## How to generate plots

Plots are generated automatically by `main.py`. They are written to
`experiments/plots/` and copied to `docs/assets/plots/`:
- `macro_f1_by_method.png`
- `per_pathology_f1.png`
- `error_count_by_method.png`
- `confusion_matrix_pneumonia.png`

## How to view the GitHub Pages dashboard

Open [`docs/index.html`](docs/index.html) locally in a browser, or enable
GitHub Pages: **Settings → Pages → Source: `Deploy from a branch` → Branch:
`main` → Folder: `/docs`**. The site will then be served at
`https://<your-username>.github.io/radiology-report-labeling-benchmark/`.

---

## Results (synthetic demo, mock mode)

Macro-F1 is averaged over the four label classes per pathology, then averaged
across pathologies. Numbers below come from the included synthetic dataset.

| Method | Overall accuracy | Macro-F1 | Exact-match / report |
|---|:---:|:---:|:---:|
| keyword | 0.708 | 0.353 | 0.267 |
| negation_rule | 0.875 | 0.749 | 0.567 |
| llm_mock | 0.875 | 0.749 | 0.567 |

*(In mock mode the LLM labeler intentionally mirrors the negation-aware
labeler. With real Gemma via Ollama the `llm_ollama` numbers will differ.)*

The keyword baseline collapses on negated and uncertain statements, while the
negation-aware labeler recovers most of them. See
[`report/mini_report.md`](report/mini_report.md) for discussion and
[`docs/index.html`](docs/index.html) for embedded plots.

---

## Limitations

- Tiny **synthetic** dataset (30 snippets); numbers are illustrative only.
- The rule-based labeler uses a small, fixed cue list and simple windows; it
  misses out-of-vocabulary phrasing (e.g. *"not present"*, *"none seen"*) and
  findings expressed without a keyword (e.g. *"heart size is normal"*).
- Mock mode is not a real language model; it exists so the repo runs anywhere.
- Not clinically validated; **not** a clinical tool.

## Future work

- Evaluate real **Gemma** (and other local models) via Ollama on larger,
  properly annotated report sets.
- Move toward a **MIMIC-CXR / MIMIC-CXR-JPG** labeling benchmark (with proper
  data-use agreements — data never redistributed).
- Study how label quality affects **downstream chest X-ray classifier training**.
- Add inter-annotator style agreement metrics and calibration of LLM confidence.

---

## Repository

GitHub: `https://github.com/<your-username>/radiology-report-labeling-benchmark`
*(placeholder — update after publishing)*
