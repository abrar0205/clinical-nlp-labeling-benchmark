# Radiology Report Labeling Mini-Benchmark: A Research-Preparation Study

*Keyword, Negation-Aware, and Hugging Face Local Inference for Chest X-ray Report Labeling*

---

## Abstract

Large-scale chest X-ray classifiers are trained on pathology labels extracted from free-text radiology reports; the quality of those labels bounds the quality of every downstream model. Extracting labels reliably is hard precisely because reports are full of **negation** ("no pleural effusion") and **uncertainty** ("effusion cannot be excluded"). This mini-benchmark compares three labeling strategies — a keyword baseline, a transparent negation-aware rule-based labeler, and an optional Hugging Face instruction-model labeler run locally with `transformers` — across four pathologies and four label classes on a small, fully synthetic dataset. On the synthetic demo, the keyword baseline reaches a macro-F1 of **0.35**, while the negation-aware labeler reaches **0.75**, illustrating how much of the labeling problem is negation and uncertainty rather than concept detection. This is a preparation exercise, not a clinical tool, and is not clinically validated.

## 1. Introduction

Datasets such as CheXpert and MIMIC-CXR provide image labels that were themselves produced by automatic labelers applied to radiology reports. Errors introduced at the labeling stage propagate silently into any model trained on the images. Understanding *how* labelers fail — and which failures dominate — is therefore a prerequisite for responsible medical imaging research. This project reproduces the *core challenge* at a small scale rather than reproducing any full benchmark.

## 2. Research question

> How do keyword-based, negation-aware, and LLM-based labeling approaches differ when extracting pathology labels from chest X-ray radiology-style reports containing negation, uncertainty, and clinical context?

- **Pathologies:** pneumonia, pleural effusion, pneumothorax, cardiomegaly.
- **Label space:** `positive`, `negative`, `uncertain`, `not_mentioned`.

## 3. Methods

### 3.1 Preprocessing

Cleaning is intentionally conservative: lowercasing and whitespace normalisation only. Negation and uncertainty phrasing is preserved because it carries the signal the labelers depend on.

### 3.2 Keyword baseline

A small dictionary maps each pathology to surface keywords (e.g. pneumonia → {pneumonia, consolidation, infiltrate, airspace opacity}). If any keyword is present the label is `positive`, otherwise `not_mentioned`. The method has no notion of negation or uncertainty and is expected to fail on those cases; it establishes a naive floor.

### 3.3 Negation-aware rule-based labeler

The report is split into clauses on sentence terminators so a negation in one sentence does not leak into another. For each pathology keyword found in a clause, a small character window around the keyword is inspected for negation cues (*no, without, no evidence of, negative for, absence of, free of, ruled out, not seen*) or uncertainty cues (*possible, possibly, cannot exclude, may represent, may reflect, suggestive of, suspicious for, questionable, equivocal*). Negation yields `negative`, uncertainty yields `uncertain`, otherwise `positive`; an unmentioned pathology yields `not_mentioned`. When a pathology is mentioned in multiple clauses, candidate labels are combined with a simple priority. The design is deliberately transparent and imperfect.

### 3.4 Hugging Face local-inference labeler

An `LLMLabeler` supports two modes. In **mock** mode it reuses the negation-aware labeler so the pipeline is fully runnable with no model download. In **huggingface** mode it loads an instruction model locally through `transformers` (default `Qwen/Qwen2.5-0.5B-Instruct`) and prompts the model to return **JSON only**. The response is parsed robustly — markdown fences are stripped, the first JSON object is extracted, labels are validated against the allowed set — and on malformed output the labeler returns `not_mentioned` for all conditions and records a parse error. No hosted API and no API keys are used.

## 4. Synthetic dataset

Thirty artificial report snippets were written for this project. None are copied from real reports. The set covers clear positives, clear negations, uncertainty phrasing, not-mentioned cases, mixed multi-pathology reports, and difficult cases where keyword matching fails (e.g. findings expressed without a keyword, or out-of-vocabulary negations such as "not present"). Each report has gold labels for all four pathologies.

## 5. Evaluation metrics

Using scikit-learn, for each method and pathology we compute accuracy and macro-averaged precision, recall, and F1 over the four label classes. We then average macro-F1 across the four pathologies and report an exact-match rate (fraction of reports for which all four labels are correct). Results are written to `experiments/results.csv`; per-row predictions and a typed error analysis are written to `experiments/predictions.csv` and `experiments/error_analysis.csv`.

## 6. Expected / observed results

On the included synthetic demo (mock mode):

| Method | Overall accuracy | Macro-F1 | Exact-match / report |
|---|:---:|:---:|:---:|
| keyword | 0.708 | 0.353 | 0.267 |
| negation_rule | 0.875 | 0.749 | 0.567 |
| llm_mock | 0.875 | 0.749 | 0.567 |

The keyword baseline is competitive on accuracy for clearly positive/absent concepts but collapses on macro-F1 because it cannot represent `negative` or `uncertain`. The negation-aware labeler recovers most negations and many uncertain cases. In mock mode the LLM labeler mirrors the rule-based labeler by construction; with real Hugging Face local inference, `llm_hf_local` numbers will differ, and the comparison of prompt-based labeling against explicit rules becomes the interesting axis.

## 7. Error analysis

Each incorrect prediction is assigned a coarse type: `false_positive_negation_error`, `uncertainty_misclassified`, `false_negative_missed_concept`, `not_mentioned_confusion`, `llm_parse_error`, or `other`. On the demo, false-positive negation errors dominate the keyword baseline, confirming that negation — not concept detection — is the main failure mode. The rule-based labeler's residual errors concentrate on out-of-vocabulary phrasing and on findings stated without an explicit keyword (notably several cardiomegaly cases such as "the heart is enlarged").

## 8. Limitations

- The dataset is tiny and synthetic; absolute numbers are illustrative only.
- The rule-based cue lists and windows are fixed and simple.
- Mock mode is not a language model; it guarantees runnability, not realism.
- The project is not clinically validated and is not a clinical tool.
- No real MIMIC-CXR data is used or redistributed.
- The current project uses prompt-based inference only; it does not fine-tune a model.

## 9. Future work

Natural extensions include evaluating additional local Hugging Face instruction models on larger, properly annotated corpora; measuring how label noise propagates into **downstream chest X-ray classifier training**; and moving toward a MIMIC-CXR / MIMIC-CXR-JPG labeling benchmark under appropriate data-use agreements (never redistributing the data). Fine-tuning should only be explored after access to a properly annotated dataset.
