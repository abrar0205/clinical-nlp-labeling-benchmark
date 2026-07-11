# Radiology Report Labeling Mini-Benchmark

*Keyword, negation-aware, and local Hugging Face inference for chest X-ray report labeling*

---

## Abstract

This mini-benchmark tests three ways of extracting structured pathology labels from short chest-X-ray-style reports: a keyword baseline, a negation-aware rule-based labeler, and a local Hugging Face instruction model loaded with `transformers`. The dataset is fully synthetic and contains 30 report snippets with gold labels for pneumonia, pleural effusion, pneumothorax, and cardiomegaly.

The latest local Hugging Face run used `Qwen/Qwen2.5-0.5B-Instruct` and completed all 30 reports without fallback. The negation-aware rule labeler performed best on this small dataset, reaching a macro-F1 of 0.749. The local Hugging Face model reached a macro-F1 of 0.368, showing that a small instruction model does not automatically outperform simple clinical text rules. The main value of the project is the evaluation pipeline and failure analysis, not the absolute score.

This is a research-preparation project only. It is not clinically validated and is not a clinical tool.

---

## 1. Motivation

Chest X-ray models often rely on labels extracted from free-text reports. If the report labeler makes mistakes, the image model trained on those labels inherits those mistakes.

The hard part is not only finding disease words. The report may say:

```text
No pleural effusion.
Pneumothorax cannot be excluded.
Findings suspicious for pneumonia.
```

These phrases require the labeler to distinguish `positive`, `negative`, `uncertain`, and `not_mentioned`.

---

## 2. Research question

> How do keyword-based, negation-aware, and local LLM-based approaches differ when extracting pathology labels from reports containing negation and uncertainty?

Pathologies:

```text
pneumonia, pleural_effusion, pneumothorax, cardiomegaly
```

Label space:

```text
positive, negative, uncertain, not_mentioned
```

---

## 3. Methods

### 3.1 Keyword baseline

The keyword baseline checks whether a pathology-related word appears in the report. If it finds the word, it predicts `positive`; otherwise it predicts `not_mentioned`.

This gives a simple baseline, but it cannot understand negation or uncertainty. For example, `No pneumothorax` can still be predicted as `positive` because the word pneumothorax appears.

### 3.2 Negation-aware rule labeler

The rule labeler looks for pathology terms together with nearby negation or uncertainty cues.

Examples of negation cues:

```text
no, without, no evidence of, negative for, absence of
```

Examples of uncertainty cues:

```text
possible, cannot exclude, may represent, suspicious for, questionable
```

This makes the rule method much stronger on the synthetic dataset, especially for short reports with explicit wording.

### 3.3 Hugging Face local-inference labeler

The LLM labeler loads an instruction model locally using Hugging Face `transformers`. The current default model is:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The model receives a prompt and is asked to return JSON labels for the four pathologies. The output is parsed and evaluated in the same format as the other methods.

This is local inference only. The project does not fine-tune a model.

---

## 4. Dataset

The dataset contains 30 synthetic chest-X-ray-style report snippets. No real reports are included. No MIMIC-CXR or MIMIC-CXR-JPG data is redistributed.

File:

```text
data/synthetic_demo_reports.csv
```

Each report has gold labels for all four pathologies.

---

## 5. Evaluation

For each method and pathology, the project computes:

- accuracy,
- macro precision,
- macro recall,
- macro-F1.

It also writes per-report predictions and a coarse error analysis.

Output files:

```text
experiments/predictions.csv
experiments/results.csv
experiments/error_analysis.csv
experiments/run_metadata.json
experiments/plots/
```

---

## 6. Latest results

The latest committed run used real local Hugging Face inference.

Metadata:

```text
model: Qwen/Qwen2.5-0.5B-Instruct
real_huggingface_used: true
fallback_mock_used: false
real_hf_calls: 30
```

Summary:

| Method | Overall accuracy | Macro-F1 |
|---|:---:|:---:|
| `keyword` | 0.708 | 0.353 |
| `negation_rule` | 0.875 | 0.749 |
| `llm_hf_local` | 0.667 | 0.368 |

The keyword method performs reasonably on accuracy because many labels are `not_mentioned`, but it struggles on macro-F1 because it cannot handle `negative` and `uncertain` labels well.

The negation-aware rule method performs best because many synthetic reports use explicit cues such as `no`, `without`, `cannot be excluded`, and `suspicious for`.

The local Hugging Face model completed the run successfully, but the 0.5B model often confused `negative` with `not_mentioned`. This is still useful: it shows why LLM-based labeling needs proper benchmarking and why stronger models or better prompts should be tested before drawing conclusions.

---

## 7. Error analysis

The keyword baseline mainly fails when a pathology word appears inside a negated or uncertain phrase. Typical errors are false positives such as predicting `positive` for `No pneumothorax`.

The rule-based method handles many of these cases but still fails on wording outside its cue list or on findings expressed indirectly.

The local Hugging Face model makes a different type of error: it often avoids assigning explicit `negative` labels and instead predicts `not_mentioned`. This is visible in reports where the absence of a finding is clearly stated.

---

## 8. Limitations

- The dataset is tiny and synthetic.
- The absolute numbers are illustrative only.
- The Hugging Face run uses a small instruction model.
- No clinical validation was performed.
- No real clinical data is included.
- The project uses prompt-based inference only, not fine-tuning.

---

## 9. Future work

Useful next steps:

- test a stronger local instruction model,
- add few-shot examples to the prompt,
- compare against clinical NLP labelers on approved datasets,
- measure how labeling errors affect downstream chest X-ray classifiers,
- explore fine-tuning only with a properly annotated dataset.
