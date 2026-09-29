# Clinical Note Classification: Classical ML vs. BERT vs. LoRA-Tuned LLM

> **Work in progress:** a TF-IDF + logistic regression baseline reaches 0.82 macro-F1 across 8 specialties. BERT and LoRA-tuned LLM comparisons are next.

## Problem
Much of the clinically important information in health records, such as symptoms, severity, and clinical reasoning, lives in free-text notes rather than structured fields, and reviewing notes manually does not scale. Automatically classifying notes is a building block for applications like cohort identification (phenotyping), clinical trial screening, and document routing. This project uses medical specialty classification on public transcription data as a testbed to compare three approaches: TF-IDF with logistic regression, a fine-tuned BERT-class model, and a LoRA-tuned LLM, asking whether larger models deliver enough improvement to justify their cost. The pipeline is designed to transfer to real EHR tasks, such as identifying depression from discharge summaries.

## Data
- **Source:** MTSamples medical transcriptions (public, via Kaggle). The dataset is not redistributed in this repository.
- **Task:** predict a note's medical specialty (8 most common specialties).
- **Cleaning:** removed missing and duplicate notes, very short notes, and labels that describe note types rather than specialties (e.g., "Discharge Summary", "SOAP / Chart / Progress Notes").
- **Size after cleaning:** 1,898 notes, 8 classes; stratified 80/20 train/test split (1,518 / 380).
- **Class imbalance:** Surgery makes up about half of the notes, so macro-F1 is the headline metric.

## Methods
1. **Baseline:** TF-IDF (unigrams + bigrams) + logistic regression with balanced class weights
2. **Encoder:** fine-tuned BERT-class model *(in progress)*
3. **LLM:** LoRA fine-tune of a small open LLM *(in progress)*

## Results
| Model | Macro-F1 | Accuracy | Training time | Notes |
|---|---|---|---|---|
| TF-IDF + LR | 0.820 | 0.890 | under 1 min | Strong baseline; weakest on Neurology and Cardio/Pulm |
| BERT-class | | | | in progress |
| LoRA LLM | | | | in progress |

![Baseline confusion matrix](results/baseline_confusion_matrix.png)

## Error analysis (baseline)
- **Strongest classes:** Surgery (F1 0.97) and Psychiatry/Psychology (F1 0.95), both with distinctive vocabulary.
- **Weakest classes:** Neurology (F1 0.62) and Cardiovascular/Pulmonary (F1 0.63). These have few test examples (15 and 21), so their scores are also the least stable.
- **Urology:** precision 1.00 but recall 0.74. The model correctly identified 23 of 31 urology notes; the 8 missed notes were spread across Surgery (3), General Medicine (2), Neurology (2), and Radiology (1). Specialties overlap in real documentation (e.g., urologic procedures are also surgeries), which a single-label setup cannot capture.
- **Next step:** read the misclassified notes to identify patterns, and use cross-validation for more stable estimates on small classes.

## How to run
```bash
conda create -n clinical-nlp python=3.12 -y
conda activate clinical-nlp
pip install -r requirements.txt
pip install -e .
# download mtsamples.csv from Kaggle into data/
pytest
python -m clinical_nlp.data       # data pipeline
python -m clinical_nlp.baseline   # train + evaluate baseline
```

## Serving
FastAPI endpoint + Docker *(planned)*.

## Limitations
- Transcribed sample notes, not real EHR notes; real-world performance would likely be lower.
- Single-label task even though specialties overlap.
- Small test sets for minority classes make their scores noisy.
- No patient identifiers, so a patient-level split is not possible here.
