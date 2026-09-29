# Clinical Note Classification: Classical ML vs. BERT vs. LoRA-Tuned LLM

> One-sentence summary of what this project shows and the headline result (fill in at the end).

## Problem
Why classifying clinical notes matters (routing, cohort identification, phenotyping) in 3-4 sentences.

## Data
- Source: MTSamples medical transcriptions (public). Later version: MIMIC-IV notes (credentialed access; not redistributed here).
- Task: predict note specialty (top-N specialties).
- Size after cleaning:1,898 notes, 8 classes.

## Methods
1. Baseline: TF-IDF + logistic regression
2. Encoder: fine-tuned BERT-class model
3. LLM: LoRA fine-tune of a small open LLM

## Results
| Model | Macro-F1 | Accuracy | Training time | Notes |
|---|---|---|---|---|
| | TF-IDF + LR | 0.820 | 0.890 | under 1 min | strong baseline; weakest on Neurology and Cardio/Pulm |
| BERT-class | | | | |
| LoRA LLM | | | | |

Figure: confusion matrix of the best model.

## Error analysis
What the best model gets wrong and why.

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # makes the clinical_nlp package importable
pytest
python -m clinical_nlp.data       # data pipeline
python -m clinical_nlp.baseline   # train + evaluate baseline
```

## Serving
FastAPI endpoint + Docker (added later).

## Limitations
Dataset size, label noise, transcribed notes vs. real EHR, no patient-level split available.
