# Model Card: Clinical Note Specialty Classifier

*Format follows Mitchell et al., "Model Cards for Model Reporting" (2019).*

## Model details
- **Deployed model:** TF-IDF features with logistic regression, serving the
  [live demo](https://clinical-note-classifier.streamlit.app), which shows the terms behind each prediction.
- **Benchmarked alternatives:** Bio_ClinicalBERT (fine-tuned, 256 and 512 tokens, with and without class weights) and
  Qwen2.5-0.5B with a LoRA adapter (rank 16, attention projections, about 0.44% of parameters trained), published at
  [Hugging Face](https://huggingface.co/oshinmiranda26/qwen2.5-0.5b-lora-clinical-note-specialty).
- **Developer:** Oshin Miranda. **License:** MIT.

## Intended use
- **Intended:** demonstrating a controlled comparison of classical, fine-tuned transformer, and parameter-efficient
  LLM approaches to clinical text classification; teaching.
- **Out of scope:** routing, coding, billing, or any decision about real patients or real clinical documents.

## Data
MTSamples: publicly available transcribed medical sample reports (via Kaggle), labeled by medical specialty. Not
included in the repository. Sample reports are cleaner and more uniform than real clinical documentation.

## Performance (held-out test set)
| Model | Macro-F1 | Accuracy |
|---|---|---|
| **TF-IDF + logistic regression (deployed)** | **0.820** | **0.890** |
| Bio_ClinicalBERT, 256 tokens | 0.707 | 0.855 |
| Bio_ClinicalBERT, 256 tokens, class weights | 0.733 | 0.842 |
| Bio_ClinicalBERT, 512 tokens, class weights | 0.794 | 0.874 |
| Qwen2.5-0.5B + LoRA, 512 tokens, class weights | 0.783 | 0.879 |

Why the simplest model is deployed: it performed best, runs on CPU, and is fully explainable.

## Ethical considerations
- Specialty predictions could misroute documents if used operationally; the model is not validated for that.
- Clinical text can contain identifiers; any real-world use would require de-identification and privacy review.

## Limitations
- No separate validation set was used for model selection; reported test results may be slightly optimistic.
- Class imbalance across specialties; performance is weakest where specialties overlap (e.g., neurology and
  psychiatry).
- Results come from one dataset and do not represent larger LLMs.
