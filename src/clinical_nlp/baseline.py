"""Step 2: the baseline model, TF-IDF features + logistic regression.

Every stronger model later in the project must beat this number to justify its cost.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without needing a screen
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.pipeline import Pipeline

from clinical_nlp.data import load_dataset

RESULTS = Path("results")


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),   # single words and two-word phrases ("chest pain")
            min_df=2,             # ignore terms seen in only one note
            max_features=50_000,
            sublinear_tf=True,    # dampen very frequent words
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",  # give rare specialties fair weight
        )),
    ])


def top_terms(model: Pipeline, k: int = 8) -> dict:
    """The words that push the model most strongly toward each specialty (interpretability)."""
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    clf = model.named_steps["clf"]
    return {label: [vocab[i] for i in coefs.argsort()[-k:][::-1]]
            for label, coefs in zip(clf.classes_, clf.coef_)}


def main():
    RESULTS.mkdir(exist_ok=True)
    train_df, test_df = load_dataset()

    model = build_model()
    model.fit(train_df["text"], train_df["label"])      # learn only from training data
    preds = model.predict(test_df["text"])               # evaluate on unseen notes

    metrics = {
        "model": "tfidf_logreg",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "accuracy": round(accuracy_score(test_df["label"], preds), 4),
        "macro_f1": round(f1_score(test_df["label"], preds, average="macro"), 4),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(test_df["label"], preds))

    (RESULTS / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS / "baseline_top_terms.json").write_text(json.dumps(top_terms(model), indent=2))

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        test_df["label"], preds, ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title("TF-IDF + Logistic Regression: confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS / "baseline_confusion_matrix.png", dpi=150)
    print("Saved results to results/")


if __name__ == "__main__":
    main()
