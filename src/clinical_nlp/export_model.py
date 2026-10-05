"""Train the TF-IDF + logistic regression baseline and package it for the Streamlit demo.

Run from the project root:  python -m clinical_nlp.export_model
Writes demo/model.joblib and demo/requirements.txt. The requirements pin the exact scikit-learn
version used here, because a saved model must be loaded with the same version it was saved with.
"""
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import accuracy_score, f1_score

from clinical_nlp.baseline import build_model
from clinical_nlp.data import load_dataset

DEMO = Path("demo")


def main():
    DEMO.mkdir(exist_ok=True)
    train_df, test_df = load_dataset()
    model = build_model()
    model.fit(train_df["text"], train_df["label"])
    preds = model.predict(test_df["text"])
    print(f"Test accuracy {accuracy_score(test_df['label'], preds):.3f} | "
          f"macro-F1 {f1_score(test_df['label'], preds, average='macro'):.3f}")

    joblib.dump(model, DEMO / "model.joblib", compress=3)
    size_mb = (DEMO / "model.joblib").stat().st_size / 1e6
    print(f"Saved {DEMO / 'model.joblib'} ({size_mb:.1f} MB)")

    (DEMO / "requirements.txt").write_text(
        f"streamlit\nscikit-learn=={sklearn.__version__}\njoblib\nnumpy\npandas\n"
    )
    print(f"Wrote {DEMO / 'requirements.txt'} (scikit-learn=={sklearn.__version__})")


if __name__ == "__main__":
    main()
