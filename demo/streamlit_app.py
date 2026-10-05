"""Live demo: predict a clinical note's medical specialty and explain the prediction."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clinical Note Specialty Classifier", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model():
    # Load once and reuse across visitors; the file sits next to this script
    return joblib.load(Path(__file__).parent / "model.joblib")


model = load_model()
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["clf"]
vocab = vectorizer.get_feature_names_out()


def explain(text: str, top_k: int = 10):
    """Return class probabilities and the terms that pushed the model toward its top prediction."""
    X = vectorizer.transform([text])
    probs = classifier.predict_proba(X)[0]
    top = int(np.argmax(probs))
    # Contribution of each term = its TF-IDF weight in this note x the model's weight for the predicted class
    contrib = X.multiply(classifier.coef_[top]).tocsr()
    order = np.argsort(contrib.data)[::-1][:top_k]
    terms = pd.DataFrame(
        [(vocab[contrib.indices[i]], round(float(contrib.data[i]), 3)) for i in order if contrib.data[i] > 0],
        columns=["Term", "Contribution"],
    )
    probs_df = (pd.DataFrame({"Specialty": classifier.classes_, "Probability": probs})
                .sort_values("Probability", ascending=False).reset_index(drop=True))
    return probs_df, terms


EXAMPLES = {
    "Cardiology example": "Patient presents with exertional chest pain and shortness of breath. Echocardiogram "
                          "shows reduced ejection fraction. Plan: cardiac catheterization and start beta blocker.",
    "Neurology example": "Patient reports progressive numbness in both hands and intermittent headaches. MRI of "
                         "the brain ordered; nerve conduction study scheduled to evaluate for neuropathy.",
    "Urology example": "Patient with gross hematuria and urinary frequency. Cystoscopy performed under local "
                       "anesthesia; a small bladder lesion was identified and biopsied.",
}

st.title("Clinical Note Specialty Classifier")
st.markdown(
    "Paste a clinical note to see the predicted medical specialty and the words that drove the prediction. "
    "Model: TF-IDF + logistic regression (macro-F1 0.82 on held-out notes), trained on the public MTSamples "
    "transcription dataset. [Code and full evaluation on GitHub]"
    "(https://github.com/oshinmiranda26/clinical-note-classifier)"
)
st.info(
    "**Known limitation:** notes centered on imaging (e.g., an MRI order) are often classified as Radiology, "
    "reflecting overlap between Neurology and Radiology in the training labels. Confidence is also lower on short "
    "notes, since the model was trained on long transcriptions."
)
st.caption("Research demo only, not for clinical use. Do not enter real patient information.")

choice = st.selectbox("Start from an example, or write your own", ["Write your own"] + list(EXAMPLES))
note = st.text_area("Clinical note", value=EXAMPLES.get(choice, ""), height=180)

if st.button("Classify", type="primary"):
    if not note.strip():
        st.warning("Please enter a note first.")
    else:
        probs_df, terms = explain(note)
        left, right = st.columns(2)
        with left:
            st.subheader(f"Prediction: {probs_df.loc[0, 'Specialty']}")
            st.dataframe(
                probs_df.head(3),
                column_config={"Probability": st.column_config.ProgressColumn(
                    "Probability", format="%.0f%%", min_value=0, max_value=1)},
                hide_index=True, width="stretch",
            )
        with right:
            st.subheader("Top terms behind the prediction")
            st.dataframe(terms, hide_index=True, width="stretch")
