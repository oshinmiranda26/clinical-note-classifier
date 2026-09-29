"""Step 1: load, clean, and split the clinical notes.

Every line here is something you should be able to explain in an interview.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/mtsamples.csv"

# MTSamples mixes true specialties with *note types* (e.g., "Discharge Summary").
# A note type is not a specialty, so keeping these would add label noise.
NOTE_TYPE_LABELS = {
    "Consult - History and Phy.",
    "SOAP / Chart / Progress Notes",
    "Discharge Summary",
    "Emergency Room Reports",
    "Office Notes",
    "Letters",
    "IME-QME-Work Comp etc.",
}


def load_notes(path: str = RAW_PATH) -> pd.DataFrame:
    """Read the CSV and keep two columns: 'text' (the note) and 'label' (the specialty)."""
    df = pd.read_csv(path)
    df = df[["transcription", "medical_specialty"]].rename(
        columns={"transcription": "text", "medical_specialty": "label"}
    )
    df["label"] = df["label"].str.strip()  # labels in the raw file have stray spaces
    return df


def clean_notes(df: pd.DataFrame, min_chars: int = 50) -> pd.DataFrame:
    """Remove rows a model cannot learn from, or that would distort evaluation."""
    df = df.dropna(subset=["text"])                 # no text = nothing to learn
    df = df[~df["label"].isin(NOTE_TYPE_LABELS)]     # drop note types (label noise)
    df = df.drop_duplicates(subset=["text"])         # duplicates could land in train AND test
    df = df[df["text"].str.len() >= min_chars]       # near-empty notes carry no signal
    return df.reset_index(drop=True)


def keep_top_labels(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Keep the n most common specialties so every class has enough examples."""
    top = df["label"].value_counts().head(n).index
    return df[df["label"].isin(top)].reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified split: train and test keep the same class proportions."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_dataset(n_labels: int = 8):
    """One call that runs the whole data pipeline."""
    return split(keep_top_labels(clean_notes(load_notes()), n=n_labels))


if __name__ == "__main__":
    train_df, test_df = load_dataset()
    print(train_df["label"].value_counts())
    print(f"train={len(train_df)} test={len(test_df)}")
