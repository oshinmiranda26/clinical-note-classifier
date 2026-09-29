import pandas as pd
from clinical_nlp.data import clean_notes, keep_top_labels, split


def make_df():
    return pd.DataFrame({
        "text": ["a" * 60, "a" * 60, None, "short", "b" * 60, "c" * 60,
                 "d" * 60, "e" * 60, "f" * 60, "g" * 60],
        "label": ["Psych", "Psych", "Cardio", "Cardio", "Cardio", "Psych",
                  "Cardio", "Psych", "Discharge Summary", "Cardio"],
    })


def test_clean_removes_missing_duplicates_short_and_note_types():
    out = clean_notes(make_df())
    assert out["text"].notna().all()
    assert out["text"].duplicated().sum() == 0
    assert (out["text"].str.len() >= 50).all()
    assert "Discharge Summary" not in set(out["label"])


def test_keep_top_labels():
    out = keep_top_labels(clean_notes(make_df()), n=1)
    assert out["label"].nunique() == 1


def test_split_preserves_all_rows_and_no_overlap():
    df = clean_notes(make_df())
    train, test = split(df, test_size=0.5)
    assert len(train) + len(test) == len(df)
    assert set(train["text"]).isdisjoint(set(test["text"]))
