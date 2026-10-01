"""Step 3: fine-tune a BERT-class model to classify clinical notes by specialty.

Default model: Bio_ClinicalBERT, a BERT model further pretrained on clinical notes,
so it already "speaks" clinical language before we fine-tune it on our task.

Run:  python -m clinical_nlp.bert
Faster/smaller option:  python -m clinical_nlp.bert --model distilbert-base-uncased
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from clinical_nlp.data import load_dataset

RESULTS = Path("results")


def set_seed(seed: int) -> None:
    """Make runs repeatable: same seed -> same shuffling and initial weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Use an NVIDIA GPU if present, else the Apple-silicon GPU (MPS), else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NotesDataset(Dataset):
    """Turns note text into token IDs the model can read."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(
            list(texts),
            truncation=True,          # cut notes longer than max_len tokens
            max_length=max_len,
            padding="max_length",     # pad short notes so every input is the same length
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],  # 1 = real token, 0 = padding
            "labels": self.labels[i],
        }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()  # turn off dropout for evaluation
    preds = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**inputs).logits             # one score per specialty
        preds.extend(logits.argmax(dim=-1).cpu().tolist())  # pick the highest score
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | Model: {args.model}")

    # Same data and same split as the baseline -> a fair comparison
    train_df, test_df = load_dataset()
    labels = sorted(train_df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id
    ).to(device)

    train_ds = NotesDataset(train_df["text"], train_df["label"].map(label2id).tolist(),
                            tokenizer, args.max_len)
    test_ds = NotesDataset(test_df["text"], test_df["label"].map(label2id).tolist(),
                           tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2)

    # AdamW + small learning rate + warmup: the standard recipe for fine-tuning BERT
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss          # how wrong the predictions were
            loss.backward()                     # compute how to adjust each weight
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent huge updates
            optimizer.step()                    # adjust the weights
            scheduler.step()                    # adjust the learning rate
            optimizer.zero_grad()               # reset for the next batch
            running_loss += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                      f"avg loss {running_loss / step:.3f}")
        print(f"epoch {epoch} done | avg loss {running_loss / len(train_loader):.3f}")
    minutes = (time.time() - start) / 60

    preds = [id2label[i] for i in predict(model, test_loader, device)]
    metrics = {
        "model": args.model,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "accuracy": round(accuracy_score(test_df["label"], preds), 4),
        "macro_f1": round(f1_score(test_df["label"], preds, average="macro"), 4),
        "train_minutes": round(minutes, 1),
        "device": str(device),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(test_df["label"], preds))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "bert_metrics.json").write_text(json.dumps(metrics, indent=2))
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        test_df["label"], preds, ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"{args.model.split('/')[-1]}: confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS / "bert_confusion_matrix.png", dpi=150)
    print("Saved results to results/")


if __name__ == "__main__":
    main()
