"""Step 4: LoRA fine-tuning of a small open LLM for clinical note classification.

The LLM's original weights stay frozen. LoRA adds small trainable "adapter" matrices to the
attention layers, so only a tiny fraction of parameters is trained.

Smoke test first (~2-5 min):  python -m clinical_nlp.lora --limit 64 --epochs 1 --tag lora_smoke
Full run:                     python -m clinical_nlp.lora --class-weights --tag lora_qwen
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from clinical_nlp.bert import NotesDataset, get_device, predict, set_seed
from clinical_nlp.data import load_dataset

RESULTS = Path("results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch-size x grad-accum")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="use only N training notes (smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="lora")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | Model: {args.model}")

    # Same data and split as the baseline and BERT runs -> a fair comparison
    train_df, test_df = load_dataset()
    if args.limit:
        train_df = train_df.sample(args.limit, random_state=args.seed)
    labels = sorted(test_df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id,
        dtype=torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id  # LLMs need this to find the last real token

    # LoRA: freeze the LLM, add rank-r adapters to the attention projections.
    # task_type SEQ_CLS also keeps the new classification head trainable.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    train_ds = NotesDataset(train_df["text"], train_df["label"].map(label2id).tolist(),
                            tokenizer, args.max_len)
    test_ds = NotesDataset(test_df["text"], test_df["label"].map(label2id).tolist(),
                           tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    loss_fn = None
    if args.class_weights:
        counts = train_df["label"].map(label2id).value_counts().reindex(range(len(labels)), fill_value=1).values
        weights = len(train_df) / (len(labels) * counts)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs // args.grad_accum)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            if loss_fn is None:
                loss = model(**batch).loss
            else:
                y = batch.pop("labels")
                loss = loss_fn(model(**batch).logits, y)
            (loss / args.grad_accum).backward()  # gradient accumulation: update every grad_accum batches
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            running += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} avg loss {running / step:.3f} "
                      f"| {(time.time() - start) / 60:.1f} min")
        print(f"epoch {epoch} done | avg loss {running / len(train_loader):.3f}")
    minutes = (time.time() - start) / 60

    preds = [id2label[i] for i in predict(model, test_loader, device)]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    metrics = {
        "model": args.model, "method": "LoRA", "rank": args.rank, "max_len": args.max_len,
        "epochs": args.epochs, "class_weights": args.class_weights,
        "n_train": len(train_df), "n_test": len(test_df),
        "trainable_params": trainable, "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "accuracy": round(accuracy_score(test_df["label"], preds), 4),
        "macro_f1": round(f1_score(test_df["label"], preds, average="macro"), 4),
        "train_minutes": round(minutes, 1), "device": str(device),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(test_df["label"], preds, zero_division=0))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(test_df["label"], preds, ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{args.model.split('/')[-1]} + LoRA: confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{args.tag}_confusion_matrix.png", dpi=150)
    print("Saved results to results/")


if __name__ == "__main__":
    main()
