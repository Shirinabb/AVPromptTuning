
import argparse, json, os, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def load_jsonl(p):
    data = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="augmented.jsonl")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    records = load_jsonl(args.data)
    # Simple split
    idx = list(range(len(records)))
    random.shuffle(idx)
    n = len(idx); tr = int(0.7*n); va = int(0.85*n)
    train = [records[i] for i in idx[:tr]]
    val = [records[i] for i in idx[tr:va]]
    test = [records[i] for i in idx[va:]]

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def to_ds(split):
        return Dataset.from_dict({
            "text": [r["command"] for r in split],
            "label": [int(r["safety"]) for r in split]
        })

    ds_tr = to_ds(train)
    ds_va = to_ds(val)
    ds_te = to_ds(test)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=args.max_len)

    ds_tr = ds_tr.map(tok, batched=True)
    ds_va = ds_va.map(tok, batched=True)
    ds_te = ds_te.map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=2e-5,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate(ds_va)
    print("VAL:", metrics)

    # Save basic predictions for evaluation
    preds = trainer.predict(ds_va)
    y_true = preds.label_ids.tolist()
    y_pred = preds.predictions.argmax(-1).tolist()
    with open(out_dir / "gold_val.json", "w") as f:
        json.dump(y_true, f)
    with open(out_dir / "preds_val.json", "w") as f:
        json.dump(y_pred, f)

    trainer.save_model(str(out_dir))

if __name__ == "__main__":
    main()
