import os, json, argparse, random, time
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    GPT2ForSequenceClassification,
    get_linear_schedule_with_decay
)

# مدل‌های سفارشی مقاله
from models.salmon import Salmon, SalmonConfig
from models.salmonn import SalmonN, SalmonNConfig

# ====== Utils: ثبات‌پذیری ======
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ====== دیتاست JSONL متنی ======
class JsonlTextDataset(Dataset):
    """
    انتظار دارد هر خط JSON شامل کلیدهای:
    - text: str
    - label: str (یا int)
    - context_tags: List[str] (اختیاری)
    """
    def __init__(self, path: str, label2id: Dict[str, int] = None):
        self.items = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append(obj)
        # نگاشت برچسب‌ها
        if label2id is None:
            labels = sorted(list({it["label"] for it in self.items}))
            self.label2id = {lbl: i for i, lbl in enumerate(labels)}
        else:
            self.label2id = label2id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x = self.items[idx]
        text = x["text"]
        label = x["label"]
        label_id = self.label2id[label] if isinstance(label, str) else int(label)
        ctx_tags = x.get("context_tags", [])
        return {
            "text": text,
            "label": label_id,
            "context_tags": ctx_tags
        }

# ====== واژگان کانتکست ======
def load_context_vocab(path: str) -> Dict[str, int]:
    if not path or (not os.path.exists(path)):
        return {"_PAD": 0}
    with open(path, "r", encoding="utf8") as f:
        tags = json.load(f)  # ["RAIN","FOG",...]
    return {t: i for i, t in enumerate(tags)}

# ====== Collate: توکن‌سازی + نگاشت تگ‌ها ======
def build_collate(tokenizer, ctx_vocab: Dict[str, int], max_len: int = 128):
    def _collate(batch: List[Dict[str, Any]]):
        texts = [b["text"] for b in batch]
        toks = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        # context_tags → indices (pad = -1)
        max_tags = max(len(b.get("context_tags", [])) for b in batch) if batch else 1
        ctx_ids = []
        for b in batch:
            ids = [ctx_vocab.get(t, 0) for t in b.get("context_tags", [])]
            ids = ids + [-1] * (max_tags - len(ids))
            ctx_ids.append(ids)
        ctx_ids = torch.tensor(ctx_ids, dtype=torch.long)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "context_tag_ids": ctx_ids,
            "labels": labels,
            # برای وزن‌دهی مبتنی بر برچسب (در خود batch نگه می‌داریم)
            "_raw_ctx_tags": [b.get("context_tags", []) for b in batch]
        }
    return _collate

# ====== ساخت مدل ======
def build_model(args, num_labels: int, ctx_vocab: Dict[str, int]):
    if args.model == "salmon":
        cfg = SalmonConfig(
            pretrained_name=args.pretrained_name,
            num_labels=num_labels,
            ctx_dim=args.ctx_dim,
            dropout=args.dropout,
            ctx_vocab=ctx_vocab,
            use_context=not args.ctx_off
        )
        model = Salmon(cfg)
        tok = model.tokenizer

    elif args.model == "salmonn":
        cfg = SalmonNConfig(
            text_backbone=args.pretrained_name,
            num_labels=num_labels,
            ctx_dim=args.ctx_dim,
            dropout=args.dropout,
            ctx_vocab=ctx_vocab,
            soft_prompt_len=args.soft_prompt_len,
            use_context=not args.ctx_off,
            use_soft_prompt=not args.no_soft
        )
        model = SalmonN(cfg)
        tok = model.tokenizer

    elif args.model == "bert":
        tok = AutoTokenizer.from_pretrained(args.pretrained_name)
        model = BertForSequenceClassification.from_pretrained(
            args.pretrained_name, num_labels=num_labels
        )

    elif args.model == "gpt2":
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = GPT2ForSequenceClassification.from_pretrained(
            "gpt2", num_labels=num_labels
        )
        model.config.pad_token_id = tok.pad_token_id
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model, tok

# ====== وزن‌دهی α,β برای ایمنی/فوریت ======
def make_sample_weights(labels: torch.Tensor,
                        batch_raw_ctx_tags: List[List[str]],
                        safety_class_id: int,
                        prior_on: bool,
                        alpha: float,
                        beta: float,
                        urgency_tags: List[str]) -> torch.Tensor:
    B = labels.size(0)
    w = torch.ones(B, dtype=torch.float, device=labels.device)
    if not prior_on:
        return w
    for i in range(B):
        # اگر کلاس Safety باشد → α
        if labels[i].item() == safety_class_id:
            w[i] += alpha
        # اگر تگ فوریت داشته باشد → β
        tags = set(batch_raw_ctx_tags[i] or [])
        if any(t in tags for t in urgency_tags):
            w[i] += beta
    return w

# ====== متریک‌ها ======
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# ====== ذخیره نتایج ======
def save_row_csv(path: str, row: Dict[str, Any]):
    import pandas as pd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        import pandas as pd
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)

# ====== حلقه آموزش ======
def train_one_epoch(model, loader, optimizer, scheduler, device, args, safety_class_id, urgency_tags):
    model.train()
    tr_loss, n_steps = 0.0, 0
    for batch in loader:
        batch_dev = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        # فراخوانی مدل با سوییچ‌ها (برای مدل‌های سفارشی)
        fwd_kwargs = {}
        if args.model in ("salmon", "salmonn"):
            fwd_kwargs["use_context"] = (not args.ctx_off)
        if args.model == "salmonn":
            fwd_kwargs["use_soft_prompt"] = (not args.no_soft)

        out = model(
            input_ids=batch_dev["input_ids"],
            attention_mask=batch_dev["attention_mask"],
            context_tag_ids=batch_dev["context_tag_ids"],
            labels=batch_dev["labels"],
            **fwd_kwargs
        )

        logits = out["logits"]
        labels = batch_dev["labels"]

        # وزن‌دهی نمونه‌ها
        weights = make_sample_weights(
            labels=labels,
            batch_raw_ctx_tags=batch["_raw_ctx_tags"],
            safety_class_id=safety_class_id,
            prior_on=args.prior_on,
            alpha=args.alpha,
            beta=args.beta,
            urgency_tags=urgency_tags
        )

        loss_per = nn.functional.cross_entropy(logits, labels, reduction="none")
        loss = (loss_per * weights).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        tr_loss += loss.item()
        n_steps += 1

    return tr_loss / max(1, n_steps)

@torch.no_grad()
def evaluate(model, loader, device, args, safety_class_id):
    model.eval()
    y_true, y_pred = [], []
    aux_gate_means = []

    for batch in loader:
        batch_dev = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        fwd_kwargs = {}
        if args.model in ("salmon", "salmonn"):
            fwd_kwargs["use_context"] = (not args.ctx_off)
        if args.model == "salmonn":
            fwd_kwargs["use_soft_prompt"] = (not args.no_soft)

        out = model(
            input_ids=batch_dev["input_ids"],
            attention_mask=batch_dev["attention_mask"],
            context_tag_ids=batch_dev["context_tag_ids"],
            labels=batch_dev["labels"],
            **fwd_kwargs
        )
        logits = out["logits"]
        preds = logits.argmax(-1)

        y_true.extend(batch_dev["labels"].cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        # جمع‌آوری aux برای SALMONN
        if isinstance(out, dict) and "aux" in out and "gate_mean" in out["aux"]:
            aux_gate_means.append(out["aux"]["gate_mean"])

    # overall
    overall = compute_metrics(y_true, y_pred)

    # safety vs non-safety split
    s_idx = [i for i, l in enumerate(y_true) if l == safety_class_id]
    ns_idx = [i for i, l in enumerate(y_true) if l != safety_class_id]

    def _submetrics(idxs):
        if len(idxs) == 0:
            return {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        return compute_metrics(yt, yp)

    safety_m = _submetrics(s_idx)
    nonsafety_m = _submetrics(ns_idx)

    extras = {}
    if aux_gate_means:
        extras["avg_gate_mean"] = float(np.mean(aux_gate_means))

    return overall, safety_m, nonsafety_m, extras

# ====== آرگومان‌ها ======
def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="salmon", choices=["salmon","salmonn","bert","gpt2"])
    ap.add_argument("--pretrained_name", type=str, default="bert-base-uncased")

    ap.add_argument("--train_path", type=str, default="Data/samples/mini_train.jsonl")
    ap.add_argument("--val_path", type=str, default="Data/samples/mini_val.jsonl")
    ap.add_argument("--ctx_vocab_path", type=str, default="Data/context_vocab.json")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)

    # مقاله: ابعاد و dropout
    ap.add_argument("--ctx_dim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--soft_prompt_len", type=int, default=16)

    # سوییچ‌های ادعا
    ap.add_argument("--ctx_off", action="store_true", help="خاموش کردن استفاده از context tags در مدل‌های SALMON/SALMONN")
    ap.add_argument("--no_soft", action="store_true", help="خاموش کردن soft-prompt در SALMONN (آبلیشن)")
    ap.add_argument("--prior_on", action="store_true", help="فعال‌سازی وزن‌دهی loss برای safety/urgency")

    # ضرایب α و β
    ap.add_argument("--alpha", type=float, default=0.7, help="وزن ایمنی")
    ap.add_argument("--beta", type=float, default=0.3, help="وزن فوریت")

    # تعریف برچسب و تگ‌های فوریت
    ap.add_argument("--safety_label", type=str, default="Safety")
    ap.add_argument("--urgency_tags", type=str, default="URGENCY_CRITICAL,URGENCY_HIGH")

    # خروجی
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--save_csv", type=str, default=None,
                    help="نام فایل CSV خروجی؛ اگر خالی باشد بر اساس تنظیمات ساخته می‌شود.")

    return ap.parse_args()

def main():
    args = build_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    # لود داده
    train_ds = JsonlTextDataset(args.train_path)
    val_ds   = JsonlTextDataset(args.val_path, label2id=train_ds.label2id)

    num_labels = len(train_ds.label2id)
    id2label = {v:k for k,v in train_ds.label2id.items()}

    # شناسایی safety_class_id
    if args.safety_label in train_ds.label2id:
        safety_class_id = train_ds.label2id[args.safety_label]
    else:
        # اگر نام برچسب Safety متفاوت بود، کلاس 0 را ایمن فرض می‌کنیم
        safety_class_id = 0

    # لود واژگان کانتکست
    ctx_vocab = load_context_vocab(args.ctx_vocab_path)

    # ساخت مدل و توکنایزر
    model, tokenizer = build_model(args, num_labels=num_labels, ctx_vocab=ctx_vocab)
    model.to(device)

    collate = build_collate(tokenizer, ctx_vocab, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, collate_fn=collate)

    # بهینه‌ساز و شِدولر
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_decay(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    urgency_tags = [t.strip() for t in args.urgency_tags.split(",") if t.strip()]

    # آموزش
    best_f1 = -1.0
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, args, safety_class_id, urgency_tags)
        overall, s_m, ns_m, extras = evaluate(model, val_loader, device, args, safety_class_id)

        print(f"[Epoch {ep}] train_loss={tr_loss:.4f}  "
              f"val_f1={overall['f1']:.4f}  val_acc={overall['accuracy']:.4f}")

        # ذخیره بهترین
        if overall["f1"] > best_f1:
            best_f1 = overall["f1"]

        # ذخیره ردیف نتایج هر epoch
        row = {
            "epoch": ep,
            "model": args.model,
            "ctx_off": args.ctx_off,
            "no_soft": args.no_soft,
            "prior_on": args.prior_on,
            "alpha": args.alpha,
            "beta": args.beta,
            "seed": args.seed,
            "overall_accuracy": overall["accuracy"],
            "overall_precision": overall["precision"],
            "overall_recall": overall["recall"],
            "overall_f1": overall["f1"],
            "safety_accuracy": s_m["accuracy"],
            "safety_precision": s_m["precision"],
            "safety_recall": s_m["recall"],
            "safety_f1": s_m["f1"],
            "nonsafety_accuracy": ns_m["accuracy"],
            "nonsafety_precision": ns_m["precision"],
            "nonsafety_recall": ns_m["recall"],
            "nonsafety_f1": ns_m["f1"],
        }
        row.update(extras)

        # نام فایل خروجی
        if args.save_csv:
            out_csv = os.path.join(args.results_dir, args.save_csv)
        else:
            tag = []
            if args.ctx_off: tag.append("ctxOff")
            if args.no_soft and args.model == "salmonn": tag.append("noSoft")
            if args.prior_on: tag.append("prior")
            tag = "_".join(tag) if tag else "base"
            out_csv = os.path.join(args.results_dir, f"{args.model}_{tag}.csv")

        save_row_csv(out_csv, row)

    print("Done.")

if __name__ == "__main__":
    main()
