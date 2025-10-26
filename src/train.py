import os, random, argparse, yaml, numpy as np, pandas as pd, hashlib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    os.environ["PYTHONHASHSEED"] = str(seed)

def sha(lst): return hashlib.sha256("\n".join(lst).encode("utf-8")).hexdigest()

LABELS = ["ROUTING","PARKING","TRAFFIC_MGMT","ENTERTAINMENT"]

class CommandDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, mode: str,
                 ctx_field: str | None = None, ctx_pos: str = "append"):
        texts, context_attached = [], 0
        ctx_pos = (ctx_pos or "append").lower()

        for _, row in df.iterrows():
            cmd = str(row["command_text"])
            t = cmd
            if mode in ["context", "full"]:
                if ctx_field:
                    ctx = str(row.get(ctx_field, "")).strip()
                    if not ctx:
                        alt = "context_tags" if ctx_field == "context_text" else "context_text"
                        ctx = str(row.get(alt, "")).strip()
                else:
                    ctx = str(row.get("context_text","") or row.get("context_tags","")).strip()
                if ctx:
                    t = f"[CTX] {ctx} [CMD] {cmd}" if ctx_pos=="prepend" else f"{cmd} {ctx}"
                    context_attached += 1
            texts.append(t)

        if mode in ["context", "full"]:
            print(f"[SANITY] Context attached to {context_attached}/{len(df)} samples in mode={mode}")

        self.raw_texts = texts
        enc = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=int(max_len), return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attn_mask = enc["attention_mask"]
        self.labels    = torch.tensor(df["label_id"].astype(int).values, dtype=torch.long)
        self.priority  = torch.tensor(df["priority_score"].astype(float).values, dtype=torch.float)

    def __len__(self): return self.input_ids.size(0)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "labels":         self.labels[idx],
            "priority":       self.priority[idx],
        }

def train_one_model(df_train, df_test, model_name, mode, cfg):
    device = cfg["general"]["device"]
    seed_eff = int(cfg["general"]["seed_base"]); set_seed(seed_eff)
    print(f">>> Effective seed: {seed_eff}")

    pretrained = cfg["models"][model_name]["pretrained"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    # Special tokens (برای تمایز CTX/CMD اگر prepend می‌کنی)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[CTX]", "[CMD]"]})

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained, num_labels=int(cfg["general"]["num_labels"])
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    ctx_field = cfg["general"].get("context_field", None)
    ctx_pos   = cfg["general"].get("context_position", "append")

    def build_texts(df, m):
        out = []
        for _, row in df.iterrows():
            cmd = str(row["command_text"])
            if m in ["context","full"]:
                if ctx_field:
                    ctx = str(row.get(ctx_field,"")).strip() or str(row.get("context_tags","")).strip()
                else:
                    ctx = str(row.get("context_text","") or row.get("context_tags","")).strip()
                t = f"[CTX] {ctx} [CMD] {cmd}" if (ctx and ctx_pos=="prepend") else (f"{cmd} {ctx}" if ctx else cmd)
            else:
                t = cmd
            out.append(t)
        return out

    if mode in ["context","full"]:
        same_train = (sha(build_texts(df_train, "baseline")) == sha(build_texts(df_train, mode)))
        same_test  = (sha(build_texts(df_test,  "baseline")) == sha(build_texts(df_test,  mode)))
        print(f"[HASH] train same? {same_train} | test same? {same_test}")
        if same_train or same_test:
            print("[WARN] Baseline and Context texts identical after preprocessing (context ineffective).")

    train_ds = CommandDataset(df_train, tokenizer, cfg["general"]["max_len"], mode, ctx_field=ctx_field, ctx_pos=ctx_pos)
    test_ds  = CommandDataset(df_test,  tokenizer, cfg["general"]["max_len"], mode, ctx_field=ctx_field, ctx_pos=ctx_pos)

    pin = (device == "cuda")
    train_dl = DataLoader(train_ds, batch_size=int(cfg["general"]["batch_size"]), shuffle=True,  pin_memory=pin)
    test_dl  = DataLoader(test_ds,  batch_size=int(cfg["general"]["batch_size"]), shuffle=False, pin_memory=pin)

    optimizer = AdamW(model.parameters(), lr=float(cfg["general"]["lr"]))
    total_steps = len(train_dl) * int(cfg["general"]["epochs"])
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    criterion_none = torch.nn.CrossEntropyLoss(reduction="none")
    alpha = float(cfg["general"].get("alpha_priority", 0.35))

    for epoch in range(int(cfg["general"]["epochs"])):
        model.train()
        loop = tqdm(train_dl, desc=f"{model_name}-{mode} | epoch {epoch+1}")
        for batch in loop:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            pri            = batch["priority"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if mode in ["prioritize","full"]:
                per_sample_loss = criterion_none(outputs.logits, labels)  # [B]
                pri_centered = pri - pri.mean()
                w = 1.0 + alpha * pri_centered
                w = torch.clamp(w, 0.5, 1.5)
                loss = (per_sample_loss * w).mean()
            else:
                loss = outputs.loss

            loss.backward()
            optimizer.step(); scheduler.step()
            loop.set_postfix(loss=float(loss))

    # EVAL
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_dl:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="macro")

    # Debug CSV
    dbg_path = os.path.join(cfg["general"]["save_dir"], f"debug_{model_name}_{mode}_seed{seed_eff}.csv")
    pd.DataFrame({"text": test_ds.raw_texts[:len(trues)], "true_label": trues, "pred_label": preds}).to_csv(dbg_path, index=False)
    print(f"[DEBUG] wrote per-sample predictions to {dbg_path}")

    # Save checkpoint
    ckpt_dir = os.path.join(cfg["general"]["save_dir"], f"ckpt_{model_name}_{mode}_seed{seed_eff}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"[SAVE] checkpoint -> {ckpt_dir}")

    return acc, prec, rec, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train_path", default="data/train_mimic2000_hist.csv")
    parser.add_argument("--test_path",  default="data/test_mimic40_hist.csv")
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_test",  type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    g = cfg.get("general", {})
    g["lr"]         = float(g.get("lr", 2e-5))
    g["epochs"]     = int(g.get("epochs", 3))
    g["batch_size"] = int(g.get("batch_size", 8))
    g["num_labels"] = int(g.get("num_labels", 4))
    g["max_len"]    = int(g.get("max_len", 128))
    cfg["general"]  = g

    if cfg["general"].get("device", "auto") == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["general"]["device"]
    cfg["general"]["device"] = device
    print(f">>> Device: {device}")

    df_train = pd.read_csv(args.train_path)
    df_test  = pd.read_csv(args.test_path)
    if args.limit_train > 0: df_train = df_train.head(args.limit_train)
    if args.limit_test  > 0: df_test  = df_test.head(args.limit_test)
    print(f">>> Using data: train={len(df_train)}  test={len(df_test)}")

    label2id = {lab:i for i,lab in enumerate(LABELS)}
    df_train["label_id"] = df_train["target_label"].map(label2id)
    df_test["label_id"]  = df_test["target_label"].map(label2id)
    if df_train["label_id"].isna().any() or df_test["label_id"].isna().any():
        bad = set(df_train.loc[df_train["label_id"].isna(),"target_label"]).union(
              set(df_test.loc[df_test["label_id"].isna(),"target_label"]))
        raise ValueError(f"Unknown labels: {bad}. Expected one of {LABELS}")

    os.makedirs(cfg["general"]["save_dir"], exist_ok=True)
    seeds = cfg["general"].get("repeat_seeds", [cfg["general"]["seed_base"]])
    results = []
    for model_name in cfg["models"].keys():
        for mode in cfg["modes"]:
            accs, precs, recs, f1s = [], [], [], []
            for s in seeds:
                cfg["general"]["seed_base"] = int(s)
                acc, prec, rec, f1 = train_one_model(df_train, df_test, model_name, mode, cfg)
                accs.append(acc*100); precs.append(prec*100); recs.append(rec*100); f1s.append(f1*100)
            row = {
                "model": model_name, "mode": mode,
                "accuracy_mean": round(float(np.mean(accs)), 2),
                "accuracy_std":  round(float(np.std(accs)),  2),
                "precision_mean":round(float(np.mean(precs)),2),
                "recall_mean":   round(float(np.mean(recs)), 2),
                "f1_mean":       round(float(np.mean(f1s)),  2),
                "f1_std":        round(float(np.std(f1s)),   2),
            }
            results.append(row)
            print(f"✅ {model_name}-{mode} | Acc={row['accuracy_mean']:.2f}±{row['accuracy_std']:.2f}  "
                  f"F1={row['f1_mean']:.2f}±{row['f1_std']:.2f}")

    out_csv = os.path.join(cfg["general"]["save_dir"], "results_table7_reproduced.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n✅ All results saved to {out_csv}")
