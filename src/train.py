import os, random, argparse, yaml, numpy as np, pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ---------------------------
# Dataset
# ---------------------------
class CommandDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, mode: str):
        self.texts = []
        for _, row in df.iterrows():
            text = str(row["command_text"])
            if mode in ["context", "full"]:
                ctx = str(row.get("context_text", "")).strip()
                if ctx:
                    text += " " + ctx
            self.texts.append(text)

        # Labling
        self.labels = df["label_id"].astype(int).values
        self.priority = df["priority_score"].astype(float).values

        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.mode = mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "priority": torch.tensor(self.priority[idx], dtype=torch.float),
        }

# ---------------------------
# Train/Eval for one (model, mode)
# ---------------------------
def train_one_model(df_train, df_test, model_name, mode, cfg):
    device = cfg["general"]["device"]
    set_seed(int(cfg["general"]["seed_base"]))

    pretrained_name = cfg["models"][model_name]["pretrained"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    # GPT-2 needs pad_token
    if "gpt2" in pretrained_name and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=int(cfg["general"]["num_labels"])
    ).to(device)

    # اگر GPT-2 است، pad_token_id را ست کن تا warning نده
    if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_ds = CommandDataset(df_train, tokenizer, cfg["general"]["max_len"], mode)
    test_ds  = CommandDataset(df_test,  tokenizer, cfg["general"]["max_len"], mode)

    pin = (device == "cuda")
    train_dl = DataLoader(train_ds, batch_size=int(cfg["general"]["batch_size"]), shuffle=True,  pin_memory=pin)
    test_dl  = DataLoader(test_ds,  batch_size=int(cfg["general"]["batch_size"]), shuffle=False, pin_memory=pin)

    optimizer = AdamW(model.parameters(), lr=float(cfg["general"]["lr"]))
    total_steps = len(train_dl) * int(cfg["general"]["epochs"])
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    criterion_none = torch.nn.CrossEntropyLoss(reduction="none")

    # ----- TRAIN
    for epoch in range(int(cfg["general"]["epochs"])):
        model.train()
        loop = tqdm(train_dl, desc=f"{model_name}-{mode} | epoch {epoch+1}")
        for batch in loop:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            pri            = batch["priority"].to(device)  # [B], float

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            if mode in ["prioritize", "full"]:
                per_sample_loss = criterion_none(outputs.logits, labels)     # [B]
                loss = (per_sample_loss * (0.5 + pri)).mean()                # scalar
            else:
                loss = outputs.loss                                          # mean CE

            loss.backward()
            optimizer.step()
            scheduler.step()
            loop.set_postfix(loss=float(loss))

    # ----- EVAL
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
    return acc, prec, rec, f1

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print(">>> Running:", __file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train_path", default="data/train_mimic2000_hist.csv")
    parser.add_argument("--test_path",  default="data/test_mimic40_hist.csv")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Normalize numeric types
    g = cfg.get("general", {})
    g["lr"]         = float(g.get("lr", 2e-5))
    g["epochs"]     = int(g.get("epochs", 3))
    g["batch_size"] = int(g.get("batch_size", 8))
    g["num_labels"] = int(g.get("num_labels", 4))
    g["max_len"]    = int(g.get("max_len", 128))
    cfg["general"]  = g

    # Device selection
    if cfg["general"].get("device", "auto") == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["general"]["device"]
    cfg["general"]["device"] = device
    print(f">>> Device selected: {device}")
    # Load datasets
    df_train = pd.read_csv(args.train_path)
    df_test  = pd.read_csv(args.test_path)
    label_order = ["ROUTING","PARKING","TRAFFIC_MGMT","ENTERTAINMENT"]
    label2id = {lab:i for i,lab in enumerate(label_order)}
    df_train["label_id"] = df_train["target_label"].map(label2id)
    df_test["label_id"]  = df_test["target_label"].map(label2id)
    if df_train["label_id"].isna().any() or df_test["label_id"].isna().any():
        unknown = set(df_train.loc[df_train["label_id"].isna(),"target_label"].unique().tolist()
                      + df_test.loc[df_test["label_id"].isna(),"target_label"].unique().tolist())
        raise ValueError(f"Unknown target_label(s) found: {unknown}. Expected one of {label_order}")

    # Train/Eval
    os.makedirs(cfg["general"]["save_dir"], exist_ok=True)
    results = []
    for model_name in cfg["models"].keys():
        for mode in cfg["modes"]:
            print(f"\n=== Training {model_name} - {mode} ===")
            acc, prec, rec, f1 = train_one_model(df_train, df_test, model_name, mode, cfg)
            row = {
                "model": model_name,
                "mode": mode,
                "accuracy": round(acc*100, 2),
                "precision": round(prec*100, 2),
                "recall": round(rec*100, 2),
                "f1": round(f1*100, 2),
            }
            results.append(row)
            print(f"✅ {model_name}-{mode} | Acc={row['accuracy']:.2f}  F1={row['f1']:.2f}")

    out_csv = os.path.join(cfg["general"]["save_dir"], "results_table7_reproduced.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n✅ All results saved to {out_csv}")

