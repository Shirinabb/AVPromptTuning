import os, random, argparse, yaml, numpy as np, pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ---------------------------
# Dataset
# ---------------------------
class CommandDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, mode):
        self.texts = []
        for _, row in df.iterrows():
            text = row["command_text"]
            if mode in ["context", "full"]:
                text += " " + str(row["context_text"])
            self.texts.append(text)
        self.labels = df["target_label"].astype("category").cat.codes.values
        self.priority = df["priority_score"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx],
                             padding="max_length",
                             truncation=True,
                             max_length=self.max_len,
                             return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "priority": torch.tensor(self.priority[idx], dtype=torch.float)
        }

# ---------------------------
# Training Loop
# ---------------------------
def train_one_model(args, df_train, df_test, model_name, mode, cfg):
    set_seed(cfg["general"]["seed_base"])
    device = cfg["general"]["device"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["models"][model_name]["pretrained"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["models"][model_name]["pretrained"],
        num_labels=cfg["general"]["num_labels"]
    ).to(device)

    train_ds = CommandDataset(df_train, tokenizer, cfg["general"]["max_len"], mode)
    test_ds = CommandDataset(df_test, tokenizer, cfg["general"]["max_len"], mode)
    train_dl = DataLoader(train_ds, batch_size=cfg["general"]["batch_size"], shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=cfg["general"]["batch_size"])

    optimizer = AdamW(model.parameters(), lr=cfg["general"]["lr"])
    total_steps = len(train_dl) * cfg["general"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    for epoch in range(cfg["general"]["epochs"]):
        model.train()
        loop = tqdm(train_dl, desc=f"{model_name}-{mode} Epoch {epoch+1}")
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pri = batch["priority"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if mode in ["prioritize", "full"]:
                with torch.no_grad():
                    # weight loss per-sample
                    per_sample_loss = torch.nn.functional.cross_entropy(
                        outputs.logits, labels, reduction="none")
                loss = (per_sample_loss * (0.5 + pri)).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            loop.set_postfix(loss=float(loss))

    # ---------------------------
    # Evaluation
    # ---------------------------
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train_path", default="data/train_mimic2000_hist.csv")
    parser.add_argument("--test_path", default="data/test_mimic40_hist.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df_train = pd.read_csv(args.train_path)
    df_test = pd.read_csv(args.test_path)

    results = []
    for model_name in cfg["models"].keys():
        for mode in cfg["modes"]:
            acc, prec, rec, f1 = train_one_model(args, df_train, df_test, model_name, mode, cfg)
            results.append({
                "model": model_name,
                "mode": mode,
                "accuracy": round(acc*100,2),
                "precision": round(prec*100,2),
                "recall": round(rec*100,2),
                "f1": round(f1*100,2)
            })
            print(f"✅ {model_name}-{mode}: Acc={acc*100:.2f}  F1={f1*100:.2f}")

    os.makedirs(cfg["general"]["save_dir"], exist_ok=True)
    out_csv = os.path.join(cfg["general"]["save_dir"], "results_table7_reproduced.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nSaved all results to {out_csv}")
