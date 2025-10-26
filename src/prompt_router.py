import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["ROUTING","PARKING","TRAFFIC_MGMT","ENTERTAINMENT"]

class Router:
    def __init__(self, ckpt_dir, device="cpu"):
        self.tok = AutoTokenizer.from_pretrained(ckpt_dir)
        self.m = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
        self.device = device

    def predict(self, command_text, context_text=""):
        text = f"{command_text} {context_text}".strip()
        enc = self.tok(text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.m(**enc).logits
        y = int(torch.argmax(logits, dim=1).cpu().item())
        return LABELS[y]
