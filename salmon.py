# src/models/salmon.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

@dataclass
class SalmonConfig:
    pretrained_name: str = "bert-base-uncased"
    num_labels: int = 4
    ctx_dim: int = 16
    dropout: float = 0.1
    ctx_vocab: Dict[str, int] = None  # {"RAIN":0, ...}
    use_context: bool = True          

class ContextTagEmbed(nn.Module):
    def __init__(self, vocab_size: int, ctx_dim: int, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, ctx_dim)
        self.norm = nn.LayerNorm(ctx_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, tag_ids: torch.LongTensor):
        # tag_ids: (B, T_ctx) با -1 برای pad
        mask = (tag_ids >= 0).float().unsqueeze(-1)   # (B,T,1)
        safe = tag_ids.clamp(min=0)
        e = self.emb(safe) * mask                     # (B,T,ctx_dim)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = e.sum(dim=1) / denom                 # (B,ctx_dim)
        return self.drop(self.norm(pooled))

class Salmon(nn.Module):
    """
    SALMON (text-only): BERT + context-tag embeddings → fuse → classifier
    """
    def __init__(self, cfg: SalmonConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_name)
        self.backbone = BertModel.from_pretrained(cfg.pretrained_name)
        hid = self.backbone.config.hidden_size

        vocab_size = max(1, len(cfg.ctx_vocab) if cfg.ctx_vocab else 0)
        self.ctx = ContextTagEmbed(vocab_size, cfg.ctx_dim, cfg.dropout)

        self.fuse = nn.Sequential(
            nn.Linear(hid + cfg.ctx_dim, hid),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.classifier = nn.Linear(hid, cfg.num_labels)

        # اگر کانتکست خاموش باشد، یک بردار صفر با اندازه ctx_dim تزریق می‌کنیم
        self.register_buffer("_zero_ctx", torch.zeros(1, cfg.ctx_dim))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        context_tag_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_context: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if use_context is None:
            use_context = self.cfg.use_context

        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (B,H)

        B = input_ids.size(0)
        if (not use_context) or (context_tag_ids is None):
            ctx_vec = self._zero_ctx.expand(B, -1)    # (B,ctx_dim) صفر
            used_ctx = False
        else:
            ctx_vec = self.ctx(context_tag_ids)       # (B,ctx_dim)
            used_ctx = True

        fused = self.fuse(torch.cat([cls, ctx_vec], dim=-1))  # (B,H)
        logits = self.classifier(fused)                       # (B,C)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        aux = {
            "used_context": used_ctx,
            "ctx_dim": self.cfg.ctx_dim
        }
        return {"loss": loss, "logits": logits, "aux": aux}
