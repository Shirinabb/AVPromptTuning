# src/models/salmonn.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
@dataclass
class SalmonNConfig:
    text_backbone: str = "bert-base-uncased"
    num_labels: int = 4
    ctx_dim: int = 16
    dropout: float = 0.1
    ctx_vocab: Dict[str, int] = None
    soft_prompt_len: int = 16
    use_context: bool = True
    use_soft_prompt: bool = True
class SoftPrompt(nn.Module):
    def __init__(self, length: int, hidden: int):
        super().__init__()
        self.length = length
        self.emb = nn.Parameter(torch.randn(length, hidden) * 0.02)

    def forward(self, B: int):
        return self.emb.unsqueeze(0).expand(B, -1, -1)

class ContextTagEmbed(nn.Module):
    def __init__(self, vocab_size: int, ctx_dim: int, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, ctx_dim)
        self.proj = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(ctx_dim)
        )

    def forward(self, tag_ids: torch.LongTensor):
        mask = (tag_ids >= 0).float().unsqueeze(-1)
        safe = tag_ids.clamp(min=0)
        e = self.emb(safe) * mask          # (B,T,ctx_dim)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = e.sum(dim=1) / denom      # (B,ctx_dim)
        return self.proj(pooled)

class SalmonN(nn.Module):
    def __init__(self, cfg: SalmonNConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_backbone)
        self.backbone = BertModel.from_pretrained(cfg.text_backbone)
        hid = self.backbone.config.hidden_size

        # soft prompt
        if cfg.use_soft_prompt and cfg.soft_prompt_len > 0:
            self.soft = SoftPrompt(cfg.soft_prompt_len, hid)
        else:
            self.soft = None

        # context tags
        vocab_size = max(1, len(cfg.ctx_vocab) if cfg.ctx_vocab else 0)
        self.ctx = ContextTagEmbed(vocab_size, cfg.ctx_dim, cfg.dropout)

        self.ctx_to_h = nn.Linear(cfg.ctx_dim, hid)
        self.ctx_gate = nn.Sequential(
            nn.Linear(cfg.ctx_dim, hid),
            nn.Sigmoid()  
        )

        self.fuse = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Dropout(cfg.dropout)
        )
        self.classifier = nn.Linear(hid, cfg.num_labels)

        self.register_buffer("_zero_ctx", torch.zeros(1, cfg.ctx_dim))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        context_tag_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_context: Optional[bool] = None,
        use_soft_prompt: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if use_context is None:
            use_context = self.cfg.use_context
        if use_soft_prompt is None:
            use_soft_prompt = self.cfg.use_soft_prompt

        B = input_ids.size(0)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]    # (B,H)
        if use_soft_prompt and (self.soft is not None):
            soft_tok = self.soft(B)             # (B,Ls,H)
            soft_pooled = soft_tok.mean(dim=1)  # (B,H)
            mixed = cls + soft_pooled
            used_soft = True
        else:
            mixed = cls
            used_soft = False
        if (not use_context) or (context_tag_ids is None):
            ctx_vec = self._zero_ctx.expand(B, -1)  # (B,ctx_dim) صفر
            gate = torch.ones(B, self.backbone.config.hidden_size, device=cls.device)
            used_ctx = False
            gate_mean = gate.mean().detach()
        else:
            ctx_vec = self.ctx(context_tag_ids)      # (B,ctx_dim)
            gate = self.ctx_gate(ctx_vec)            # (B,H)
            used_ctx = True
            gate_mean = gate.mean().detach()
        ctx_h = self.ctx_to_h(ctx_vec)               # (B,H)
        mixed = mixed * (1.0 - 0.3*gate) + ctx_h * gate

        fused = self.fuse(mixed)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        aux = {
            "used_context": used_ctx,
            "used_soft_prompt": used_soft,
            "soft_prompt_len": (self.cfg.soft_prompt_len if used_soft else 0),
            "gate_mean": gate_mean.item()
        }
        return {"loss": loss, "logits": logits, "aux": aux}
