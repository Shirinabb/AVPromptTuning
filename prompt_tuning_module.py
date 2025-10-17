
# A lightweight 'soft prompt' module for HuggingFace models:
# - Prepend learnable embeddings to token embeddings during forward pass
# - Works with encoder-only (BERT) and decoder-only (GPT-2) models for classification
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel

class SoftPromptWrapper(nn.Module):
    def __init__(self, base_model_name: str, prompt_length: int = 10, freeze_base: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.model.config.hidden_size
        self.prompt_length = prompt_length
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, self.hidden_size) * 0.02)
        if freeze_base:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        bsz = inputs_embeds.size(0)
        prompt = self.soft_prompt.unsqueeze(0).expand(bsz, -1, -1)  # (B, P, H)

        # Prepend prompt embeddings
        inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)    # (B, P+L, H)

        if attention_mask is not None:
            prompt_mask = torch.ones((bsz, self.prompt_length), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        return outputs
