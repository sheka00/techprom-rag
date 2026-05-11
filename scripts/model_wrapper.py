"""Обёртка BGE-M3 для экспорта в ONNX."""

import torch.nn as nn
import torch.nn.functional as F


class BGE_M3_Wrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, input_ids, attention_mask):
        model_output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        cls_embedding = model_output.last_hidden_state[:, 0, :]
        normalized = F.normalize(cls_embedding, p=2, dim=1)
        return normalized
