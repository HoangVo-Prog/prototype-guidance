from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        normalize_output: bool = True,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError('Projector dimensions must be positive.')
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)
        self.normalize_output = bool(normalize_output)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, inputs: torch.Tensor, return_debug: bool = False):
        projected_raw = self.net(inputs)
        projected = F.normalize(projected_raw, dim=-1) if self.normalize_output else projected_raw
        if not return_debug:
            return projected
        return projected, {
            'projected_features': projected,
            'projected_features_norm': projected,
            'projected_features_pre_norm': projected_raw,
            'projected_features_raw': projected_raw,
        }
