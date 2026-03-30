from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECTOR_TYPE_ALIASES = {
    'mlp2': 'mlp2',
    'mlp': 'mlp2',
    'linear': 'linear',
}


class MLPProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        normalize_output: bool = True,
        projector_type: str = 'mlp2',
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError('Projector dimensions must be positive.')
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)
        self.normalize_output = bool(normalize_output)
        self.projector_type = PROJECTOR_TYPE_ALIASES.get(str(projector_type).lower())
        if self.projector_type is None:
            raise ValueError(f'Unsupported projector type: {projector_type}')
        if self.projector_type == 'linear':
            self.net = nn.Linear(self.input_dim, self.output_dim)
        else:
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
            'projector_type': self.projector_type,
        }
