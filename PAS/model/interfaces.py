from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class EncoderOutput:
    tokens: torch.Tensor
    pooled: torch.Tensor
    projected_tokens: torch.Tensor
    projected_pooled: torch.Tensor
    pre_projection_tokens: Optional[torch.Tensor] = None
    pre_projection_pooled: Optional[torch.Tensor] = None
    attention_weights: Optional[Any] = None
    token_mask: Optional[torch.Tensor] = None
    special_token_positions: Dict[str, torch.Tensor] = field(default_factory=dict)
    pooling_mode: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
