from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLosses(nn.Module):
    def __init__(
        self,
        temperature_init: float = 0.07,
        learnable_temperature: bool = True,
        use_diversity_loss: bool = False,
        diversity_loss_weight: float = 0.0,
        use_balance_loss: bool = False,
        balance_loss_weight: float = 0.0,
    ):
        super().__init__()
        if temperature_init <= 0:
            raise ValueError('temperature_init must be positive.')
        self.learnable_temperature = bool(learnable_temperature)
        self.use_diversity_loss = bool(use_diversity_loss)
        self.lambda_div = float(diversity_loss_weight)
        self.use_balance_loss = bool(use_balance_loss)
        self.lambda_bal = float(balance_loss_weight)

        initial_logit_scale = torch.log(torch.tensor(1.0 / temperature_init, dtype=torch.float32))
        if self.learnable_temperature:
            self.logit_scale = nn.Parameter(initial_logit_scale.clone())
        else:
            self.register_buffer('logit_scale', initial_logit_scale.clone())

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def symmetric_infonce(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError('InfoNCE inputs must have shape [B, D].')
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError('Image and text embeddings must have the same shape.')
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        logits = text_embeddings @ image_embeddings.t()
        logits = logits * self.get_logit_scale()
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.t(), labels)
        loss_infonce = 0.5 * (loss_t2i + loss_i2t)
        return {
            'loss_infonce': loss_infonce,
            'contrastive_logits': logits,
        }

    def diversity_loss(self, prototypes: Optional[torch.Tensor]) -> torch.Tensor:
        if prototypes is None or not self.use_diversity_loss:
            device = self.logit_scale.device if prototypes is None else prototypes.device
            return torch.zeros((), device=device)
        normalized = F.normalize(prototypes, dim=-1)
        similarity = normalized @ normalized.t()
        identity = torch.eye(similarity.size(0), device=similarity.device, dtype=similarity.dtype)
        return (similarity - identity).pow(2).sum()

    def balance_loss(self, routing_weights: Optional[torch.Tensor]) -> torch.Tensor:
        if routing_weights is None or not self.use_balance_loss:
            device = self.logit_scale.device if routing_weights is None else routing_weights.device
            return torch.zeros((), device=device)
        usage = routing_weights.mean(dim=0)
        target = torch.full_like(usage, 1.0 / usage.numel())
        return (usage - target).pow(2).sum()

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        info = self.symmetric_infonce(image_embeddings, text_embeddings)
        loss_diversity = self.diversity_loss(prototypes)
        loss_balance = self.balance_loss(routing_weights)
        loss_total = info['loss_infonce'] + (self.lambda_div * loss_diversity) + (self.lambda_bal * loss_balance)
        outputs = {
            'loss_total': loss_total,
            'loss_infonce': info['loss_infonce'],
            'loss_diversity': loss_diversity,
            'loss_balance': loss_balance,
            'loss_diversity_weighted': self.lambda_div * loss_diversity,
            'loss_balance_weighted': self.lambda_bal * loss_balance,
            'lambda_div': torch.tensor(self.lambda_div, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_bal': torch.tensor(self.lambda_bal, device=loss_total.device, dtype=loss_total.dtype),
            'logit_scale': self.get_logit_scale(),
        }
        if return_debug:
            outputs['contrastive_logits'] = info['contrastive_logits']
        return outputs
