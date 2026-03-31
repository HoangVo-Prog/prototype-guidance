from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLosses(nn.Module):
    def __init__(
        self,
        temperature_init: float = 0.07,
        learnable_temperature: bool = True,
        normalize_embeddings: bool = True,
        use_diversity_loss: bool = False,
        diversity_loss_weight: float = 0.0,
        use_balance_loss: bool = False,
        balance_loss_weight: float = 0.0,
    ):
        super().__init__()
        if temperature_init <= 0:
            raise ValueError('temperature_init must be positive.')
        self.learnable_temperature = bool(learnable_temperature)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.use_diversity_loss = bool(use_diversity_loss)
        self.lambda_div = float(diversity_loss_weight)
        self.use_balance_loss = bool(use_balance_loss)
        self.lambda_bal = float(balance_loss_weight)

        if not self.use_balance_loss and self.lambda_bal != 0.0:
            raise ValueError('lambda_bal must be 0.0 when use_balance_loss is disabled.')
        if self.use_balance_loss and self.lambda_bal <= 0.0:
            raise ValueError('use_balance_loss requires lambda_bal to be positive.')

        initial_logit_scale = torch.log(torch.tensor(1.0 / temperature_init, dtype=torch.float32))
        if self.learnable_temperature:
            self.logit_scale = nn.Parameter(initial_logit_scale.clone())
        else:
            self.register_buffer('logit_scale', initial_logit_scale.clone())

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def prepare_embeddings(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor):
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError('InfoNCE inputs must have shape [B, D].')
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError('Image and text embeddings must have the same shape.')
        if self.normalize_embeddings:
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
        return image_embeddings, text_embeddings

    def compute_contrastive_logits(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        image_embeddings, text_embeddings = self.prepare_embeddings(image_embeddings, text_embeddings)
        logits = text_embeddings @ image_embeddings.t()
        logit_scale = self.get_logit_scale().to(device=logits.device, dtype=logits.dtype)
        return logits * logit_scale

    def compute_paired_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        image_embeddings, text_embeddings = self.prepare_embeddings(image_embeddings, text_embeddings)
        similarity = (text_embeddings * image_embeddings).sum(dim=-1)
        logit_scale = self.get_logit_scale().to(device=similarity.device, dtype=similarity.dtype)
        return similarity * logit_scale

    def _build_positive_mask(self, pids: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if pids is None:
            return torch.eye(batch_size, device=device, dtype=torch.bool)
        if pids.ndim != 1 or pids.numel() != batch_size:
            raise ValueError(f'pids must have shape [B], received {tuple(pids.shape)} for batch size {batch_size}.')
        pids = pids.to(device=device)
        return pids.view(-1, 1).eq(pids.view(1, -1))

    def _multi_positive_loss(self, logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.size(0) != logits.size(1):
            raise ValueError('logits must have shape [B, B].')
        if positive_mask.shape != logits.shape:
            raise ValueError('positive_mask must match logits shape [B, B].')
        positive_mask = positive_mask.to(dtype=logits.dtype)
        positive_counts = positive_mask.sum(dim=-1)
        if torch.any(positive_counts <= 0):
            raise ValueError('Each row must contain at least one positive pair.')
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return -((log_probs * positive_mask).sum(dim=-1) / positive_counts).mean()

    def symmetric_infonce(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        logits = self.compute_contrastive_logits(image_embeddings, text_embeddings)
        positive_mask = self._build_positive_mask(pids, logits.size(0), logits.device)
        loss_t2i = self._multi_positive_loss(logits, positive_mask)
        loss_i2t = self._multi_positive_loss(logits.t(), positive_mask.t())
        loss_infonce = 0.5 * (loss_t2i + loss_i2t)
        outputs = {
            'loss_infonce': loss_infonce,
            'contrastive_logits': logits,
        }
        if return_debug:
            outputs.update(
                {
                    'contrastive_positive_mask': positive_mask,
                    'contrastive_positive_counts': positive_mask.sum(dim=-1),
                    'loss_t2i': loss_t2i,
                    'loss_i2t': loss_i2t,
                }
            )
        return outputs

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
        pids: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        info = self.symmetric_infonce(image_embeddings, text_embeddings, pids=pids, return_debug=return_debug)
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
            outputs['contrastive_positive_mask'] = info['contrastive_positive_mask']
            outputs['contrastive_positive_counts'] = info['contrastive_positive_counts']
            outputs['loss_t2i'] = info['loss_t2i']
            outputs['loss_i2t'] = info['loss_i2t']
        return outputs
