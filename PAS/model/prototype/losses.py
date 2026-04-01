from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLosses(nn.Module):
    def __init__(
        self,
        temperature_init: float = 0.07,
        learnable_temperature: bool = False,
        normalize_embeddings: bool = True,
        num_classes: int = 0,
        embedding_dim: int = 0,
        proxy_temperature: float = 0.07,
        lambda_proxy: float = 1.0,
        lambda_align: float = 1.0,
        lambda_diag: float = 1.0,
        use_diversity_loss: bool = False,
        diversity_loss_weight: float = 0.0,
        use_balance_loss: bool = False,
        balance_loss_weight: float = 0.0,
    ):
        super().__init__()
        if temperature_init <= 0:
            raise ValueError('temperature_init must be positive.')
        if proxy_temperature <= 0:
            raise ValueError('proxy_temperature must be positive.')
        if num_classes <= 0:
            raise ValueError('num_classes must be positive for the amortized proxy objective.')
        if embedding_dim <= 0:
            raise ValueError('embedding_dim must be positive for the amortized proxy objective.')
        if learnable_temperature:
            raise ValueError(
                'Learnable retrieval logit scaling is not supported under the amortized surrogate objective. '
                'Exact retrieval scoring keeps a fixed temperature.'
            )

        self.normalize_embeddings = bool(normalize_embeddings)
        self.use_diversity_loss = bool(use_diversity_loss)
        self.lambda_div = float(diversity_loss_weight)
        self.use_balance_loss = bool(use_balance_loss)
        self.lambda_bal = float(balance_loss_weight)
        self.lambda_proxy = float(lambda_proxy)
        self.lambda_align = float(lambda_align)
        self.lambda_diag = float(lambda_diag)
        self.proxy_temperature = float(proxy_temperature)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)

        if not self.use_balance_loss and self.lambda_bal != 0.0:
            raise ValueError('lambda_bal must be 0.0 when use_balance_loss is disabled.')
        if self.use_balance_loss and self.lambda_bal <= 0.0:
            raise ValueError('use_balance_loss requires lambda_bal to be positive.')

        initial_logit_scale = torch.log(torch.tensor(1.0 / temperature_init, dtype=torch.float32))
        self.register_buffer('logit_scale', initial_logit_scale.clone())
        self.class_proxies = nn.Parameter(torch.randn(self.num_classes, self.embedding_dim, dtype=torch.float32))
        with torch.no_grad():
            self.class_proxies.copy_(F.normalize(self.class_proxies, dim=-1))

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def get_retrieval_temperature(self) -> torch.Tensor:
        return torch.reciprocal(self.get_logit_scale())

    def prepare_embeddings(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor):
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError('Similarity inputs must have shape [B, D].')
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError('Image and text embeddings must have the same shape.')
        if self.normalize_embeddings:
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
        return image_embeddings, text_embeddings

    def compute_similarity_matrix(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        image_embeddings, text_embeddings = self.prepare_embeddings(image_embeddings, text_embeddings)
        similarity = text_embeddings @ image_embeddings.t()
        logit_scale = self.get_logit_scale().to(device=similarity.device, dtype=similarity.dtype)
        return similarity * logit_scale

    def compute_paired_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        image_embeddings, text_embeddings = self.prepare_embeddings(image_embeddings, text_embeddings)
        similarity = (text_embeddings * image_embeddings).sum(dim=-1)
        logit_scale = self.get_logit_scale().to(device=similarity.device, dtype=similarity.dtype)
        return similarity * logit_scale

    def _validate_class_labels(self, pids: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if pids is None:
            raise ValueError('pids must be provided as class labels for the amortized proxy objective.')
        if pids.ndim != 1 or pids.numel() != batch_size:
            raise ValueError(f'pids must have shape [B], received {tuple(pids.shape)} for batch size {batch_size}.')
        pids = pids.to(device=device, dtype=torch.long)
        if pids.min().item() < 0 or pids.max().item() >= self.num_classes:
            raise ValueError(
                f'pids must be in [0, {self.num_classes - 1}] for the amortized proxy objective; '
                f'got range [{int(pids.min().item())}, {int(pids.max().item())}].'
            )
        return pids

    def proxy_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError('embeddings must have shape [B, D].')
        embeddings = F.normalize(embeddings, dim=-1)
        proxies = F.normalize(self.class_proxies.to(device=embeddings.device, dtype=embeddings.dtype), dim=-1)
        return (embeddings @ proxies.t()) / self.proxy_temperature

    def proxy_loss(self, embeddings: torch.Tensor, pids: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.proxy_logits(embeddings)
        return {
            'loss': F.cross_entropy(logits, pids),
            'logits': logits,
        }

    def cosine_alignment_loss(self, source_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        source_embeddings = F.normalize(source_embeddings, dim=-1)
        target_embeddings = F.normalize(target_embeddings, dim=-1)
        return (1.0 - (source_embeddings * target_embeddings).sum(dim=-1)).mean()

    def diagonal_fidelity_loss(self, surrogate_embeddings: torch.Tensor, exact_embeddings: torch.Tensor) -> torch.Tensor:
        return self.cosine_alignment_loss(surrogate_embeddings, exact_embeddings.detach())

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
        surrogate_text_embeddings: torch.Tensor,
        exact_text_embeddings: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if image_embeddings.ndim != 2 or surrogate_text_embeddings.ndim != 2 or exact_text_embeddings.ndim != 2:
            raise ValueError('All amortized objective embeddings must have shape [B, D].')
        if image_embeddings.shape != surrogate_text_embeddings.shape or image_embeddings.shape != exact_text_embeddings.shape:
            raise ValueError('Image, surrogate text, and exact text embeddings must share shape [B, D].')

        pids = self._validate_class_labels(pids, image_embeddings.size(0), image_embeddings.device)
        loss_proxy_image_info = self.proxy_loss(image_embeddings, pids)
        loss_proxy_text_info = self.proxy_loss(surrogate_text_embeddings, pids)
        loss_proxy_image = loss_proxy_image_info['loss']
        loss_proxy_text = loss_proxy_text_info['loss']
        loss_proxy = loss_proxy_image + loss_proxy_text
        loss_align = self.cosine_alignment_loss(image_embeddings, surrogate_text_embeddings)
        loss_diag = self.diagonal_fidelity_loss(surrogate_text_embeddings, exact_text_embeddings)
        loss_diversity = self.diversity_loss(prototypes)
        loss_balance = self.balance_loss(routing_weights)
        loss_total = (
            (self.lambda_proxy * loss_proxy)
            + (self.lambda_align * loss_align)
            + (self.lambda_diag * loss_diag)
            + (self.lambda_div * loss_diversity)
            + (self.lambda_bal * loss_balance)
        )
        outputs = {
            'loss_total': loss_total,
            'loss_proxy': loss_proxy,
            'loss_proxy_image': loss_proxy_image,
            'loss_proxy_text': loss_proxy_text,
            'loss_align': loss_align,
            'loss_diag': loss_diag,
            'loss_diversity': loss_diversity,
            'loss_balance': loss_balance,
            'loss_proxy_weighted': self.lambda_proxy * loss_proxy,
            'loss_align_weighted': self.lambda_align * loss_align,
            'loss_diag_weighted': self.lambda_diag * loss_diag,
            'loss_diversity_weighted': self.lambda_div * loss_diversity,
            'loss_balance_weighted': self.lambda_bal * loss_balance,
            'lambda_proxy': torch.tensor(self.lambda_proxy, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_align': torch.tensor(self.lambda_align, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_diag': torch.tensor(self.lambda_diag, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_div': torch.tensor(self.lambda_div, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_bal': torch.tensor(self.lambda_bal, device=loss_total.device, dtype=loss_total.dtype),
            'proxy_temperature': torch.tensor(self.proxy_temperature, device=loss_total.device, dtype=loss_total.dtype),
            'retrieval_temperature': self.get_retrieval_temperature().to(device=loss_total.device, dtype=loss_total.dtype),
            'logit_scale': self.get_logit_scale().to(device=loss_total.device, dtype=loss_total.dtype),
        }
        if return_debug:
            outputs['image_proxy_logits'] = loss_proxy_image_info['logits']
            outputs['text_proxy_logits'] = loss_proxy_text_info['logits']
            outputs['class_proxies'] = self.class_proxies.detach()
        return outputs
