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
        lambda_proxy_image: Optional[float] = None,
        lambda_proxy_text: Optional[float] = None,
        lambda_proxy_text_exact: Optional[float] = None,
        use_loss_proxy_image: bool = True,
        use_loss_proxy_text: bool = True,
        use_loss_proxy_text_exact: bool = True,
        use_loss_align: bool = True,
        lambda_align: float = 1.0,
        use_loss_diag: bool = True,
        lambda_diag: float = 1.0,
        use_loss_ret: bool = True,
        lambda_ret: float = 1.0,
        use_loss_support: bool = False,
        support_loss_weight: float = 0.0,
        support_min: float = 2.0,
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
        self.lambda_proxy_image = self.lambda_proxy if lambda_proxy_image is None else float(lambda_proxy_image)
        self.lambda_proxy_text = self.lambda_proxy if lambda_proxy_text is None else float(lambda_proxy_text)
        self.lambda_proxy_text_exact = self.lambda_proxy if lambda_proxy_text_exact is None else float(lambda_proxy_text_exact)
        self.use_loss_proxy_image = bool(use_loss_proxy_image)
        self.use_loss_proxy_text = bool(use_loss_proxy_text)
        self.use_loss_proxy_text_exact = bool(use_loss_proxy_text_exact)
        self.use_loss_align = bool(use_loss_align)
        self.lambda_align = float(lambda_align)
        self.use_loss_diag = bool(use_loss_diag)
        self.lambda_diag = float(lambda_diag)
        self.use_loss_ret = bool(use_loss_ret)
        self.lambda_ret = float(lambda_ret)
        self.use_loss_support = bool(use_loss_support)
        self.lambda_support = float(support_loss_weight)
        self.support_min = float(support_min)
        self.proxy_temperature = float(proxy_temperature)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)

        if self.support_min <= 0.0:
            raise ValueError('support_min must be positive.')


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

    def _norm_stats(self, prefix: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        norms = tensor.norm(dim=-1)
        return {
            f'{prefix}_mean': norms.mean().detach(),
            f'{prefix}_std': norms.std(unbiased=False).detach(),
            f'{prefix}_min': norms.min().detach(),
            f'{prefix}_max': norms.max().detach(),
        }

    def _proxy_debug_metrics(self, prefix: str, logits: Optional[torch.Tensor], pids: torch.Tensor) -> Dict[str, torch.Tensor]:
        if logits is None:
            return {}
        cosines = logits * self.proxy_temperature
        positive = cosines.gather(1, pids.view(-1, 1)).squeeze(1)
        negative_mask = torch.ones_like(cosines, dtype=torch.bool)
        negative_mask.scatter_(1, pids.view(-1, 1), False)
        hardest_negative = cosines.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
        margin = positive - hardest_negative
        return {
            f'{prefix}_proxy_logit_mean': logits.mean().detach(),
            f'{prefix}_proxy_logit_std': logits.std(unbiased=False).detach(),
            f'{prefix}_proxy_logit_min': logits.min().detach(),
            f'{prefix}_proxy_logit_max': logits.max().detach(),
            f'{prefix}_positive_proxy_cosine_mean': positive.mean().detach(),
            f'{prefix}_positive_proxy_cosine_std': positive.std(unbiased=False).detach(),
            f'{prefix}_hardest_negative_proxy_cosine_mean': hardest_negative.mean().detach(),
            f'{prefix}_hardest_negative_proxy_cosine_std': hardest_negative.std(unbiased=False).detach(),
            f'{prefix}_proxy_margin_mean': margin.mean().detach(),
            f'{prefix}_proxy_margin_min': margin.min().detach(),
        }

    def _cross_modal_debug_metrics(
        self,
        prefix: str,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        pids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            return {}
        if image_embeddings.shape != text_embeddings.shape:
            return {}

        image_normalized = F.normalize(image_embeddings, dim=-1)
        text_normalized = F.normalize(text_embeddings, dim=-1)
        cosine_matrix = text_normalized @ image_normalized.t()
        positive = cosine_matrix.diagonal()
        if positive.numel() == 0:
            return {}

        same_pid_mask = pids.view(-1, 1).eq(pids.view(1, -1))
        negative_mask = ~same_pid_mask
        if negative_mask.size(1) <= 1 or not negative_mask.any(dim=1).all():
            negative_mask = ~torch.eye(cosine_matrix.size(0), device=cosine_matrix.device, dtype=torch.bool)
        if negative_mask.size(1) <= 1 or not negative_mask.any(dim=1).all():
            hardest_negative = torch.zeros_like(positive)
        else:
            hardest_negative = cosine_matrix.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
        margin = positive - hardest_negative

        logit_scale = self.get_logit_scale().to(device=cosine_matrix.device, dtype=cosine_matrix.dtype)
        scaled_similarity = cosine_matrix * logit_scale
        positive_logits = scaled_similarity.diagonal()
        if negative_mask.size(1) <= 1 or not negative_mask.any(dim=1).all():
            hardest_negative_logits = torch.zeros_like(positive_logits)
        else:
            hardest_negative_logits = scaled_similarity.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
        return {
            f'{prefix}_positive_cosine_mean': positive.mean().detach(),
            f'{prefix}_positive_cosine_std': positive.std(unbiased=False).detach(),
            f'{prefix}_hardest_negative_cosine_mean': hardest_negative.mean().detach(),
            f'{prefix}_hardest_negative_cosine_std': hardest_negative.std(unbiased=False).detach(),
            f'{prefix}_margin_mean': margin.mean().detach(),
            f'{prefix}_margin_min': margin.min().detach(),
            f'{prefix}_positive_logit_mean': positive_logits.mean().detach(),
            f'{prefix}_hardest_negative_logit_mean': hardest_negative_logits.mean().detach(),
        }

    def _surrogate_pairwise_debug_metrics(self, surrogate_pairwise_logits: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if surrogate_pairwise_logits is None:
            return {}
        if surrogate_pairwise_logits.ndim != 2 or surrogate_pairwise_logits.size(0) != surrogate_pairwise_logits.size(1):
            return {}

        batch_size = surrogate_pairwise_logits.size(0)
        base_logit_scale = self.get_logit_scale().to(
            device=surrogate_pairwise_logits.device,
            dtype=surrogate_pairwise_logits.dtype,
        )
        pairwise_cosine = surrogate_pairwise_logits / base_logit_scale.clamp_min(1e-12)
        positive = pairwise_cosine.diagonal()
        if batch_size <= 1:
            hardest_negative = torch.zeros_like(positive)
        else:
            negative_mask = ~torch.eye(batch_size, device=pairwise_cosine.device, dtype=torch.bool)
            hardest_negative = pairwise_cosine.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
        margin = positive - hardest_negative
        positive_logits = surrogate_pairwise_logits.diagonal()
        if batch_size <= 1:
            hardest_negative_logits = torch.zeros_like(positive_logits)
        else:
            negative_mask = ~torch.eye(batch_size, device=surrogate_pairwise_logits.device, dtype=torch.bool)
            hardest_negative_logits = surrogate_pairwise_logits.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
        return {
            'surrogate_pairwise_positive_cosine_mean': positive.mean().detach(),
            'surrogate_pairwise_positive_cosine_std': positive.std(unbiased=False).detach(),
            'surrogate_pairwise_hardest_negative_cosine_mean': hardest_negative.mean().detach(),
            'surrogate_pairwise_hardest_negative_cosine_std': hardest_negative.std(unbiased=False).detach(),
            'surrogate_pairwise_margin_mean': margin.mean().detach(),
            'surrogate_pairwise_margin_min': margin.min().detach(),
            'surrogate_pairwise_positive_logit_mean': positive_logits.mean().detach(),
            'surrogate_pairwise_hardest_negative_logit_mean': hardest_negative_logits.mean().detach(),
            'surrogate_pairwise_logit_mean': surrogate_pairwise_logits.mean().detach(),
            'surrogate_pairwise_logit_std': surrogate_pairwise_logits.std(unbiased=False).detach(),
        }

    def _normalized_norm_stats(self, prefix: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized = F.normalize(tensor, dim=-1)
        norms = normalized.norm(dim=-1)
        return {
            f'{prefix}_mean': norms.mean().detach(),
            f'{prefix}_std': norms.std(unbiased=False).detach(),
            f'{prefix}_min': norms.min().detach(),
            f'{prefix}_max': norms.max().detach(),
        }

    def cosine_alignment_loss(self, source_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        source_embeddings = F.normalize(source_embeddings, dim=-1)
        target_embeddings = F.normalize(target_embeddings, dim=-1)
        return (1.0 - (source_embeddings * target_embeddings).sum(dim=-1)).clamp_min(0.0).mean()

    def diagonal_fidelity_loss(self, surrogate_embeddings: torch.Tensor, exact_embeddings: torch.Tensor) -> torch.Tensor:
        return self.cosine_alignment_loss(surrogate_embeddings, exact_embeddings.detach())

    def effective_support(self, routing_weights: torch.Tensor) -> torch.Tensor:
        return torch.reciprocal(routing_weights.pow(2).sum(dim=-1).clamp_min(1e-12))

    def support_loss(self, routing_weights: Optional[torch.Tensor]) -> torch.Tensor:
        if routing_weights is None or not self.use_loss_support:
            device = self.logit_scale.device if routing_weights is None else routing_weights.device
            return torch.zeros((), device=device)
        return (self.support_min - self.effective_support(routing_weights)).clamp_min(0.0).pow(2).mean()

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

    def surrogate_retrieval_loss(self, surrogate_pairwise_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        if surrogate_pairwise_logits.ndim != 2 or surrogate_pairwise_logits.size(0) != surrogate_pairwise_logits.size(1):
            raise ValueError(
                'surrogate_pairwise_logits must have shape [B, B] with image rows and text columns; '
                f'received {tuple(surrogate_pairwise_logits.shape)}.'
            )
        targets = torch.arange(surrogate_pairwise_logits.size(0), device=surrogate_pairwise_logits.device)
        return {
            'loss': F.cross_entropy(surrogate_pairwise_logits, targets),
            'logits': surrogate_pairwise_logits,
        }

    def forward(
        self,
        image_embeddings: torch.Tensor,
        surrogate_text_embeddings: torch.Tensor,
        exact_text_embeddings: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        surrogate_pairwise_logits: Optional[torch.Tensor] = None,
        return_debug: bool = False,
        disable_proxy_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if image_embeddings.ndim != 2 or surrogate_text_embeddings.ndim != 2 or exact_text_embeddings.ndim != 2:
            raise ValueError('All amortized objective embeddings must have shape [B, D].')
        if image_embeddings.shape != surrogate_text_embeddings.shape or image_embeddings.shape != exact_text_embeddings.shape:
            raise ValueError('Image, surrogate text, and exact text embeddings must share shape [B, D].')

        batch_size = image_embeddings.size(0)
        if self.use_loss_ret:
            if surrogate_pairwise_logits is None:
                raise ValueError('surrogate_pairwise_logits must be provided when use_loss_ret is enabled.')
            if surrogate_pairwise_logits.ndim != 2 or surrogate_pairwise_logits.shape != (batch_size, batch_size):
                raise ValueError(
                    'surrogate_pairwise_logits must have shape [B, B] with image rows matching the image batch; '
                    f'got {tuple(surrogate_pairwise_logits.shape)} for batch size {batch_size}.'
                )

        proxy_losses_active = bool((self.use_loss_proxy_image or self.use_loss_proxy_text or self.use_loss_proxy_text_exact) and not disable_proxy_losses)
        pids_for_metrics = pids
        if pids_for_metrics is None:
            pids_for_metrics = torch.arange(batch_size, device=image_embeddings.device, dtype=torch.long)
        else:
            pids_for_metrics = pids_for_metrics.to(device=image_embeddings.device, dtype=torch.long)
            if pids_for_metrics.ndim != 1 or pids_for_metrics.numel() != batch_size:
                raise ValueError(f'pids must have shape [B], received {tuple(pids_for_metrics.shape)} for batch size {batch_size}.')

        if proxy_losses_active:
            proxy_pids = self._validate_class_labels(pids_for_metrics, batch_size, image_embeddings.device)
            loss_proxy_image_info = self.proxy_loss(image_embeddings, proxy_pids)
            loss_proxy_text_info = self.proxy_loss(surrogate_text_embeddings, proxy_pids)
            loss_proxy_text_exact_info = self.proxy_loss(exact_text_embeddings, proxy_pids)
        else:
            proxy_pids = None
            loss_proxy_image_info = {'loss': image_embeddings.new_zeros(()), 'logits': None}
            loss_proxy_text_info = {'loss': image_embeddings.new_zeros(()), 'logits': None}
            loss_proxy_text_exact_info = {'loss': image_embeddings.new_zeros(()), 'logits': None}

        loss_ret_info = self.surrogate_retrieval_loss(surrogate_pairwise_logits) if self.use_loss_ret else None
        zero = image_embeddings.new_zeros(())
        loss_proxy_image = loss_proxy_image_info['loss'] if self.use_loss_proxy_image else zero
        loss_proxy_text = loss_proxy_text_info['loss'] if self.use_loss_proxy_text else zero
        loss_proxy_text_exact = loss_proxy_text_exact_info['loss'] if self.use_loss_proxy_text_exact else zero
        loss_proxy = loss_proxy_image + loss_proxy_text + loss_proxy_text_exact
        loss_proxy_image_weighted = self.lambda_proxy_image * loss_proxy_image
        loss_proxy_text_weighted = self.lambda_proxy_text * loss_proxy_text
        loss_proxy_text_exact_weighted = self.lambda_proxy_text_exact * loss_proxy_text_exact
        loss_proxy_weighted = loss_proxy_image_weighted + loss_proxy_text_weighted + loss_proxy_text_exact_weighted
        loss_align = self.cosine_alignment_loss(image_embeddings, surrogate_text_embeddings) if self.use_loss_align else zero
        loss_diag = self.diagonal_fidelity_loss(surrogate_text_embeddings, exact_text_embeddings) if self.use_loss_diag else zero
        loss_ret = loss_ret_info['loss'] if self.use_loss_ret else zero
        loss_ret_weighted = self.lambda_ret * loss_ret
        loss_support = self.support_loss(routing_weights)
        loss_diversity = self.diversity_loss(prototypes)
        loss_balance = self.balance_loss(routing_weights)
        loss_total = (
            loss_proxy_weighted
            + (self.lambda_align * loss_align)
            + (self.lambda_diag * loss_diag)
            + loss_ret_weighted
            + (self.lambda_support * loss_support)
            + (self.lambda_div * loss_diversity)
            + (self.lambda_bal * loss_balance)
        )

        outputs = {
            'loss_total': loss_total,
            'loss_proxy': loss_proxy,
            'loss_proxy_image': loss_proxy_image,
            'loss_proxy_text': loss_proxy_text,
            'loss_proxy_text_exact': loss_proxy_text_exact,
            'loss_ret': loss_ret,
            'loss_align': loss_align,
            'loss_diag': loss_diag,
            'loss_support': loss_support,
            'loss_diversity': loss_diversity,
            'loss_balance': loss_balance,
            'loss_proxy_image_weighted': loss_proxy_image_weighted,
            'loss_proxy_text_weighted': loss_proxy_text_weighted,
            'loss_proxy_text_exact_weighted': loss_proxy_text_exact_weighted,
            'loss_proxy_weighted': loss_proxy_weighted,
            'loss_ret_weighted': loss_ret_weighted,
            'loss_align_weighted': self.lambda_align * loss_align,
            'loss_diag_weighted': self.lambda_diag * loss_diag,
            'loss_support_weighted': self.lambda_support * loss_support,
            'loss_diversity_weighted': self.lambda_div * loss_diversity,
            'loss_balance_weighted': self.lambda_bal * loss_balance,
            'lambda_proxy': torch.tensor(self.lambda_proxy, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_proxy_image': torch.tensor(self.lambda_proxy_image, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_proxy_text': torch.tensor(self.lambda_proxy_text, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_proxy_text_exact': torch.tensor(self.lambda_proxy_text_exact, device=loss_total.device, dtype=loss_total.dtype),
            'use_loss_proxy_image': torch.tensor(float(self.use_loss_proxy_image), device=loss_total.device, dtype=loss_total.dtype),
            'use_loss_proxy_text': torch.tensor(float(self.use_loss_proxy_text), device=loss_total.device, dtype=loss_total.dtype),
            'use_loss_proxy_text_exact': torch.tensor(float(self.use_loss_proxy_text_exact), device=loss_total.device, dtype=loss_total.dtype),
            'use_loss_ret': torch.tensor(float(self.use_loss_ret), device=loss_total.device, dtype=loss_total.dtype),
            'lambda_ret': torch.tensor(self.lambda_ret, device=loss_total.device, dtype=loss_total.dtype),
            'use_loss_align': torch.tensor(float(self.use_loss_align), device=loss_total.device, dtype=loss_total.dtype),
            'lambda_align': torch.tensor(self.lambda_align, device=loss_total.device, dtype=loss_total.dtype),
            'use_loss_diag': torch.tensor(float(self.use_loss_diag), device=loss_total.device, dtype=loss_total.dtype),
            'lambda_diag': torch.tensor(self.lambda_diag, device=loss_total.device, dtype=loss_total.dtype),
            'use_loss_support': torch.tensor(float(self.use_loss_support), device=loss_total.device, dtype=loss_total.dtype),
            'lambda_support': torch.tensor(self.lambda_support, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_div': torch.tensor(self.lambda_div, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_bal': torch.tensor(self.lambda_bal, device=loss_total.device, dtype=loss_total.dtype),
            'proxy_temperature': torch.tensor(self.proxy_temperature, device=loss_total.device, dtype=loss_total.dtype),
            'retrieval_temperature': self.get_retrieval_temperature().to(device=loss_total.device, dtype=loss_total.dtype),
            'logit_scale': self.get_logit_scale().to(device=loss_total.device, dtype=loss_total.dtype),
            'debug_metrics': {
                **self._proxy_debug_metrics('image', loss_proxy_image_info['logits'], proxy_pids if proxy_pids is not None else pids_for_metrics),
                **self._proxy_debug_metrics('text', loss_proxy_text_info['logits'], proxy_pids if proxy_pids is not None else pids_for_metrics),
                **self._proxy_debug_metrics('text_exact', loss_proxy_text_exact_info['logits'], proxy_pids if proxy_pids is not None else pids_for_metrics),
                **self._cross_modal_debug_metrics('image_surrogate', image_embeddings, surrogate_text_embeddings, pids_for_metrics),
                **self._cross_modal_debug_metrics('image_exact', image_embeddings, exact_text_embeddings, pids_for_metrics),
                **self._surrogate_pairwise_debug_metrics(loss_ret_info['logits'] if loss_ret_info is not None else None),
                **self._norm_stats('class_proxy_norm', self.class_proxies.detach()),
                **self._normalized_norm_stats('class_proxy_norm_normalized', self.class_proxies.detach()),
            },
        }
        if return_debug:
            outputs['image_proxy_logits'] = loss_proxy_image_info['logits']
            outputs['text_proxy_logits'] = loss_proxy_text_info['logits']
            outputs['text_exact_proxy_logits'] = loss_proxy_text_exact_info['logits']
            if loss_ret_info is not None:
                outputs['surrogate_retrieval_logits'] = loss_ret_info['logits']
            outputs['class_proxies'] = self.class_proxies.detach()
        return outputs
