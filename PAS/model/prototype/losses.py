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
        use_loss_ret_exact: bool = False,
        use_loss_ret_exact_image: Optional[bool] = None,
        use_loss_ret_exact_text: Optional[bool] = None,
        lambda_ret_exact: float = 1.0,
        lambda_ret_exact_image: Optional[float] = None,
        lambda_ret_exact_text: Optional[float] = None,
        ret_exact_temperature: Optional[float] = None,
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
        if ret_exact_temperature is not None and ret_exact_temperature <= 0:
            raise ValueError('ret_exact_temperature must be positive when provided.')
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
        self.use_loss_ret_exact = bool(use_loss_ret_exact)
        self.use_loss_ret_exact_image = self.use_loss_ret_exact if use_loss_ret_exact_image is None else bool(use_loss_ret_exact_image)
        self.use_loss_ret_exact_text = False if use_loss_ret_exact_text is None else bool(use_loss_ret_exact_text)
        self.use_loss_ret_exact = bool(self.use_loss_ret_exact_image or self.use_loss_ret_exact_text)
        self.lambda_ret_exact = float(lambda_ret_exact)
        self.lambda_ret_exact_image = self.lambda_ret_exact if lambda_ret_exact_image is None else float(lambda_ret_exact_image)
        self.lambda_ret_exact_text = self.lambda_ret_exact if lambda_ret_exact_text is None else float(lambda_ret_exact_text)
        self.ret_exact_temperature = None if ret_exact_temperature is None else float(ret_exact_temperature)
        self.use_loss_support = bool(use_loss_support)
        self.lambda_support = float(support_loss_weight)
        self.support_min = float(support_min)
        self.proxy_temperature = float(proxy_temperature)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)

        if self.support_min <= 0.0:
            raise ValueError('support_min must be positive.')
        if not self.use_loss_support and self.lambda_support != 0.0:
            raise ValueError('lambda_support must be 0.0 when use_loss_support is disabled.')
        if self.use_loss_support and self.lambda_support <= 0.0:
            raise ValueError('use_loss_support requires lambda_support to be positive.')
        if not self.use_balance_loss and self.lambda_bal != 0.0:
            raise ValueError('lambda_bal must be 0.0 when use_balance_loss is disabled.')
        if self.use_balance_loss and self.lambda_bal <= 0.0:
            raise ValueError('use_balance_loss requires lambda_bal to be positive.')

        if not any((
            self.use_loss_proxy_image,
            self.use_loss_proxy_text,
            self.use_loss_proxy_text_exact,
            self.use_loss_align,
            self.use_loss_diag,
            self.use_loss_ret_exact_image,
            self.use_loss_ret_exact_text,
        )):
            raise ValueError('At least one task-supervised loss must remain enabled so the training objective does not collapse to prototype-only regularization.')

        initial_logit_scale = torch.log(torch.tensor(1.0 / temperature_init, dtype=torch.float32))
        self.register_buffer('logit_scale', initial_logit_scale.clone())
        self.class_proxies = nn.Parameter(torch.randn(self.num_classes, self.embedding_dim, dtype=torch.float32))
        with torch.no_grad():
            self.class_proxies.copy_(F.normalize(self.class_proxies, dim=-1))

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def get_retrieval_temperature(self) -> torch.Tensor:
        return torch.reciprocal(self.get_logit_scale())

    def get_ret_exact_temperature(self) -> torch.Tensor:
        if self.ret_exact_temperature is None:
            return self.get_retrieval_temperature()
        return torch.tensor(self.ret_exact_temperature, device=self.logit_scale.device, dtype=self.logit_scale.dtype)

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

    def _pairwise_retrieval_debug_metrics(
        self,
        exact_pairwise_logits: Optional[torch.Tensor],
        loss_pairwise_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if exact_pairwise_logits is None:
            return {}
        if exact_pairwise_logits.ndim != 2 or exact_pairwise_logits.size(0) != exact_pairwise_logits.size(1):
            return {}

        batch_size = exact_pairwise_logits.size(0)
        base_logit_scale = self.get_logit_scale().to(device=exact_pairwise_logits.device, dtype=exact_pairwise_logits.dtype)
        pairwise_cosine = exact_pairwise_logits / base_logit_scale.clamp_min(1e-12)
        positive = pairwise_cosine.diagonal()
        if batch_size <= 1:
            hardest_negative = torch.zeros_like(positive)
        else:
            negative_mask = ~torch.eye(batch_size, device=pairwise_cosine.device, dtype=torch.bool)
            hardest_negative = pairwise_cosine.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
        margin = positive - hardest_negative

        scaled_logits = exact_pairwise_logits if loss_pairwise_logits is None else loss_pairwise_logits
        positive_logits = scaled_logits.diagonal()
        if batch_size <= 1:
            hardest_negative_logits = torch.zeros_like(positive_logits)
        else:
            negative_mask = ~torch.eye(batch_size, device=scaled_logits.device, dtype=torch.bool)
            hardest_negative_logits = scaled_logits.masked_fill(~negative_mask, float('-inf')).max(dim=1).values

        effective_logit_scale = self.get_logit_scale() if self.ret_exact_temperature is None else torch.reciprocal(self.get_ret_exact_temperature())
        effective_logit_scale = effective_logit_scale.to(device=scaled_logits.device, dtype=scaled_logits.dtype)
        return {
            'image_exact_positive_cosine_mean': positive.mean().detach(),
            'image_exact_positive_cosine_std': positive.std(unbiased=False).detach(),
            'image_exact_hardest_negative_cosine_mean': hardest_negative.mean().detach(),
            'image_exact_hardest_negative_cosine_std': hardest_negative.std(unbiased=False).detach(),
            'image_exact_margin_mean': margin.mean().detach(),
            'image_exact_margin_min': margin.min().detach(),
            'image_exact_positive_logit_mean': positive_logits.mean().detach(),
            'image_exact_hardest_negative_logit_mean': hardest_negative_logits.mean().detach(),
            'exact_pairwise_logit_mean': scaled_logits.mean().detach(),
            'exact_pairwise_logit_std': scaled_logits.std(unbiased=False).detach(),
            'exact_pairwise_logit_scale_or_norm': effective_logit_scale.detach(),
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
        return (1.0 - (source_embeddings * target_embeddings).sum(dim=-1)).mean()

    def diagonal_fidelity_loss(self, surrogate_embeddings: torch.Tensor, exact_embeddings: torch.Tensor) -> torch.Tensor:
        return self.cosine_alignment_loss(surrogate_embeddings, exact_embeddings.detach())

    def effective_support(self, routing_weights: torch.Tensor) -> torch.Tensor:
        return torch.reciprocal(routing_weights.pow(2).sum(dim=-1).clamp_min(1e-12))

    def support_loss(self, routing_weights: Optional[torch.Tensor]) -> torch.Tensor:
        if routing_weights is None or not self.use_loss_support:
            device = self.logit_scale.device if routing_weights is None else routing_weights.device
            return torch.zeros((), device=device)
        # Penalize only degenerate low-support routing without pushing toward full uniform usage.
        effective_support = self.effective_support(routing_weights)
        return (self.support_min - effective_support).clamp_min(0.0).pow(2).mean()

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

    def scale_pairwise_retrieval_logits(self, exact_pairwise_logits: torch.Tensor) -> torch.Tensor:
        if exact_pairwise_logits.ndim != 2 or exact_pairwise_logits.size(0) != exact_pairwise_logits.size(1):
            raise ValueError(
                f'exact_pairwise_logits must have shape [B, B]; received {tuple(exact_pairwise_logits.shape)}.'
            )
        if self.ret_exact_temperature is None:
            return exact_pairwise_logits
        base_logit_scale = self.get_logit_scale().to(device=exact_pairwise_logits.device, dtype=exact_pairwise_logits.dtype)
        pairwise_cosine = exact_pairwise_logits / base_logit_scale.clamp_min(1e-12)
        ret_logit_scale = torch.reciprocal(self.get_ret_exact_temperature()).to(device=exact_pairwise_logits.device, dtype=exact_pairwise_logits.dtype)
        return pairwise_cosine * ret_logit_scale

    def exact_retrieval_loss(self, exact_pairwise_logits: torch.Tensor, transpose: bool = False) -> Dict[str, torch.Tensor]:
        loss_logits = self.scale_pairwise_retrieval_logits(exact_pairwise_logits)
        if transpose:
            loss_logits = loss_logits.t().contiguous()
        targets = torch.arange(loss_logits.size(0), device=loss_logits.device)
        return {
            'loss': F.cross_entropy(loss_logits, targets),
            'logits': loss_logits,
        }

    def forward(
        self,
        image_embeddings: torch.Tensor,
        surrogate_text_embeddings: torch.Tensor,
        exact_text_embeddings: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        exact_pairwise_logits: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if image_embeddings.ndim != 2 or surrogate_text_embeddings.ndim != 2 or exact_text_embeddings.ndim != 2:
            raise ValueError('All amortized objective embeddings must have shape [B, D].')
        if image_embeddings.shape != surrogate_text_embeddings.shape or image_embeddings.shape != exact_text_embeddings.shape:
            raise ValueError('Image, surrogate text, and exact text embeddings must share shape [B, D].')

        batch_size = image_embeddings.size(0)
        if self.use_loss_ret_exact:
            if exact_pairwise_logits is None:
                raise ValueError('exact_pairwise_logits must be provided when use_loss_ret_exact is enabled.')
            if exact_pairwise_logits.ndim != 2 or exact_pairwise_logits.shape != (batch_size, batch_size):
                raise ValueError(
                    'exact_pairwise_logits must have shape [B, B] matching the image batch; '
                    f'got {tuple(exact_pairwise_logits.shape)} for batch size {batch_size}.'
                )

        pids = self._validate_class_labels(pids, batch_size, image_embeddings.device)
        loss_proxy_image_info = self.proxy_loss(image_embeddings, pids)
        loss_proxy_text_info = self.proxy_loss(surrogate_text_embeddings, pids)
        loss_proxy_text_exact_info = self.proxy_loss(exact_text_embeddings, pids)
        loss_ret_exact_image_info = self.exact_retrieval_loss(exact_pairwise_logits, transpose=False) if self.use_loss_ret_exact_image else None
        loss_ret_exact_text_info = self.exact_retrieval_loss(exact_pairwise_logits, transpose=True) if self.use_loss_ret_exact_text else None
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
        loss_ret_exact_image = loss_ret_exact_image_info['loss'] if self.use_loss_ret_exact_image else zero
        loss_ret_exact_text = loss_ret_exact_text_info['loss'] if self.use_loss_ret_exact_text else zero
        loss_ret_exact = loss_ret_exact_image + loss_ret_exact_text
        loss_ret_exact_image_weighted = self.lambda_ret_exact_image * loss_ret_exact_image
        loss_ret_exact_text_weighted = self.lambda_ret_exact_text * loss_ret_exact_text
        loss_ret_exact_weighted = loss_ret_exact_image_weighted + loss_ret_exact_text_weighted
        loss_support = self.support_loss(routing_weights)
        loss_diversity = self.diversity_loss(prototypes)
        loss_balance = self.balance_loss(routing_weights)
        loss_total = (
            loss_proxy_weighted
            + (self.lambda_align * loss_align)
            + (self.lambda_diag * loss_diag)
            + loss_ret_exact_weighted
            + (self.lambda_support * loss_support)
            + (self.lambda_div * loss_diversity)
            + (self.lambda_bal * loss_balance)
        )

        debug_pairwise_logits = None
        if loss_ret_exact_image_info is not None:
            debug_pairwise_logits = loss_ret_exact_image_info['logits']
        elif loss_ret_exact_text_info is not None:
            debug_pairwise_logits = loss_ret_exact_text_info['logits'].t().contiguous()
        exact_debug_metrics = (
            self._pairwise_retrieval_debug_metrics(exact_pairwise_logits, debug_pairwise_logits)
            if self.use_loss_ret_exact and debug_pairwise_logits is not None
            else self._cross_modal_debug_metrics('image_exact', image_embeddings, exact_text_embeddings, pids)
        )
        outputs = {
            'loss_total': loss_total,
            'loss_proxy': loss_proxy,
            'loss_proxy_image': loss_proxy_image,
            'loss_proxy_text': loss_proxy_text,
            'loss_proxy_text_exact': loss_proxy_text_exact,
            'loss_ret_exact': loss_ret_exact,
            'loss_ret_exact_image': loss_ret_exact_image,
            'loss_ret_exact_text': loss_ret_exact_text,
            'loss_align': loss_align,
            'loss_diag': loss_diag,
            'loss_support': loss_support,
            'loss_diversity': loss_diversity,
            'loss_balance': loss_balance,
            'loss_proxy_image_weighted': loss_proxy_image_weighted,
            'loss_proxy_text_weighted': loss_proxy_text_weighted,
            'loss_proxy_text_exact_weighted': loss_proxy_text_exact_weighted,
            'loss_proxy_weighted': loss_proxy_weighted,
            'loss_ret_exact_image_weighted': loss_ret_exact_image_weighted,
            'loss_ret_exact_text_weighted': loss_ret_exact_text_weighted,
            'loss_ret_exact_weighted': loss_ret_exact_weighted,
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
            'use_loss_ret_exact': torch.tensor(float(self.use_loss_ret_exact), device=loss_total.device, dtype=loss_total.dtype),
            'use_loss_ret_exact_image': torch.tensor(float(self.use_loss_ret_exact_image), device=loss_total.device, dtype=loss_total.dtype),
            'use_loss_ret_exact_text': torch.tensor(float(self.use_loss_ret_exact_text), device=loss_total.device, dtype=loss_total.dtype),
            'lambda_ret_exact': torch.tensor(self.lambda_ret_exact, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_ret_exact_image': torch.tensor(self.lambda_ret_exact_image, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_ret_exact_text': torch.tensor(self.lambda_ret_exact_text, device=loss_total.device, dtype=loss_total.dtype),
            'ret_exact_temperature': self.get_ret_exact_temperature().to(device=loss_total.device, dtype=loss_total.dtype),
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
                **self._proxy_debug_metrics('image', loss_proxy_image_info['logits'], pids),
                **self._proxy_debug_metrics('text', loss_proxy_text_info['logits'], pids),
                **self._proxy_debug_metrics('text_exact', loss_proxy_text_exact_info['logits'], pids),
                **self._cross_modal_debug_metrics('image_surrogate', image_embeddings, surrogate_text_embeddings, pids),
                **exact_debug_metrics,
                **self._norm_stats('class_proxy_norm', self.class_proxies.detach()),
                **self._normalized_norm_stats('class_proxy_norm_normalized', self.class_proxies.detach()),
            },
        }
        if return_debug:
            outputs['image_proxy_logits'] = loss_proxy_image_info['logits']
            outputs['text_proxy_logits'] = loss_proxy_text_info['logits']
            outputs['text_exact_proxy_logits'] = loss_proxy_text_exact_info['logits']
            if loss_ret_exact_image_info is not None:
                outputs['ret_exact_logits'] = loss_ret_exact_image_info['logits']
            elif loss_ret_exact_text_info is not None:
                outputs['ret_exact_logits'] = loss_ret_exact_text_info['logits']
            outputs['class_proxies'] = self.class_proxies.detach()
        return outputs
