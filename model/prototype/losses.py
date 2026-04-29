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
        use_loss_diag: bool = True,
        lambda_diag: float = 1.0,
        diag_temperature: float = 0.07,
        use_loss_semantic_pbt: bool = False,
        lambda_semantic_pbt: float = 0.0,
        use_loss_semantic_hardneg_margin: bool = False,
        lambda_semantic_hardneg_margin: float = 0.0,
        semantic_hardneg_margin: float = 0.05,
        semantic_hardneg_eps: float = 1e-8,
        semantic_hardneg_host_global_weight: float = 1.0,
        semantic_hardneg_host_global_tau: float = 0.1,
        use_loss_semantic_hosthard_weighted: bool = False,
        lambda_semantic_hosthard_weighted: float = 0.0,
        semantic_hosthard_margin_ref: float = 0.0,
        semantic_hosthard_tau: float = 0.1,
        semantic_hosthard_eps: float = 1e-8,
        semantic_hosthard_normalize_weights: bool = True,
        prototype_method_role: str = 'retrieval_branch',
        prototype_semantic_enabled: bool = False,
        semantic_structure_enabled: bool = False,
        semantic_feature_space: str = 'prototype_projected',
        semantic_pbt_enabled: bool = True,
        semantic_soft_target_enabled: bool = True,
        semantic_target_temperature: float = 0.01,
        semantic_pred_temperature: float = 0.07,
        semantic_min_cluster_count_for_pbt: float = 1.0,
        semantic_empty_cluster_policy: str = 'skip',
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
        if diag_temperature <= 0:
            raise ValueError('diag_temperature must be positive.')
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
        self.use_loss_diag = bool(use_loss_diag)
        self.lambda_diag = float(lambda_diag)
        self.use_loss_semantic_pbt = bool(use_loss_semantic_pbt)
        self.lambda_semantic_pbt = float(lambda_semantic_pbt)
        self.use_loss_semantic_hardneg_margin = bool(use_loss_semantic_hardneg_margin)
        self.lambda_semantic_hardneg_margin = float(lambda_semantic_hardneg_margin)
        self.semantic_hardneg_margin = float(semantic_hardneg_margin)
        self.semantic_hardneg_eps = float(semantic_hardneg_eps)
        self.semantic_hardneg_host_global_weight = float(semantic_hardneg_host_global_weight)
        self.semantic_hardneg_host_global_tau = float(semantic_hardneg_host_global_tau)
        self.use_loss_semantic_hosthard_weighted = bool(use_loss_semantic_hosthard_weighted)
        self.lambda_semantic_hosthard_weighted = float(lambda_semantic_hosthard_weighted)
        self.semantic_hosthard_margin_ref = float(semantic_hosthard_margin_ref)
        self.semantic_hosthard_tau = float(semantic_hosthard_tau)
        self.semantic_hosthard_eps = float(semantic_hosthard_eps)
        self.semantic_hosthard_normalize_weights = bool(semantic_hosthard_normalize_weights)
        self.prototype_method_role = str(prototype_method_role).lower()
        self.prototype_semantic_enabled = bool(prototype_semantic_enabled)
        self.semantic_structure_enabled = bool(semantic_structure_enabled)
        self.semantic_feature_space = str(semantic_feature_space).lower()
        self.semantic_pbt_enabled = bool(semantic_pbt_enabled)
        self.semantic_soft_target_enabled = bool(semantic_soft_target_enabled)
        self.semantic_target_temperature = float(semantic_target_temperature)
        self.semantic_pred_temperature = float(semantic_pred_temperature)
        self.semantic_min_cluster_count_for_pbt = float(semantic_min_cluster_count_for_pbt)
        self.semantic_empty_cluster_policy = str(semantic_empty_cluster_policy).lower()
        self.proxy_temperature = float(proxy_temperature)
        self.diag_temperature = float(diag_temperature)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)

        if self.semantic_target_temperature <= 0.0:
            raise ValueError('semantic_target_temperature must be positive.')
        if self.semantic_pred_temperature <= 0.0:
            raise ValueError('semantic_pred_temperature must be positive.')
        if self.semantic_min_cluster_count_for_pbt <= 0.0:
            raise ValueError('semantic_min_cluster_count_for_pbt must be positive.')
        if self.semantic_hardneg_margin < 0.0:
            raise ValueError('semantic_hardneg_margin must be non-negative.')
        if self.semantic_hardneg_eps <= 0.0:
            raise ValueError('semantic_hardneg_eps must be positive.')
        if self.semantic_hardneg_host_global_weight < 0.0:
            raise ValueError('semantic_hardneg_host_global_weight must be non-negative.')
        if self.semantic_hardneg_host_global_tau <= 0.0:
            raise ValueError('semantic_hardneg_host_global_tau must be positive.')
        if self.semantic_hosthard_tau <= 0.0:
            raise ValueError('semantic_hosthard_tau must be positive.')
        if self.semantic_hosthard_eps <= 0.0:
            raise ValueError('semantic_hosthard_eps must be positive.')
        if self.semantic_empty_cluster_policy not in {'skip', 'reseed'}:
            raise ValueError(
                f'Unsupported semantic_empty_cluster_policy={self.semantic_empty_cluster_policy!r}. '
                'Allowed values: ["skip", "reseed"].'
            )


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

    def _host_hard_margin_debug_metrics(
        self,
        host_pairwise_logits: Optional[torch.Tensor],
        pids: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(host_pairwise_logits, torch.Tensor):
            return {}
        if host_pairwise_logits.ndim != 2 or host_pairwise_logits.size(0) != host_pairwise_logits.size(1):
            return {}

        batch_size = int(host_pairwise_logits.size(0))
        if batch_size < 2:
            zero = host_pairwise_logits.new_zeros(())
            return {
                'host_margin_mean': zero.detach(),
                'host_margin_min': zero.detach(),
            }

        host_scores = host_pairwise_logits.detach()
        diagonal = host_scores.diagonal()
        if diagonal.numel() != batch_size:
            return {}

        if isinstance(pids, torch.Tensor) and pids.ndim == 1 and pids.numel() == batch_size:
            pids_local = pids.to(device=host_scores.device, dtype=torch.long)
            negative_mask = ~pids_local.view(-1, 1).eq(pids_local.view(1, -1))
        else:
            negative_mask = ~torch.eye(batch_size, device=host_scores.device, dtype=torch.bool)

        if not negative_mask.any(dim=1).all():
            negative_mask = ~torch.eye(batch_size, device=host_scores.device, dtype=torch.bool)
        if not negative_mask.any(dim=1).all():
            zero = host_pairwise_logits.new_zeros(())
            return {
                'host_margin_mean': zero.detach(),
                'host_margin_min': zero.detach(),
            }

        hardest_negative = host_scores.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
        margins = diagonal - hardest_negative
        return {
            'host_margin_mean': margins.mean().detach(),
            'host_margin_min': margins.min().detach(),
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

    def directional_fidelity_loss(self, surrogate_embeddings: torch.Tensor, exact_embeddings: torch.Tensor) -> torch.Tensor:
        return self.symmetric_relative_diagonal_loss(surrogate_embeddings, exact_embeddings)['loss']

    def diagonal_fidelity_loss(self, surrogate_embeddings: torch.Tensor, exact_embeddings: torch.Tensor) -> torch.Tensor:
        return self.symmetric_relative_diagonal_loss(surrogate_embeddings, exact_embeddings)['loss']

    def symmetric_relative_diagonal_loss(
        self,
        surrogate_embeddings: torch.Tensor,
        exact_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if surrogate_embeddings.ndim != 2 or exact_embeddings.ndim != 2:
            raise ValueError('symmetric_relative_diagonal_loss expects [B, D] tensors.')
        if surrogate_embeddings.shape != exact_embeddings.shape:
            raise ValueError(
                'symmetric_relative_diagonal_loss requires matching [B, D] tensor shapes; '
                f'got {tuple(surrogate_embeddings.shape)} and {tuple(exact_embeddings.shape)}.'
            )

        # Relative diagonal fidelity: each surrogate diagonal text embedding must match
        # its own exact teacher embedding more than other in-batch candidates.
        student = F.normalize(surrogate_embeddings.float(), dim=-1)
        teacher = F.normalize(exact_embeddings.detach().float(), dim=-1)
        similarity = student @ teacher.t()
        batch_size = similarity.size(0)
        zero = similarity.new_zeros(())
        zero_loss = similarity.sum() * 0.0
        positive = similarity.diagonal()
        positive_mean = positive.mean() if positive.numel() > 0 else zero
        if batch_size <= 1:
            return {
                'loss': zero_loss,
                'loss_row': zero_loss,
                'loss_col': zero_loss,
                'positive_mean': positive_mean,
                'offdiag_mean': zero,
                'margin': positive_mean,
            }

        targets = torch.arange(batch_size, device=similarity.device, dtype=torch.long)
        logits = similarity / self.diag_temperature
        loss_row = F.cross_entropy(logits, targets)
        loss_col = F.cross_entropy(logits.t(), targets)
        loss = 0.5 * (loss_row + loss_col)
        offdiag_mask = ~torch.eye(batch_size, device=similarity.device, dtype=torch.bool)
        offdiag_mean = similarity[offdiag_mask].mean()
        return {
            'loss': loss,
            'loss_row': loss_row,
            'loss_col': loss_col,
            'positive_mean': positive_mean,
            'offdiag_mean': offdiag_mean,
            'margin': positive_mean - offdiag_mean,
        }

    def effective_support(self, routing_weights: torch.Tensor) -> torch.Tensor:
        return torch.reciprocal(routing_weights.pow(2).sum(dim=-1).clamp_min(1e-12))

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

    def _pairwise_correlation(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        lhs_flat = lhs.detach().float().reshape(-1)
        rhs_flat = rhs.detach().float().reshape(-1)
        lhs_centered = lhs_flat - lhs_flat.mean()
        rhs_centered = rhs_flat - rhs_flat.mean()
        lhs_scale = lhs_centered.pow(2).mean().sqrt()
        rhs_scale = rhs_centered.pow(2).mean().sqrt()
        denom = lhs_scale * rhs_scale
        if denom <= 0:
            return lhs_flat.new_zeros(())
        return (lhs_centered * rhs_centered).mean() / denom

    def _semantic_mode_enabled(self) -> bool:
        return bool(
            self.prototype_method_role == 'semantic_structure'
            and self.prototype_semantic_enabled
            and self.semantic_structure_enabled
            and self.semantic_pbt_enabled
        )

    def _soft_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return (-(targets * log_probs).sum(dim=-1)).mean()

    def _semantic_pbt_loss(
        self,
        *,
        image_student: Optional[torch.Tensor],
        text_student: Optional[torch.Tensor],
        text_teacher: Optional[torch.Tensor],
        base_prototypes: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        reference = image_student
        if not isinstance(reference, torch.Tensor):
            for candidate in (text_student, text_teacher, base_prototypes):
                if isinstance(candidate, torch.Tensor):
                    reference = candidate
                    break
        if not isinstance(reference, torch.Tensor):
            device = self.logit_scale.device
            reference = torch.zeros(1, self.embedding_dim, device=device, dtype=torch.float32)
        zero = reference.new_zeros(())
        outputs = {
            'loss': zero,
            'assignment_entropy_image': zero.detach(),
            'assignment_entropy_teacher': zero.detach(),
            'valid_cluster_count': zero.detach(),
            'empty_cluster_count': zero.detach(),
            'target_entropy': zero.detach(),
            'image_student_probs': None,
            'text_student_probs': None,
            'text_targets': None,
            'image_targets': None,
        }
        if not self._semantic_mode_enabled():
            return outputs
        if not (
            isinstance(image_student, torch.Tensor)
            and isinstance(text_student, torch.Tensor)
            and isinstance(text_teacher, torch.Tensor)
            and isinstance(base_prototypes, torch.Tensor)
        ):
            return outputs
        if image_student.ndim != 2 or text_student.ndim != 2 or text_teacher.ndim != 2 or base_prototypes.ndim != 2:
            return outputs
        if image_student.size(0) == 0 or base_prototypes.size(0) == 0:
            return outputs

        anchors = F.normalize(base_prototypes.detach().float(), dim=-1)
        image_student_norm = F.normalize(image_student.float(), dim=-1)
        text_student_norm = F.normalize(text_student.float(), dim=-1)
        text_teacher_norm = F.normalize(text_teacher.detach().float(), dim=-1)

        teacher_assignment_logits = text_teacher_norm @ anchors.t()
        image_assignment_logits = image_student_norm.detach() @ anchors.t()
        teacher_assignment = torch.softmax(teacher_assignment_logits / self.semantic_target_temperature, dim=-1)
        image_assignment = torch.softmax(image_assignment_logits / self.semantic_target_temperature, dim=-1)

        anchor_similarity = anchors @ anchors.t()
        proto_rel_targets = torch.softmax(anchor_similarity / self.semantic_target_temperature, dim=-1)
        teacher_targets = teacher_assignment @ proto_rel_targets
        image_targets = image_assignment @ proto_rel_targets
        if not self.semantic_soft_target_enabled:
            teacher_targets = F.one_hot(teacher_targets.argmax(dim=-1), num_classes=teacher_targets.size(-1)).to(dtype=teacher_targets.dtype)
            image_targets = F.one_hot(image_targets.argmax(dim=-1), num_classes=image_targets.size(-1)).to(dtype=image_targets.dtype)

        teacher_counts = teacher_assignment.sum(dim=0)
        image_counts = image_assignment.sum(dim=0)
        valid_mask = torch.logical_and(
            teacher_counts >= float(self.semantic_min_cluster_count_for_pbt),
            image_counts >= float(self.semantic_min_cluster_count_for_pbt),
        )
        if not valid_mask.any():
            outputs['empty_cluster_count'] = teacher_counts.new_tensor(float(teacher_counts.numel())).detach()
            return outputs

        if self.semantic_empty_cluster_policy == 'reseed':
            # Lightweight reseed: borrow teacher anchors for invalid clusters so logits remain finite.
            invalid_mask = ~valid_mask
            if invalid_mask.any():
                teacher_counts = teacher_counts.clone()
                image_counts = image_counts.clone()
                teacher_counts[invalid_mask] = 1.0
                image_counts[invalid_mask] = 1.0
                teacher_assignment = teacher_assignment.clone()
                image_assignment = image_assignment.clone()
                teacher_assignment[:, invalid_mask] = teacher_assignment[:, valid_mask].mean(dim=-1, keepdim=True)
                image_assignment[:, invalid_mask] = image_assignment[:, valid_mask].mean(dim=-1, keepdim=True)

        teacher_centroids = (teacher_assignment.detach().t() @ text_student_norm.detach()) / teacher_counts.clamp_min(1e-12).unsqueeze(-1)
        image_centroids = (image_assignment.detach().t() @ image_student_norm.detach()) / image_counts.clamp_min(1e-12).unsqueeze(-1)
        teacher_centroids = F.normalize(teacher_centroids, dim=-1)
        image_centroids = F.normalize(image_centroids, dim=-1)

        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        teacher_targets_valid = teacher_targets[:, valid_indices].detach()
        image_targets_valid = image_targets[:, valid_indices].detach()
        teacher_centroids_valid = teacher_centroids[valid_indices].detach()
        image_centroids_valid = image_centroids[valid_indices].detach()

        teacher_logits = (text_student_norm @ teacher_centroids_valid.t()) / self.semantic_pred_temperature
        image_logits = (image_student_norm @ image_centroids_valid.t()) / self.semantic_pred_temperature
        teacher_targets_valid = teacher_targets_valid / teacher_targets_valid.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        image_targets_valid = image_targets_valid / image_targets_valid.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        loss_teacher = self._soft_cross_entropy(teacher_logits, teacher_targets_valid)
        loss_image = self._soft_cross_entropy(image_logits, image_targets_valid)
        loss_semantic = 0.5 * (loss_teacher + loss_image)

        teacher_entropy = -(teacher_assignment * teacher_assignment.clamp_min(1e-12).log()).sum(dim=-1).mean()
        image_entropy = -(image_assignment * image_assignment.clamp_min(1e-12).log()).sum(dim=-1).mean()
        target_entropy = -(
            0.5 * (teacher_targets_valid + image_targets_valid)
            * (0.5 * (teacher_targets_valid + image_targets_valid)).clamp_min(1e-12).log()
        ).sum(dim=-1).mean()

        outputs.update(
            {
                'loss': loss_semantic,
                'assignment_entropy_image': image_entropy.detach(),
                'assignment_entropy_teacher': teacher_entropy.detach(),
                'valid_cluster_count': valid_mask.sum().to(dtype=loss_semantic.dtype).detach(),
                'empty_cluster_count': (~valid_mask).sum().to(dtype=loss_semantic.dtype).detach(),
                'target_entropy': target_entropy.detach(),
                'image_student_probs': torch.softmax(image_logits, dim=-1),
                'text_student_probs': torch.softmax(teacher_logits, dim=-1),
                'text_targets': teacher_targets_valid,
                'image_targets': image_targets_valid,
            }
        )
        return outputs

    def _semantic_hardneg_margin_loss(
        self,
        *,
        semantic_info: Dict[str, torch.Tensor],
        host_pairwise_logits: Optional[torch.Tensor],
        host_global_pairwise_logits: Optional[torch.Tensor],
        pids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_probs = semantic_info.get('image_student_probs')
        text_probs = semantic_info.get('text_student_probs')
        text_targets = semantic_info.get('text_targets')
        image_targets = semantic_info.get('image_targets')
        if not all(isinstance(value, torch.Tensor) for value in (image_probs, text_probs, text_targets, image_targets)):
            raise ValueError(
                'Semantic hard-negative margin requires semantic student probabilities/targets from semantic PBT path, '
                'but they are unavailable. Ensure semantic tensors are produced in the current training step.'
            )
        if not isinstance(host_pairwise_logits, torch.Tensor):
            raise ValueError(
                'Semantic hard-negative margin requires host_pairwise_logits from HostCore, but it is unavailable.'
            )
        if host_pairwise_logits.ndim != 2:
            raise ValueError(
                f'Semantic hard-negative margin requires host_pairwise_logits with shape [B, B], got {tuple(host_pairwise_logits.shape)}.'
            )
        batch_size = int(image_probs.size(0))
        if host_pairwise_logits.size(0) != batch_size or host_pairwise_logits.size(1) != batch_size:
            raise ValueError(
                'Semantic hard-negative margin requires host_pairwise_logits shape [B, B] aligned with semantic tensors; '
                f'got host={tuple(host_pairwise_logits.shape)} semantic_batch={batch_size}.'
            )
        if batch_size < 2:
            zero = image_probs.new_zeros(())
            return {
                'loss': zero,
                'loss_image': zero,
                'loss_text': zero,
                'pos_img_mean': zero.detach(),
                'neg_img_mean': zero.detach(),
                'pos_txt_mean': zero.detach(),
                'neg_txt_mean': zero.detach(),
                'pos_host_global_mean': zero.detach(),
                'neg_host_global_mean': zero.detach(),
                'margin_host_global_mean': zero.detach(),
                'bridge_weight_mean': zero.detach(),
                'bridge_loss_mean': zero.detach(),
                'valid_i2t_frac': zero.detach(),
                'valid_t2i_frac': zero.detach(),
                'same_id_i2t_rate': zero.detach(),
                'same_id_t2i_rate': zero.detach(),
            }

        if image_probs.ndim != 2 or text_probs.ndim != 2 or text_targets.ndim != 2 or image_targets.ndim != 2:
            raise ValueError('Semantic hard-negative margin expects [B, K] tensors for probs and targets.')
        if image_probs.shape != text_targets.shape:
            raise ValueError(
                'Semantic hard-negative margin requires image_student_probs and text_targets to share shape [B, K]; '
                f'got {tuple(image_probs.shape)} and {tuple(text_targets.shape)}.'
            )
        if text_probs.shape != image_targets.shape:
            raise ValueError(
                'Semantic hard-negative margin requires text_student_probs and image_targets to share shape [B, K]; '
                f'got {tuple(text_probs.shape)} and {tuple(image_targets.shape)}.'
            )

        if pids.ndim != 1 or pids.numel() != batch_size:
            raise ValueError(
                f'Semantic hard-negative margin requires pids with shape [B], got {tuple(pids.shape)} for batch {batch_size}.'
            )

        # Host score is used only for discrete hardest-negative index selection.
        host_scores = host_pairwise_logits.detach().to(device=image_probs.device, dtype=image_probs.dtype)
        labels = pids.to(device=host_scores.device, dtype=torch.long)
        same_identity = labels.view(batch_size, 1).eq(labels.view(1, batch_size))
        valid_neg = ~same_identity

        has_valid_i2t = valid_neg.any(dim=1)
        has_valid_t2i = valid_neg.any(dim=0)
        if host_scores.dtype in (torch.float16, torch.bfloat16):
            neg_inf = -1e4
        else:
            neg_inf = -1e9
        masked_scores = host_scores.masked_fill(~valid_neg, neg_inf)
        hardest_neg_caption = masked_scores.argmax(dim=1)
        hardest_neg_image = masked_scores.argmax(dim=0)

        log_p_i_from_t = torch.log(image_probs.clamp_min(self.semantic_hardneg_eps))
        log_p_t_from_i = torch.log(text_probs.clamp_min(self.semantic_hardneg_eps))

        pos_img_all = (text_targets * log_p_i_from_t).sum(dim=-1)
        neg_img_all = (text_targets.index_select(0, hardest_neg_caption) * log_p_i_from_t).sum(dim=-1)
        pos_img = pos_img_all[has_valid_i2t]
        neg_img = neg_img_all[has_valid_i2t]
        if has_valid_i2t.any():
            loss_img_sem = F.relu(self.semantic_hardneg_margin - pos_img + neg_img)
        else:
            loss_img_sem = image_probs.new_zeros((0,))

        pos_txt_all = (image_targets * log_p_t_from_i).sum(dim=-1)
        neg_txt_all = (image_targets.index_select(0, hardest_neg_image) * log_p_t_from_i).sum(dim=-1)
        pos_txt = pos_txt_all[has_valid_t2i]
        neg_txt = neg_txt_all[has_valid_t2i]
        if has_valid_t2i.any():
            loss_txt_sem = F.relu(self.semantic_hardneg_margin - pos_txt + neg_txt)
        else:
            loss_txt_sem = image_probs.new_zeros((0,))

        bridge_loss_img = torch.zeros_like(loss_img_sem)
        bridge_loss_txt = torch.zeros_like(loss_txt_sem)
        host_global_pos = pos_img.new_zeros(pos_img.shape)
        host_global_neg = neg_img.new_zeros(neg_img.shape)
        host_global_margin = pos_img.new_zeros(pos_img.shape)
        bridge_weight = pos_img.new_zeros(pos_img.shape)
        if (
            isinstance(host_global_pairwise_logits, torch.Tensor)
            and host_global_pairwise_logits.ndim == 2
            and host_global_pairwise_logits.size(0) == batch_size
            and host_global_pairwise_logits.size(1) == batch_size
            and self.semantic_hardneg_host_global_weight > 0.0
        ):
            host_global_scores = host_global_pairwise_logits.to(device=image_probs.device, dtype=image_probs.dtype)
            row_index = torch.arange(batch_size, device=host_global_scores.device)
            host_global_pos_all = host_global_scores.diagonal()
            host_global_neg_img_all = host_global_scores[row_index, hardest_neg_caption]
            host_global_neg_txt_all = host_global_scores[hardest_neg_image, row_index]
            host_global_margin_img_all = host_global_pos_all - host_global_neg_img_all
            host_global_margin_txt_all = host_global_pos_all - host_global_neg_txt_all

            if has_valid_i2t.any():
                host_global_pos = host_global_pos_all[has_valid_i2t]
                host_global_neg_img = host_global_neg_img_all[has_valid_i2t]
                host_global_margin_img = host_global_margin_img_all[has_valid_i2t]
                semantic_gap_img = (pos_img - neg_img).detach()
                bridge_weight_img = torch.sigmoid((self.semantic_hardneg_margin - semantic_gap_img) / self.semantic_hardneg_host_global_tau)
                bridge_loss_img = bridge_weight_img * F.relu(self.semantic_hardneg_margin - host_global_margin_img)
            if has_valid_t2i.any():
                host_global_neg_txt = host_global_neg_txt_all[has_valid_t2i]
                host_global_margin_txt = host_global_margin_txt_all[has_valid_t2i]
                semantic_gap_txt = (pos_txt - neg_txt).detach()
                bridge_weight_txt = torch.sigmoid((self.semantic_hardneg_margin - semantic_gap_txt) / self.semantic_hardneg_host_global_tau)
                bridge_loss_txt = bridge_weight_txt * F.relu(self.semantic_hardneg_margin - host_global_margin_txt)
            if has_valid_i2t.any() and has_valid_t2i.any():
                bridge_weight = 0.5 * (bridge_weight_img + bridge_weight_txt)
                host_global_neg = 0.5 * (host_global_neg_img + host_global_neg_txt)
                host_global_margin = 0.5 * (host_global_margin_img + host_global_margin_txt)
            elif has_valid_i2t.any():
                bridge_weight = bridge_weight_img
                host_global_neg = host_global_neg_img
                host_global_margin = host_global_margin_img
            elif has_valid_t2i.any():
                bridge_weight = bridge_weight_txt
                host_global_neg = host_global_neg_txt
                host_global_margin = host_global_margin_txt

        if loss_img_sem.numel() > 0:
            loss_img = (
                loss_img_sem
                + self.semantic_hardneg_host_global_weight * bridge_loss_img
            ).mean()
        else:
            loss_img = image_probs.new_zeros(())
        if loss_txt_sem.numel() > 0:
            loss_txt = (
                loss_txt_sem
                + self.semantic_hardneg_host_global_weight * bridge_loss_txt
            ).mean()
        else:
            loss_txt = image_probs.new_zeros(())
        bridge_loss_img_mean = bridge_loss_img.mean() if bridge_loss_img.numel() > 0 else image_probs.new_zeros(())
        bridge_loss_txt_mean = bridge_loss_txt.mean() if bridge_loss_txt.numel() > 0 else image_probs.new_zeros(())
        bridge_loss_mean = 0.5 * (bridge_loss_img_mean + bridge_loss_txt_mean)

        selected_i = has_valid_i2t.nonzero(as_tuple=False).squeeze(-1)
        selected_j = hardest_neg_caption[selected_i]
        selected_k = has_valid_t2i.nonzero(as_tuple=False).squeeze(-1)
        selected_i_txt = hardest_neg_image[selected_k]
        same_id_i2t_rate = (
            labels[selected_i].eq(labels[selected_j]).float().mean()
            if selected_i.numel() > 0
            else image_probs.new_zeros(())
        )
        same_id_t2i_rate = (
            labels[selected_k].eq(labels[selected_i_txt]).float().mean()
            if selected_k.numel() > 0
            else image_probs.new_zeros(())
        )

        return {
            'loss': 0.5 * (loss_img + loss_txt),
            'loss_image': loss_img,
            'loss_text': loss_txt,
            'pos_img_mean': (pos_img.mean() if pos_img.numel() > 0 else image_probs.new_zeros(())).detach(),
            'neg_img_mean': (neg_img.mean() if neg_img.numel() > 0 else image_probs.new_zeros(())).detach(),
            'pos_txt_mean': (pos_txt.mean() if pos_txt.numel() > 0 else image_probs.new_zeros(())).detach(),
            'neg_txt_mean': (neg_txt.mean() if neg_txt.numel() > 0 else image_probs.new_zeros(())).detach(),
            'pos_host_global_mean': (host_global_pos.mean() if host_global_pos.numel() > 0 else image_probs.new_zeros(())).detach(),
            'neg_host_global_mean': (host_global_neg.mean() if host_global_neg.numel() > 0 else image_probs.new_zeros(())).detach(),
            'margin_host_global_mean': (host_global_margin.mean() if host_global_margin.numel() > 0 else image_probs.new_zeros(())).detach(),
            'bridge_weight_mean': (bridge_weight.mean() if bridge_weight.numel() > 0 else image_probs.new_zeros(())).detach(),
            'bridge_loss_mean': bridge_loss_mean.detach(),
            'valid_i2t_frac': has_valid_i2t.float().mean().detach(),
            'valid_t2i_frac': has_valid_t2i.float().mean().detach(),
            'same_id_i2t_rate': same_id_i2t_rate.detach(),
            'same_id_t2i_rate': same_id_t2i_rate.detach(),
        }

    def _semantic_hosthard_weighted_loss(
        self,
        *,
        semantic_info: Dict[str, torch.Tensor],
        host_pairwise_logits: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        image_probs = semantic_info.get('image_student_probs')
        text_probs = semantic_info.get('text_student_probs')
        text_targets = semantic_info.get('text_targets')
        image_targets = semantic_info.get('image_targets')
        if not all(isinstance(value, torch.Tensor) for value in (image_probs, text_probs, text_targets, image_targets)):
            raise ValueError(
                'Semantic host-hardness weighted loss requires semantic student probabilities/targets from semantic '
                'PBT path, but they are unavailable. Ensure semantic tensors are produced in the current training step.'
            )
        if not isinstance(host_pairwise_logits, torch.Tensor):
            raise ValueError(
                'Semantic host-hardness weighted loss requires host_pairwise_logits from HostCore, but it is unavailable.'
            )
        if host_pairwise_logits.ndim != 2:
            raise ValueError(
                'Semantic host-hardness weighted loss requires host_pairwise_logits with shape [B, B], '
                f'got {tuple(host_pairwise_logits.shape)}.'
            )
        if image_probs.ndim != 2 or text_probs.ndim != 2 or text_targets.ndim != 2 or image_targets.ndim != 2:
            raise ValueError('Semantic host-hardness weighted loss expects [B, K] tensors for probs and targets.')
        if image_probs.shape != text_targets.shape:
            raise ValueError(
                'Semantic host-hardness weighted loss requires image_student_probs and text_targets to share shape [B, K]; '
                f'got {tuple(image_probs.shape)} and {tuple(text_targets.shape)}.'
            )
        if text_probs.shape != image_targets.shape:
            raise ValueError(
                'Semantic host-hardness weighted loss requires text_student_probs and image_targets to share shape [B, K]; '
                f'got {tuple(text_probs.shape)} and {tuple(image_targets.shape)}.'
            )

        batch_size = int(image_probs.size(0))
        if host_pairwise_logits.size(0) != batch_size or host_pairwise_logits.size(1) != batch_size:
            raise ValueError(
                'Semantic host-hardness weighted loss requires host_pairwise_logits shape [B, B] aligned with semantic '
                f'tensors; got host={tuple(host_pairwise_logits.shape)} semantic_batch={batch_size}.'
            )
        if batch_size < 2:
            zero = image_probs.new_zeros(())
            return {
                'loss': zero,
                'loss_image': zero,
                'loss_text': zero,
                'weight_mean': zero.detach(),
                'weight_max': zero.detach(),
                'margin_row_mean': zero.detach(),
                'margin_col_mean': zero.detach(),
                'margin_mean': zero.detach(),
            }

        host_scores = host_pairwise_logits.detach().to(device=image_probs.device, dtype=image_probs.dtype)
        offdiag_mask = ~torch.eye(batch_size, device=host_scores.device, dtype=torch.bool)
        masked_scores = host_scores.masked_fill(~offdiag_mask, float('-inf'))
        diagonal = host_scores.diagonal()
        margin_row = diagonal - masked_scores.max(dim=1).values
        margin_col = diagonal - masked_scores.max(dim=0).values
        margin = torch.minimum(margin_row, margin_col)

        weights = torch.sigmoid((self.semantic_hosthard_margin_ref - margin) / self.semantic_hosthard_tau)
        if self.semantic_hosthard_normalize_weights:
            weights = weights / weights.mean().clamp_min(self.semantic_hosthard_eps)

        log_p_i_from_t = torch.log(image_probs.clamp_min(self.semantic_hosthard_eps))
        log_p_t_from_i = torch.log(text_probs.clamp_min(self.semantic_hosthard_eps))
        ce_img = -(text_targets * log_p_i_from_t).sum(dim=-1)
        ce_txt = -(image_targets * log_p_t_from_i).sum(dim=-1)

        loss_image = (weights * ce_img).mean()
        loss_text = (weights * ce_txt).mean()
        return {
            'loss': 0.5 * (loss_image + loss_text),
            'loss_image': loss_image,
            'loss_text': loss_text,
            'weight_mean': weights.mean().detach(),
            'weight_max': weights.max().detach(),
            'margin_row_mean': margin_row.mean().detach(),
            'margin_col_mean': margin_col.mean().detach(),
            'margin_mean': margin.mean().detach(),
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
        host_pairwise_logits: Optional[torch.Tensor] = None,
        host_global_pairwise_logits: Optional[torch.Tensor] = None,
        semantic_image_student_embeddings: Optional[torch.Tensor] = None,
        semantic_text_student_embeddings: Optional[torch.Tensor] = None,
        semantic_text_teacher_embeddings: Optional[torch.Tensor] = None,
        semantic_base_prototypes: Optional[torch.Tensor] = None,
        diag_loss_scale: Optional[float] = None,
        semantic_pbt_loss_scale: Optional[float] = None,
        semantic_hardneg_margin_loss_scale: Optional[float] = None,
        semantic_hosthard_weighted_loss_scale: Optional[float] = None,
        # Backward-compatible aliases for older call sites.
        prototype_loss_scale: Optional[float] = None,
        semantic_loss_scale: Optional[float] = None,
        return_debug: bool = False,
        disable_proxy_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if image_embeddings.ndim != 2 or surrogate_text_embeddings.ndim != 2 or exact_text_embeddings.ndim != 2:
            raise ValueError('All amortized objective embeddings must have shape [B, D].')
        if image_embeddings.shape != surrogate_text_embeddings.shape or image_embeddings.shape != exact_text_embeddings.shape:
            raise ValueError('Image, surrogate text, and exact text embeddings must share shape [B, D].')

        # Preserve a stable reference for downstream logic that may need host logits.
        # This guards against accidental local deletion/regression of `host_pairwise_logits`.
        host_pairwise_logits_ref = host_pairwise_logits
        host_global_pairwise_logits_ref = host_global_pairwise_logits
        batch_size = image_embeddings.size(0)

        pids_for_metrics = pids
        if pids_for_metrics is None:
            pids_for_metrics = torch.arange(batch_size, device=image_embeddings.device, dtype=torch.long)
        else:
            pids_for_metrics = pids_for_metrics.to(device=image_embeddings.device, dtype=torch.long)
            if pids_for_metrics.ndim != 1 or pids_for_metrics.numel() != batch_size:
                raise ValueError(f'pids must have shape [B], received {tuple(pids_for_metrics.shape)} for batch size {batch_size}.')

        zero = image_embeddings.new_zeros(())
        if diag_loss_scale is not None:
            resolved_diag_scale = float(diag_loss_scale)
        else:
            resolved_diag_scale = 1.0
        if resolved_diag_scale < 0.0:
            resolved_diag_scale = 0.0

        if semantic_pbt_loss_scale is not None:
            resolved_semantic_pbt_scale = float(semantic_pbt_loss_scale)
        elif prototype_loss_scale is not None:
            resolved_semantic_pbt_scale = float(prototype_loss_scale)
        elif semantic_loss_scale is not None:
            resolved_semantic_pbt_scale = float(semantic_loss_scale)
        else:
            resolved_semantic_pbt_scale = 1.0
        if resolved_semantic_pbt_scale < 0.0:
            resolved_semantic_pbt_scale = 0.0

        if semantic_hardneg_margin_loss_scale is not None:
            resolved_semantic_hardneg_margin_scale = float(semantic_hardneg_margin_loss_scale)
        else:
            resolved_semantic_hardneg_margin_scale = resolved_semantic_pbt_scale
        if resolved_semantic_hardneg_margin_scale < 0.0:
            resolved_semantic_hardneg_margin_scale = 0.0

        if semantic_hosthard_weighted_loss_scale is not None:
            resolved_semantic_hosthard_weighted_scale = float(semantic_hosthard_weighted_loss_scale)
        else:
            resolved_semantic_hosthard_weighted_scale = resolved_semantic_pbt_scale
        if resolved_semantic_hosthard_weighted_scale < 0.0:
            resolved_semantic_hosthard_weighted_scale = 0.0

        should_compute_diag = self.use_loss_diag and resolved_diag_scale > 0.0
        if should_compute_diag:
            dir_info = self.symmetric_relative_diagonal_loss(surrogate_text_embeddings, exact_text_embeddings)
            loss_dir = dir_info['loss']
        else:
            loss_dir = zero
            dir_info = {
                'loss_row': zero,
                'loss_col': zero,
                'positive_mean': zero,
                'offdiag_mean': zero,
                'margin': zero,
            }
        should_compute_semantic_pbt = self.use_loss_semantic_pbt and resolved_semantic_pbt_scale > 0.0
        should_compute_semantic_hardneg_margin = (
            self.use_loss_semantic_hardneg_margin and resolved_semantic_hardneg_margin_scale > 0.0
        )
        should_compute_semantic_hosthard_weighted = (
            self.use_loss_semantic_hosthard_weighted
            and resolved_semantic_hosthard_weighted_scale > 0.0
            and abs(self.lambda_semantic_hosthard_weighted) > 0.0
        )
        if should_compute_semantic_hosthard_weighted and not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (
                semantic_image_student_embeddings,
                semantic_text_student_embeddings,
                semantic_text_teacher_embeddings,
                semantic_base_prototypes,
            )
        ):
            raise ValueError(
                'Semantic host-hardness weighted loss requires semantic_image_student_embeddings, '
                'semantic_text_student_embeddings, semantic_text_teacher_embeddings, and semantic_base_prototypes tensors.'
            )
        need_semantic_artifacts = (
            should_compute_semantic_pbt
            or should_compute_semantic_hardneg_margin
            or should_compute_semantic_hosthard_weighted
        )
        if need_semantic_artifacts:
            semantic_info = self._semantic_pbt_loss(
                image_student=semantic_image_student_embeddings,
                text_student=semantic_text_student_embeddings,
                text_teacher=semantic_text_teacher_embeddings,
                base_prototypes=semantic_base_prototypes,
            )
            loss_semantic_pbt = semantic_info['loss'] if should_compute_semantic_pbt else zero
        else:
            semantic_info = self._semantic_pbt_loss(
                image_student=None,
                text_student=None,
                text_teacher=None,
                base_prototypes=None,
            )
            loss_semantic_pbt = zero
        if should_compute_semantic_hardneg_margin:
            hardneg_artifacts_ready = all(
                isinstance(semantic_info.get(key), torch.Tensor)
                for key in ('image_student_probs', 'text_student_probs', 'text_targets', 'image_targets')
            )
            valid_cluster_count = semantic_info.get('valid_cluster_count')
            valid_cluster_value = (
                float(valid_cluster_count.detach().float().item())
                if isinstance(valid_cluster_count, torch.Tensor) and valid_cluster_count.numel() == 1
                else 0.0
            )
            if hardneg_artifacts_ready:
                hardneg_info = self._semantic_hardneg_margin_loss(
                    semantic_info=semantic_info,
                    host_pairwise_logits=host_pairwise_logits_ref,
                    host_global_pairwise_logits=host_global_pairwise_logits_ref,
                    pids=pids_for_metrics,
                )
                loss_semantic_hardneg_margin = hardneg_info['loss']
                loss_semantic_hardneg_margin_image = hardneg_info['loss_image']
                loss_semantic_hardneg_margin_text = hardneg_info['loss_text']
            elif valid_cluster_value <= 0.0:
                # No valid semantic clusters this step: keep training stable by skipping only this term.
                loss_semantic_hardneg_margin = zero
                loss_semantic_hardneg_margin_image = zero
                loss_semantic_hardneg_margin_text = zero
                hardneg_info = {
                    'pos_img_mean': zero.detach(),
                    'neg_img_mean': zero.detach(),
                    'pos_txt_mean': zero.detach(),
                    'neg_txt_mean': zero.detach(),
                    'pos_host_global_mean': zero.detach(),
                    'neg_host_global_mean': zero.detach(),
                    'margin_host_global_mean': zero.detach(),
                    'bridge_weight_mean': zero.detach(),
                    'bridge_loss_mean': zero.detach(),
                    'valid_i2t_frac': zero.detach(),
                    'valid_t2i_frac': zero.detach(),
                    'same_id_i2t_rate': zero.detach(),
                    'same_id_t2i_rate': zero.detach(),
                }
            else:
                raise ValueError(
                    'Semantic hard-negative margin was enabled, but semantic probabilities/targets are unavailable '
                    'despite positive valid_cluster_count. This indicates a semantic-loss wiring bug.'
                )
        else:
            loss_semantic_hardneg_margin = zero
            loss_semantic_hardneg_margin_image = zero
            loss_semantic_hardneg_margin_text = zero
            hardneg_info = {
                'pos_img_mean': zero.detach(),
                'neg_img_mean': zero.detach(),
                'pos_txt_mean': zero.detach(),
                'neg_txt_mean': zero.detach(),
                'pos_host_global_mean': zero.detach(),
                'neg_host_global_mean': zero.detach(),
                'margin_host_global_mean': zero.detach(),
                'bridge_weight_mean': zero.detach(),
                'bridge_loss_mean': zero.detach(),
                'valid_i2t_frac': zero.detach(),
                'valid_t2i_frac': zero.detach(),
                'same_id_i2t_rate': zero.detach(),
                'same_id_t2i_rate': zero.detach(),
            }

        if should_compute_semantic_hosthard_weighted:
            hosthard_artifacts_ready = all(
                isinstance(semantic_info.get(key), torch.Tensor)
                for key in ('image_student_probs', 'text_student_probs', 'text_targets', 'image_targets')
            )
            valid_cluster_count = semantic_info.get('valid_cluster_count')
            valid_cluster_value = (
                float(valid_cluster_count.detach().float().item())
                if isinstance(valid_cluster_count, torch.Tensor) and valid_cluster_count.numel() == 1
                else 0.0
            )
            if hosthard_artifacts_ready:
                hosthard_info = self._semantic_hosthard_weighted_loss(
                    semantic_info=semantic_info,
                    host_pairwise_logits=host_pairwise_logits_ref,
                )
                loss_semantic_hosthard_weighted = hosthard_info['loss']
                loss_semantic_hosthard_weighted_image = hosthard_info['loss_image']
                loss_semantic_hosthard_weighted_text = hosthard_info['loss_text']
            elif valid_cluster_value <= 0.0:
                loss_semantic_hosthard_weighted = zero
                loss_semantic_hosthard_weighted_image = zero
                loss_semantic_hosthard_weighted_text = zero
                hosthard_info = {
                    'weight_mean': zero.detach(),
                    'weight_max': zero.detach(),
                    'margin_row_mean': zero.detach(),
                    'margin_col_mean': zero.detach(),
                    'margin_mean': zero.detach(),
                }
            else:
                raise ValueError(
                    'Semantic host-hardness weighted loss was enabled, but semantic probabilities/targets are unavailable '
                    'despite positive valid_cluster_count. This indicates a semantic-loss wiring bug.'
                )
        else:
            loss_semantic_hosthard_weighted = zero
            loss_semantic_hosthard_weighted_image = zero
            loss_semantic_hosthard_weighted_text = zero
            hosthard_info = {
                'weight_mean': zero.detach(),
                'weight_max': zero.detach(),
                'margin_row_mean': zero.detach(),
                'margin_col_mean': zero.detach(),
                'margin_mean': zero.detach(),
            }

        loss_semantic_pbt_weighted = self.lambda_semantic_pbt * resolved_semantic_pbt_scale * loss_semantic_pbt
        loss_semantic_hardneg_margin_weighted = (
            self.lambda_semantic_hardneg_margin
            * resolved_semantic_hardneg_margin_scale
            * loss_semantic_hardneg_margin
        )
        loss_semantic_hosthard_weighted_weighted = (
            self.lambda_semantic_hosthard_weighted
            * resolved_semantic_hosthard_weighted_scale
            * loss_semantic_hosthard_weighted
        )
        loss_diversity = zero
        loss_balance = zero
        loss_dir_weighted = resolved_diag_scale * self.lambda_diag * loss_dir
        loss_total = (
            loss_semantic_pbt_weighted
            + loss_semantic_hardneg_margin_weighted
            + loss_semantic_hosthard_weighted_weighted
            + loss_dir_weighted
        )

        support_values = None
        diag_pos_cosine_mean = zero.detach()
        diag_hardneg_cosine_mean = zero.detach()
        diag_gap_margin_mean = zero.detach()
        routing_support_mean = support_values.mean().detach() if support_values is not None else zero.detach()
        routing_support_std = support_values.std(unbiased=False).detach() if support_values is not None else zero.detach()
        host_margin_metrics = self._host_hard_margin_debug_metrics(host_pairwise_logits_ref, pids_for_metrics)
        host_margin_mean = host_margin_metrics.get('host_margin_mean', zero.detach())
        host_margin_min = host_margin_metrics.get('host_margin_min', zero.detach())
        host_weight_mean = zero.detach()
        host_weight_std = zero.detach()
        proto_score_mean = zero.detach()
        proto_diag_mean = zero.detach()
        if isinstance(host_pairwise_logits_ref, torch.Tensor) and isinstance(surrogate_pairwise_logits, torch.Tensor) and host_pairwise_logits_ref.shape == surrogate_pairwise_logits.shape:
            proto_host_score_corr = self._pairwise_correlation(surrogate_pairwise_logits, host_pairwise_logits_ref).detach()
        else:
            proto_host_score_corr = zero.detach()

        outputs = {
            'loss_total': loss_total,
            'loss_proto': loss_total,
            'loss_semantic_pbt': loss_semantic_pbt,
            'loss_semantic_hardneg_margin': loss_semantic_hardneg_margin,
            'loss_semantic_hardneg_margin_image': loss_semantic_hardneg_margin_image,
            'loss_semantic_hardneg_margin_text': loss_semantic_hardneg_margin_text,
            'loss_semantic_hosthard_weighted': loss_semantic_hosthard_weighted,
            'loss_semantic_hosthard_weighted_image': loss_semantic_hosthard_weighted_image,
            'loss_semantic_hosthard_weighted_text': loss_semantic_hosthard_weighted_text,
            'loss_diag': loss_dir,
            'loss_diversity': loss_diversity,
            'loss_balance': loss_balance,
            'loss_semantic_pbt_weighted': loss_semantic_pbt_weighted,
            'loss_semantic_hardneg_margin_weighted': loss_semantic_hardneg_margin_weighted,
            'loss_semantic_hosthard_weighted_weighted': loss_semantic_hosthard_weighted_weighted,
            'loss_diag_weighted': loss_dir_weighted,
            'loss_diversity_weighted': zero,
            'loss_balance_weighted': zero,
            'use_loss_semantic_pbt': torch.tensor(float(self.use_loss_semantic_pbt), device=loss_total.device, dtype=loss_total.dtype),
            'lambda_semantic_pbt': torch.tensor(self.lambda_semantic_pbt, device=loss_total.device, dtype=loss_total.dtype),
            'use_loss_semantic_hardneg_margin': torch.tensor(
                float(self.use_loss_semantic_hardneg_margin),
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'lambda_semantic_hardneg_margin': torch.tensor(
                self.lambda_semantic_hardneg_margin,
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'use_loss_semantic_hosthard_weighted': torch.tensor(
                float(self.use_loss_semantic_hosthard_weighted),
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'lambda_semantic_hosthard_weighted': torch.tensor(
                self.lambda_semantic_hosthard_weighted,
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'semantic_hardneg_margin': torch.tensor(
                self.semantic_hardneg_margin,
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'semantic_hardneg_eps': torch.tensor(
                self.semantic_hardneg_eps,
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'semantic_hardneg_host_global_weight': torch.tensor(
                self.semantic_hardneg_host_global_weight,
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'semantic_hardneg_host_global_tau': torch.tensor(
                self.semantic_hardneg_host_global_tau,
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'semantic_hosthard_margin_ref': torch.tensor(
                self.semantic_hosthard_margin_ref,
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'semantic_hosthard_tau': torch.tensor(
                self.semantic_hosthard_tau,
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'semantic_hosthard_eps': torch.tensor(
                self.semantic_hosthard_eps,
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'semantic_hosthard_normalize_weights': torch.tensor(
                float(self.semantic_hosthard_normalize_weights),
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'prototype_loss_scale': torch.tensor(resolved_semantic_pbt_scale, device=loss_total.device, dtype=loss_total.dtype),
            'prototype_loss_ramp_scale': torch.tensor(resolved_semantic_pbt_scale, device=loss_total.device, dtype=loss_total.dtype),
            'loss_diag_scale': torch.tensor(resolved_diag_scale, device=loss_total.device, dtype=loss_total.dtype),
            'loss_semantic_pbt_scale': torch.tensor(resolved_semantic_pbt_scale, device=loss_total.device, dtype=loss_total.dtype),
            'loss_semantic_hardneg_margin_scale': torch.tensor(
                resolved_semantic_hardneg_margin_scale,
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            'loss_semantic_hosthard_weighted_scale': torch.tensor(
                resolved_semantic_hosthard_weighted_scale,
                device=loss_total.device,
                dtype=loss_total.dtype,
            ),
            # Backward-compatible alias for existing logs/consumers.
            'semantic_loss_scale': torch.tensor(resolved_semantic_pbt_scale, device=loss_total.device, dtype=loss_total.dtype),
            'use_loss_diag': torch.tensor(float(self.use_loss_diag), device=loss_total.device, dtype=loss_total.dtype),
            'lambda_diag': torch.tensor(self.lambda_diag, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_div': torch.tensor(self.lambda_div, device=loss_total.device, dtype=loss_total.dtype),
            'lambda_bal': torch.tensor(self.lambda_bal, device=loss_total.device, dtype=loss_total.dtype),
            'proxy_temperature': torch.tensor(self.proxy_temperature, device=loss_total.device, dtype=loss_total.dtype),
            'diag_temperature': torch.tensor(self.diag_temperature, device=loss_total.device, dtype=loss_total.dtype),
            'retrieval_temperature': self.get_retrieval_temperature().to(device=loss_total.device, dtype=loss_total.dtype),
            'logit_scale': self.get_logit_scale().to(device=loss_total.device, dtype=loss_total.dtype),
            'debug_metrics': {
                'diag_pos_cosine_mean': diag_pos_cosine_mean,
                'diag_hardneg_cosine_mean': diag_hardneg_cosine_mean,
                'diag_gap_margin_mean': diag_gap_margin_mean,
                'diag_student_teacher_pos_mean': dir_info['positive_mean'].detach(),
                'diag_student_teacher_offdiag_mean': dir_info['offdiag_mean'].detach(),
                'diag_student_teacher_margin': dir_info['margin'].detach(),
                'loss_diag_row': dir_info['loss_row'].detach(),
                'loss_diag_col': dir_info['loss_col'].detach(),
                'routing_effective_support_mean': routing_support_mean,
                'routing_effective_support_std': routing_support_std,
                'host_margin_mean': host_margin_mean,
                'host_margin_min': host_margin_min,
                'host_weight_mean': host_weight_mean,
                'host_weight_std': host_weight_std,
                'proto_score_mean': proto_score_mean,
                'proto_diag_mean': proto_diag_mean,
                'proto_host_score_corr': proto_host_score_corr,
                'semantic_assignment_entropy_image': semantic_info['assignment_entropy_image'],
                'semantic_assignment_entropy_teacher': semantic_info['assignment_entropy_teacher'],
                'semantic_target_entropy': semantic_info['target_entropy'],
                'semantic_pbt_valid_cluster_count': semantic_info['valid_cluster_count'],
                'semantic_pbt_empty_cluster_count': semantic_info['empty_cluster_count'],
                'sem_hardneg_pos_img_mean': hardneg_info['pos_img_mean'],
                'sem_hardneg_neg_img_mean': hardneg_info['neg_img_mean'],
                'sem_hardneg_pos_txt_mean': hardneg_info['pos_txt_mean'],
                'sem_hardneg_neg_txt_mean': hardneg_info['neg_txt_mean'],
                'sem_hardneg_pos_host_global_mean': hardneg_info['pos_host_global_mean'],
                'sem_hardneg_neg_host_global_mean': hardneg_info['neg_host_global_mean'],
                'sem_hardneg_margin_host_global_mean': hardneg_info['margin_host_global_mean'],
                'sem_hardneg_bridge_weight_mean': hardneg_info['bridge_weight_mean'],
                'sem_hardneg_bridge_loss_mean': hardneg_info['bridge_loss_mean'],
                'sem_hardneg_valid_i2t_frac': hardneg_info['valid_i2t_frac'],
                'sem_hardneg_valid_t2i_frac': hardneg_info['valid_t2i_frac'],
                'sem_hardneg_same_id_i2t_rate': hardneg_info['same_id_i2t_rate'],
                'sem_hardneg_same_id_t2i_rate': hardneg_info['same_id_t2i_rate'],
                'semantic_hosthard_weight_mean': hosthard_info['weight_mean'],
                'semantic_hosthard_weight_max': hosthard_info['weight_max'],
                'semantic_hosthard_margin_row_mean': hosthard_info['margin_row_mean'],
                'semantic_hosthard_margin_col_mean': hosthard_info['margin_col_mean'],
                'semantic_hosthard_margin_mean': hosthard_info['margin_mean'],
                **self._cross_modal_debug_metrics('image_surrogate', image_embeddings, surrogate_text_embeddings, pids_for_metrics),
                **self._cross_modal_debug_metrics('image_exact', image_embeddings, exact_text_embeddings, pids_for_metrics),
                **self._surrogate_pairwise_debug_metrics(surrogate_pairwise_logits),
                **self._norm_stats('class_proxy_norm', self.class_proxies.detach()),
                **self._normalized_norm_stats('class_proxy_norm_normalized', self.class_proxies.detach()),
            },
        }
        if return_debug:
            if isinstance(surrogate_pairwise_logits, torch.Tensor):
                outputs['surrogate_retrieval_logits'] = surrogate_pairwise_logits
        return outputs
