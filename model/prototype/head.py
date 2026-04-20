from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregator import PrototypeAggregator
from .contextualizer import PrototypeContextualizer
from .losses import PrototypeLosses
from .projector import MLPProjector
from .prototype_bank import PrototypeBank
from .router import Router
from .token_mask import TokenMaskBuilder
from .token_pooler import MaskedTokenPooler
from .token_scorer import TokenScorer


class PrototypeConditionedTextHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_prototypes: int,
        prototype_dim: int,
        projector_output_dim: int,
        projector_hidden_dim: Optional[int] = None,
        projector_dropout: float = 0.0,
        projector_type: str = 'mlp2',
        prototype_init: str = 'normalized_random',
        prototype_init_path: Optional[str] = None,
        prototype_init_hybrid_ratio: float = 0.5,
        prototype_init_max_iters: int = 50,
        prototype_init_tol: float = 1e-4,
        prototype_init_seed: Optional[int] = None,
        prototype_init_features: Optional[torch.Tensor] = None,
        image_adapter: Optional[nn.Module] = None,
        text_adapter: Optional[nn.Module] = None,
        routing_type: str = 'cosine',
        routing_temperature: float = 0.07,
        routing_source: str = 'global',
        local_routing_temperature: Optional[float] = None,
        local_routing_pooling: str = 'logsumexp',
        local_routing_use_adapter: bool = True,
        local_routing_adapter_dim: Optional[int] = None,
        local_routing_normalize_inputs: bool = True,
        prototype_method_role: str = 'retrieval_branch',
        prototype_semantic_enabled: bool = False,
        prototype_recompute_enabled: bool = False,
        prototype_bank_source: str = 'learnable_legacy',
        prototype_use_base_for_semantic_targets: bool = True,
        semantic_structure_enabled: bool = False,
        semantic_feature_space: str = 'prototype_projected',
        semantic_pbt_enabled: bool = True,
        semantic_soft_target_enabled: bool = True,
        semantic_target_temperature: float = 0.01,
        semantic_pred_temperature: float = 0.07,
        semantic_recompute_schedule: str = 'epoch',
        semantic_recompute_interval: int = 1,
        semantic_min_cluster_count_for_pbt: float = 1.0,
        semantic_empty_cluster_policy: str = 'skip',
        semantic_text_teacher_source: str = 'exact_diagonal',
        semantic_text_student_source: str = 'surrogate_diagonal',
        semantic_image_student_source: str = 'image_semantic_feature',
        semantic_recompute_start_epoch: int = 0,
        semantic_recompute_start_step: int = 0,
        semantic_loss_ramp_start_epoch: int = 0,
        semantic_loss_ramp_start_step: int = 0,
        semantic_loss_ramp_epochs: int = 0,
        semantic_loss_ramp_steps: int = 0,
        semantic_ramp_loss_diag: bool = False,
        semantic_ramp_loss_semantic_pbt: bool = True,
        semantic_ramp_use_prototype: bool = False,
        token_scoring_type: str = 'cosine',
        token_temperature: float = 0.07,
        token_policy: str = 'content_only',
        special_token_ids: Optional[Dict[str, object]] = None,
        error_on_empty_kept_tokens: bool = True,
        contextualization_enabled: bool = False,
        contextualization_type: str = 'none',
        contextualization_residual: bool = True,
        normalize_for_self_interaction: bool = True,
        normalize_for_routing: bool = True,
        normalize_for_token_scoring: bool = True,
        normalize_projector_outputs: bool = True,
        use_image_conditioned_pooling: bool = True,
        num_classes: int = 0,
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
        use_diversity_loss: bool = False,
        diversity_loss_weight: float = 0.0,
        use_balance_loss: bool = False,
        balance_loss_weight: float = 0.0,
        contrastive_temperature_init: float = 0.07,
        learnable_contrastive_temperature: bool = False,
        dead_prototype_threshold: float = 0.005,
        collect_debug_metrics: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.prototype_dim = int(prototype_dim)
        self.projector_hidden_dim = int(projector_hidden_dim or prototype_dim)
        self.projector_output_dim = int(projector_output_dim)
        self.dead_prototype_threshold = float(dead_prototype_threshold)
        self.use_image_conditioned_pooling = bool(use_image_conditioned_pooling)
        self.collect_debug_metrics = bool(collect_debug_metrics)
        self.routing_source = str(routing_source).lower()
        if self.routing_source not in {'global', 'local_evidence'}:
            raise ValueError(
                f'Unsupported routing_source={routing_source!r}. Allowed values: [\"global\", \"local_evidence\"].'
            )
        self.prototype_method_role = str(prototype_method_role).lower()
        if self.prototype_method_role not in {'retrieval_branch', 'semantic_structure'}:
            raise ValueError(
                f'Unsupported prototype_method_role={prototype_method_role!r}. '
                'Allowed values: [\"retrieval_branch\", \"semantic_structure\"].'
            )
        self.prototype_semantic_enabled = bool(prototype_semantic_enabled)
        self.semantic_structure_enabled = bool(semantic_structure_enabled)
        self.prototype_recompute_enabled = bool(prototype_recompute_enabled)
        self.prototype_bank_source = str(prototype_bank_source).lower()
        if self.prototype_bank_source in {'', 'auto'}:
            self.prototype_bank_source = 'recomputed_kmeans' if self.prototype_method_role == 'semantic_structure' else 'learnable_legacy'
        if self.prototype_bank_source not in {'learnable_legacy', 'recomputed_kmeans'}:
            raise ValueError(
                f'Unsupported prototype_bank_source={prototype_bank_source!r}. '
                'Allowed values: [\"learnable_legacy\", \"recomputed_kmeans\"].'
            )
        self.prototype_use_base_for_semantic_targets = bool(prototype_use_base_for_semantic_targets)
        self.semantic_feature_space = str(semantic_feature_space).lower()
        self.semantic_pbt_enabled = bool(semantic_pbt_enabled)
        self.semantic_soft_target_enabled = bool(semantic_soft_target_enabled)
        self.semantic_target_temperature = float(semantic_target_temperature)
        self.semantic_pred_temperature = float(semantic_pred_temperature)
        self.semantic_recompute_schedule = str(semantic_recompute_schedule).lower()
        if self.semantic_recompute_schedule not in {'epoch', 'steps', 'stage'}:
            raise ValueError(
                f'Unsupported semantic_recompute_schedule={semantic_recompute_schedule!r}. '
                'Allowed values: [\"epoch\", \"steps\", \"stage\"].'
            )
        self.semantic_recompute_interval = max(int(semantic_recompute_interval), 1)
        self.semantic_min_cluster_count_for_pbt = float(semantic_min_cluster_count_for_pbt)
        self.semantic_empty_cluster_policy = str(semantic_empty_cluster_policy).lower()
        if self.semantic_empty_cluster_policy not in {'skip', 'reseed'}:
            raise ValueError(
                f'Unsupported semantic_empty_cluster_policy={semantic_empty_cluster_policy!r}. '
                'Allowed values: [\"skip\", \"reseed\"].'
            )
        self.semantic_text_teacher_source = str(semantic_text_teacher_source).lower()
        self.semantic_text_student_source = str(semantic_text_student_source).lower()
        self.semantic_image_student_source = str(semantic_image_student_source).lower()
        self.semantic_recompute_start_epoch = max(int(semantic_recompute_start_epoch), 0)
        self.semantic_recompute_start_step = max(int(semantic_recompute_start_step), 0)
        self.semantic_loss_ramp_start_epoch = max(int(semantic_loss_ramp_start_epoch), 0)
        self.semantic_loss_ramp_start_step = max(int(semantic_loss_ramp_start_step), 0)
        self.semantic_loss_ramp_epochs = max(int(semantic_loss_ramp_epochs), 0)
        self.semantic_loss_ramp_steps = max(int(semantic_loss_ramp_steps), 0)
        self.semantic_ramp_loss_diag = bool(semantic_ramp_loss_diag)
        self.semantic_ramp_loss_semantic_pbt = bool(semantic_ramp_loss_semantic_pbt)
        self.semantic_ramp_use_prototype = bool(semantic_ramp_use_prototype)
        self.defer_prototype_init_until_semantic_start = bool(
            self.semantic_ramp_use_prototype
            and self.prototype_method_role == 'semantic_structure'
            and self.prototype_semantic_enabled
            and self.semantic_structure_enabled
            and (
                self.semantic_recompute_start_epoch > 0
                or self.semantic_recompute_start_step > 0
            )
        )
        self._semantic_recompute_count = 0
        self._semantic_last_recompute_epoch: Optional[int] = None
        self._semantic_last_recompute_step: Optional[int] = None
        self._semantic_cache_initialized = False
        resolved_local_temperature = routing_temperature if local_routing_temperature is None else local_routing_temperature
        self.local_routing_temperature = float(resolved_local_temperature)
        if self.local_routing_temperature <= 0:
            raise ValueError('local_routing_temperature must be positive.')
        self.local_routing_pooling = str(local_routing_pooling).lower()
        if self.local_routing_pooling not in {'logsumexp', 'lse', 'max', 'mean', 'avg'}:
            raise ValueError(
                f'Unsupported local_routing_pooling={local_routing_pooling!r}. Allowed values: [\"logsumexp\", \"max\", \"mean\"].'
            )
        self.local_routing_use_adapter = bool(local_routing_use_adapter)
        self.local_routing_normalize_inputs = bool(local_routing_normalize_inputs)
        self.local_routing_adapter_dim = (
            None if local_routing_adapter_dim in (None, 0) else int(local_routing_adapter_dim)
        )
        if self.local_routing_adapter_dim is not None and self.local_routing_adapter_dim <= 0:
            raise ValueError('local_routing_adapter_dim must be positive when provided.')

        self.image_adapter = image_adapter if image_adapter is not None else (nn.Identity() if self.input_dim == self.prototype_dim else nn.Linear(self.input_dim, self.prototype_dim))
        self.text_adapter = text_adapter if text_adapter is not None else (nn.Identity() if self.input_dim == self.prototype_dim else nn.Linear(self.input_dim, self.prototype_dim))
        self.prototype_bank = PrototypeBank(
            num_prototypes=num_prototypes,
            prototype_dim=prototype_dim,
            init_mode=prototype_init,
            init_path=prototype_init_path,
            normalize_init=normalize_for_self_interaction,
            init_hybrid_ratio=prototype_init_hybrid_ratio,
            init_max_iters=prototype_init_max_iters,
            init_tol=prototype_init_tol,
            init_seed=prototype_init_seed,
            init_features=prototype_init_features,
            defer_initialization=self.defer_prototype_init_until_semantic_start,
        )
        # Recomputed semantic anchors are ephemeral runtime caches and should not break
        # checkpoint compatibility when older checkpoints are loaded.
        self.register_buffer(
            'semantic_base_prototypes_cache',
            torch.empty(self.prototype_bank.num_prototypes, self.prototype_dim),
            persistent=False,
        )
        self.register_buffer(
            'semantic_cluster_counts_cache',
            torch.zeros(self.prototype_bank.num_prototypes),
            persistent=False,
        )
        self.contextualizer = PrototypeContextualizer(
            enabled=contextualization_enabled,
            contextualization_type=contextualization_type,
            residual=contextualization_residual,
            normalize=normalize_for_self_interaction,
        )
        self.router = Router(
            routing_type=routing_type,
            temperature=routing_temperature,
            normalize=normalize_for_routing,
        )
        if self.local_routing_use_adapter:
            if self.local_routing_adapter_dim is None:
                self.local_routing_adapter = nn.Linear(self.prototype_dim, self.prototype_dim)
            else:
                self.local_routing_adapter = nn.Sequential(
                    nn.Linear(self.prototype_dim, self.local_routing_adapter_dim),
                    nn.GELU(),
                    nn.Linear(self.local_routing_adapter_dim, self.prototype_dim),
                )
        else:
            self.local_routing_adapter = nn.Identity()
        self.aggregator = PrototypeAggregator()
        self.token_scorer = TokenScorer(
            scoring_type=token_scoring_type,
            temperature=token_temperature,
            normalize=normalize_for_token_scoring,
        )
        self.token_mask_builder = TokenMaskBuilder(
            token_policy=token_policy,
            special_token_ids=special_token_ids,
            error_on_empty_kept_tokens=error_on_empty_kept_tokens,
        )
        self.token_pooler = MaskedTokenPooler()
        self.text_pool_query = nn.Parameter(torch.randn(self.prototype_dim, dtype=torch.float32))
        self.proto_query_proj = nn.Linear(self.prototype_dim, self.prototype_dim)
        with torch.no_grad():
            self.proto_query_proj.weight.zero_()
            self.proto_query_proj.bias.zero_()
        self.image_projector = MLPProjector(
            input_dim=prototype_dim,
            hidden_dim=self.projector_hidden_dim,
            output_dim=self.projector_output_dim,
            dropout=projector_dropout,
            normalize_output=normalize_projector_outputs,
            projector_type=projector_type,
        )
        self.text_projector = MLPProjector(
            input_dim=prototype_dim,
            hidden_dim=self.projector_hidden_dim,
            output_dim=self.projector_output_dim,
            dropout=projector_dropout,
            normalize_output=normalize_projector_outputs,
            projector_type=projector_type,
        )
        self.losses = PrototypeLosses(
            temperature_init=contrastive_temperature_init,
            learnable_temperature=learnable_contrastive_temperature,
            normalize_embeddings=normalize_projector_outputs,
            num_classes=num_classes,
            embedding_dim=self.projector_output_dim,
            proxy_temperature=proxy_temperature,
            use_loss_diag=use_loss_diag,
            lambda_diag=lambda_diag,
            diag_temperature=diag_temperature,
            use_loss_semantic_pbt=use_loss_semantic_pbt,
            lambda_semantic_pbt=lambda_semantic_pbt,
            use_loss_semantic_hardneg_margin=use_loss_semantic_hardneg_margin,
            lambda_semantic_hardneg_margin=lambda_semantic_hardneg_margin,
            semantic_hardneg_margin=semantic_hardneg_margin,
            semantic_hardneg_eps=semantic_hardneg_eps,
            prototype_method_role=prototype_method_role,
            prototype_semantic_enabled=prototype_semantic_enabled,
            semantic_structure_enabled=semantic_structure_enabled,
            semantic_feature_space=semantic_feature_space,
            semantic_pbt_enabled=semantic_pbt_enabled,
            semantic_soft_target_enabled=semantic_soft_target_enabled,
            semantic_target_temperature=semantic_target_temperature,
            semantic_pred_temperature=semantic_pred_temperature,
            semantic_min_cluster_count_for_pbt=semantic_min_cluster_count_for_pbt,
            semantic_empty_cluster_policy=semantic_empty_cluster_policy,
            use_diversity_loss=use_diversity_loss,
            diversity_loss_weight=diversity_loss_weight,
            use_balance_loss=use_balance_loss,
            balance_loss_weight=balance_loss_weight,
        )

    def _compute_special_mass(self, token_weights: torch.Tensor, special_token_positions: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_index = torch.arange(token_weights.size(0), device=token_weights.device)
        total_mass = torch.zeros((), device=token_weights.device, dtype=token_weights.dtype)
        has_special_positions = False
        for key in ('cls', 'eos'):
            positions = special_token_positions.get(key)
            if positions is None:
                continue
            total_mass = total_mass + token_weights[batch_index, positions].mean()
            has_special_positions = True
        if has_special_positions:
            return total_mass
        return torch.zeros((), device=token_weights.device, dtype=token_weights.dtype)

    def _compute_usage_metrics(self, routing_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        usage = routing_weights.mean(dim=0)
        top1_assignments = routing_weights.argmax(dim=-1)
        top1_histogram = torch.bincount(top1_assignments, minlength=routing_weights.size(1)).to(dtype=routing_weights.dtype)
        top1_usage = top1_histogram / top1_histogram.sum().clamp_min(1.0)
        support_ipr = torch.reciprocal(routing_weights.pow(2).sum(dim=-1).clamp_min(1e-12)).float()
        top_values = torch.topk(routing_weights, k=min(4, routing_weights.size(1)), dim=-1).values
        top1_values = top_values[:, 0]
        top2_values = top_values[:, 1] if top_values.size(1) > 1 else torch.zeros_like(top1_values)
        support_quantiles = torch.quantile(
            support_ipr,
            torch.tensor([0.1, 0.5, 0.9], device=support_ipr.device, dtype=support_ipr.dtype),
        )
        return {
            'prototype_usage': usage.detach(),
            'prototype_usage_entropy': (-(usage * usage.clamp_min(1e-12).log()).sum()).detach(),
            'prototype_usage_max': usage.max().detach(),
            'prototype_dead_count': (usage < self.dead_prototype_threshold).sum().detach(),
            'routing_entropy': (-(routing_weights * routing_weights.clamp_min(1e-12).log()).sum(dim=-1).mean()).detach(),
            'routing_effective_support_ipr': support_ipr.mean().detach(),
            'routing_effective_support_ipr_p10': support_quantiles[0].detach(),
            'routing_effective_support_ipr_p50': support_quantiles[1].detach(),
            'routing_effective_support_ipr_p90': support_quantiles[2].detach(),
            'routing_support_below_2_frac': (support_ipr < 2.0).float().mean().detach(),
            'routing_support_below_3_frac': (support_ipr < 3.0).float().mean().detach(),
            'routing_support_below_min_frac': (support_ipr < 2.0).float().mean().detach(),
            # These summarize whether routing is truly mixed or mostly top-1 plus residual mass.
            'routing_top1_minus_top2': (top1_values - top2_values).mean().detach(),
            'routing_top2_mass': top_values[:, :min(2, top_values.size(1))].sum(dim=-1).mean().detach(),
            'routing_top4_mass': top_values.sum(dim=-1).mean().detach(),
            'routing_top1_histogram': top1_usage.detach(),
            'routing_top1_usage_entropy': (-(top1_usage * top1_usage.clamp_min(1e-12).log()).sum()).detach(),
            'routing_top1_usage_max': top1_usage.max().detach(),
            'routing_top1_dead_count': (top1_usage == 0).sum().detach(),
            'prototype_active_count_eps_1e-3': (usage > 1e-3).sum().detach(),
            'prototype_active_count_eps_1e-2': (usage > 1e-2).sum().detach(),
        }

    def _truncate_routing_weights(self, routing_weights: torch.Tensor, k: int) -> torch.Tensor:
        keep_count = min(max(int(k), 1), routing_weights.size(1))
        topk_values, topk_indices = torch.topk(routing_weights, k=keep_count, dim=-1)
        truncated = torch.zeros_like(routing_weights)
        truncated.scatter_(1, topk_indices, topk_values)
        return truncated / truncated.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _mean_diagonal_cosine(self, surrogate_embeddings: torch.Tensor, exact_embeddings: torch.Tensor) -> torch.Tensor:
        surrogate_normalized = F.normalize(surrogate_embeddings, dim=-1)
        exact_normalized = F.normalize(exact_embeddings, dim=-1)
        return (surrogate_normalized * exact_normalized).sum(dim=-1).mean()

    def _compute_routing_certification_metrics(
        self,
        routing_weights: torch.Tensor,
        basis_bank: torch.Tensor,
        surrogate_text_projected: torch.Tensor,
        exact_text_projected: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        metrics = {}
        with torch.no_grad():
            # Re-run surrogate reconstruction under truncated top-k routing to verify that
            # diagonal fidelity depends on a genuine mixture rather than only a softened lookup.
            metrics['diag_cos_full'] = self._mean_diagonal_cosine(surrogate_text_projected, exact_text_projected).detach()
            metrics['loss_diag_full'] = self.losses.diagonal_fidelity_loss(surrogate_text_projected, exact_text_projected).detach()
            for k in (1, 2, 4):
                truncated_routing = self._truncate_routing_weights(routing_weights.detach(), k=k)
                truncated_pooled_text = self.reconstruct_surrogate_text(truncated_routing, basis_bank.detach())
                truncated_projected = self.text_projector(truncated_pooled_text)
                metrics[f'diag_cos_top{k}'] = self._mean_diagonal_cosine(truncated_projected, exact_text_projected.detach()).detach()
                metrics[f'loss_diag_top{k}'] = self.losses.diagonal_fidelity_loss(truncated_projected, exact_text_projected.detach()).detach()
        return metrics

    def _compute_pairwise_cosine_metrics(self, prefix: str, prototypes: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized = torch.nn.functional.normalize(prototypes, dim=-1)
        similarity = normalized @ normalized.t()
        mask = ~torch.eye(similarity.size(0), device=similarity.device, dtype=torch.bool)
        off_diagonal = similarity[mask]
        if off_diagonal.numel() == 0:
            off_diagonal = torch.zeros(1, device=similarity.device, dtype=similarity.dtype)
        return {
            f'{prefix}_pairwise_cosine_mean': off_diagonal.mean().detach(),
            f'{prefix}_pairwise_cosine_std': off_diagonal.std(unbiased=False).detach(),
            f'{prefix}_pairwise_cosine_max': off_diagonal.max().detach(),
        }

    def _collect_scalar_metrics(
        self,
        prototypes: torch.Tensor,
        contextualized_prototypes: torch.Tensor,
        routing_weights: torch.Tensor,
        summary: torch.Tensor,
        exact_token_weights: torch.Tensor,
        token_valid_mask: torch.Tensor,
        token_keep_mask: torch.Tensor,
        surrogate_pooled_text: torch.Tensor,
        exact_pooled_text: torch.Tensor,
        image_features: torch.Tensor,
        image_projector_debug: Dict[str, torch.Tensor],
        surrogate_text_projector_debug: Dict[str, torch.Tensor],
        exact_text_projector_debug: Dict[str, torch.Tensor],
        special_token_positions: Dict[str, torch.Tensor],
        contextualizer_debug: Optional[Dict[str, torch.Tensor]] = None,
        router_debug: Optional[Dict[str, torch.Tensor]] = None,
        pooler_debug: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        metrics = {}
        metrics.update(self._compute_usage_metrics(routing_weights))
        metrics.update(self._compute_pairwise_cosine_metrics('prototype', prototypes))
        metrics.update(self._compute_pairwise_cosine_metrics('contextualized_prototype', contextualized_prototypes))
        image_embed_norms = image_projector_debug['projected_features_raw'].norm(dim=-1)
        image_embed_unit_norms = image_projector_debug['projected_features'].norm(dim=-1)
        surrogate_text_embed_norms = surrogate_text_projector_debug['projected_features_raw'].norm(dim=-1)
        surrogate_text_embed_unit_norms = surrogate_text_projector_debug['projected_features'].norm(dim=-1)
        exact_text_embed_norms = exact_text_projector_debug['projected_features_raw'].norm(dim=-1)
        exact_text_embed_unit_norms = exact_text_projector_debug['projected_features'].norm(dim=-1)
        metrics.update(
            {
                'q_norm': summary.norm(dim=-1).mean().detach(),
                't_pool_norm': surrogate_pooled_text.norm(dim=-1).mean().detach(),
                'surrogate_t_pool_norm': surrogate_pooled_text.norm(dim=-1).mean().detach(),
                'exact_t_pool_norm': exact_pooled_text.norm(dim=-1).mean().detach(),
                'image_feature_norm': image_features.norm(dim=-1).mean().detach(),
                'image_embed_norm': image_embed_norms.mean().detach(),
                'image_embed_norm_raw': image_embed_norms.mean().detach(),
                'image_embed_unit_norm': image_embed_unit_norms.mean().detach(),
                'image_embed_norm_std': image_embed_norms.std(unbiased=False).detach(),
                'image_embed_norm_min': image_embed_norms.min().detach(),
                'image_embed_norm_max': image_embed_norms.max().detach(),
                'text_embed_norm': surrogate_text_embed_norms.mean().detach(),
                'text_embed_norm_raw': surrogate_text_embed_norms.mean().detach(),
                'text_embed_unit_norm': surrogate_text_embed_unit_norms.mean().detach(),
                'text_embed_norm_std': surrogate_text_embed_norms.std(unbiased=False).detach(),
                'text_embed_norm_min': surrogate_text_embed_norms.min().detach(),
                'text_embed_norm_max': surrogate_text_embed_norms.max().detach(),
                'surrogate_text_embed_norm': surrogate_text_embed_norms.mean().detach(),
                'surrogate_text_embed_norm_raw': surrogate_text_embed_norms.mean().detach(),
                'surrogate_text_embed_unit_norm': surrogate_text_embed_unit_norms.mean().detach(),
                'surrogate_text_embed_norm_std': surrogate_text_embed_norms.std(unbiased=False).detach(),
                'surrogate_text_embed_norm_min': surrogate_text_embed_norms.min().detach(),
                'surrogate_text_embed_norm_max': surrogate_text_embed_norms.max().detach(),
                'exact_text_embed_norm': exact_text_embed_norms.mean().detach(),
                'exact_text_embed_norm_raw': exact_text_embed_norms.mean().detach(),
                'exact_text_embed_unit_norm': exact_text_embed_unit_norms.mean().detach(),
                'exact_text_embed_norm_std': exact_text_embed_norms.std(unbiased=False).detach(),
                'exact_text_embed_norm_min': exact_text_embed_norms.min().detach(),
                'exact_text_embed_norm_max': exact_text_embed_norms.max().detach(),
                'token_valid_fraction': token_valid_mask.float().mean().detach(),
                'valid_token_fraction': token_keep_mask.float().mean().detach(),
                'token_special_mass': self._compute_special_mass(exact_token_weights, special_token_positions).detach(),
            }
        )
        if contextualizer_debug and 'prototype_contextualization_entropy' in contextualizer_debug:
            metrics['prototype_contextualization_entropy'] = contextualizer_debug['prototype_contextualization_entropy']
        if router_debug:
            metrics['routing_max_prob'] = router_debug['routing_max_prob']
            metrics['prototype_assignment_entropy'] = router_debug['prototype_assignment_entropy']
            if 'routing_effective_support' in router_debug:
                metrics['routing_effective_support'] = router_debug['routing_effective_support']
            if 'routing_source_mode' in router_debug:
                metrics['routing_source_mode'] = router_debug['routing_source_mode']
            if 'local_routing_entropy' in router_debug:
                metrics['local_routing_entropy'] = router_debug['local_routing_entropy']
            if 'local_routing_max_mean' in router_debug:
                metrics['local_routing_max_mean'] = router_debug['local_routing_max_mean']
            if 'local_routing_effective_support' in router_debug:
                metrics['local_routing_effective_support'] = router_debug['local_routing_effective_support']
            for metric_name in (
                'local_token_norm_mean',
            ):
                if metric_name in router_debug:
                    metrics[metric_name] = router_debug[metric_name]
        if pooler_debug:
            metrics['token_pool_entropy'] = pooler_debug['token_pool_entropy']
            metrics['beta_max_prob'] = pooler_debug['beta_max_prob']
        return metrics

    def _semantic_mode_enabled(self) -> bool:
        return bool(
            self.prototype_method_role == 'semantic_structure'
            and self.prototype_semantic_enabled
            and self.semantic_structure_enabled
        )

    def _semantic_schedule_started(self, *, epoch: Optional[int], current_step: Optional[int]) -> bool:
        if epoch is not None and int(epoch) < int(self.semantic_recompute_start_epoch):
            return False
        if current_step is not None and int(current_step) < int(self.semantic_recompute_start_step):
            return False
        return True

    def _prototype_loss_scale(self, *, epoch: Optional[int], current_step: Optional[int]) -> float:
        if not self._semantic_mode_enabled():
            # Ramp controls are semantic-mode scheduling knobs; keep legacy retrieval
            # behavior unaffected when semantic mode is not active.
            return 1.0
        if not self._semantic_schedule_started(epoch=epoch, current_step=current_step):
            return 0.0
        scale = 1.0
        if self.semantic_loss_ramp_epochs > 0 and epoch is not None:
            epoch_progress = (float(epoch) - float(self.semantic_loss_ramp_start_epoch) + 1.0) / float(self.semantic_loss_ramp_epochs)
            scale = min(scale, max(0.0, min(epoch_progress, 1.0)))
        if self.semantic_loss_ramp_steps > 0 and current_step is not None:
            step_progress = (float(current_step) - float(self.semantic_loss_ramp_start_step) + 1.0) / float(self.semantic_loss_ramp_steps)
            scale = min(scale, max(0.0, min(step_progress, 1.0)))
        return float(max(scale, 0.0))

    def _zero_loss_outputs(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        return {
            'loss_total': zero,
            'loss_semantic_pbt': zero,
            'loss_semantic_hardneg_margin': zero,
            'loss_semantic_hardneg_margin_image': zero,
            'loss_semantic_hardneg_margin_text': zero,
            'loss_diag': zero,
            'loss_diversity': zero,
            'loss_balance': zero,
            'loss_semantic_pbt_weighted': zero,
            'loss_semantic_hardneg_margin_weighted': zero,
            'loss_diag_weighted': zero,
            'loss_diversity_weighted': zero,
            'loss_balance_weighted': zero,
            'use_loss_semantic_pbt': zero,
            'lambda_semantic_pbt': zero,
            'use_loss_semantic_hardneg_margin': zero,
            'lambda_semantic_hardneg_margin': zero,
            'semantic_hardneg_margin': zero,
            'semantic_hardneg_eps': zero,
            'prototype_loss_scale': zero,
            'prototype_loss_ramp_scale': zero,
            'loss_diag_scale': zero,
            'loss_semantic_pbt_scale': zero,
            'semantic_loss_scale': zero,
            'use_loss_diag': zero,
            'lambda_diag': zero,
            'lambda_div': zero,
            'lambda_bal': zero,
            'proxy_temperature': zero,
            'diag_temperature': zero,
            'retrieval_temperature': zero,
            'logit_scale': zero,
            'debug_metrics': {},
        }

    # Backward-compatible alias kept while downstream references migrate.
    def _semantic_loss_scale(self, *, epoch: Optional[int], current_step: Optional[int]) -> float:
        return self._prototype_loss_scale(epoch=epoch, current_step=current_step)

    def _should_recompute_semantic_anchors(self, *, epoch: Optional[int], current_step: Optional[int]) -> bool:
        if not self._semantic_mode_enabled():
            return False
        if self.prototype_bank_source != 'recomputed_kmeans':
            return False
        if not self.prototype_recompute_enabled:
            return False
        if not self.training:
            return False
        if not self._semantic_schedule_started(epoch=epoch, current_step=current_step):
            return False

        schedule = self.semantic_recompute_schedule
        interval = max(int(self.semantic_recompute_interval), 1)
        if not self._semantic_cache_initialized:
            return True
        if schedule in {'epoch', 'stage'}:
            if epoch is None:
                return False
            if self._semantic_last_recompute_epoch is None:
                return True
            return (int(epoch) - int(self._semantic_last_recompute_epoch)) >= interval
        if schedule == 'steps':
            if current_step is None:
                return False
            if self._semantic_last_recompute_step is None:
                return True
            return (int(current_step) - int(self._semantic_last_recompute_step)) >= interval
        return False

    def _recompute_kmeans_anchors(
        self,
        features: torch.Tensor,
        *,
        num_clusters: int,
        max_iters: int = 15,
    ) -> Dict[str, torch.Tensor]:
        detached = features.detach().float()
        if detached.ndim != 2 or detached.size(0) <= 0:
            raise ValueError('Semantic recompute features must have shape [B, D] with B > 0.')
        detached = F.normalize(detached, dim=-1)
        num_samples, feature_dim = detached.shape
        cluster_count = int(max(1, num_clusters))
        initial_indices = torch.arange(cluster_count, device=detached.device) % max(num_samples, 1)
        centers = detached.index_select(0, initial_indices).clone()
        if centers.size(-1) != feature_dim:
            raise ValueError('Semantic anchor recompute produced a feature-dimension mismatch.')

        empty_cluster_count = 0
        for _ in range(max(int(max_iters), 1)):
            similarity = detached @ centers.t()
            assignments = similarity.argmax(dim=-1)
            counts = torch.bincount(assignments, minlength=cluster_count).to(dtype=detached.dtype)
            updated_centers = centers.clone()
            for cluster_index in range(cluster_count):
                member_mask = assignments.eq(cluster_index)
                if member_mask.any():
                    updated_centers[cluster_index] = F.normalize(
                        detached[member_mask].mean(dim=0, keepdim=True),
                        dim=-1,
                    ).squeeze(0)
                    continue
                empty_cluster_count += 1
                if self.semantic_empty_cluster_policy == 'reseed':
                    least_fit_index = similarity.max(dim=1).values.argmin()
                    updated_centers[cluster_index] = detached[least_fit_index]
            center_shift = (updated_centers - centers).norm(dim=-1).max()
            centers = updated_centers
            if float(center_shift.item()) <= 1e-5:
                break

        final_similarity = detached @ centers.t()
        final_assignments = final_similarity.argmax(dim=-1)
        final_counts = torch.bincount(final_assignments, minlength=cluster_count).to(dtype=detached.dtype)
        return {
            'centers': F.normalize(centers, dim=-1),
            'counts': final_counts,
            'empty_cluster_count': final_counts.eq(0).sum().to(dtype=detached.dtype),
            'empty_cluster_reseed_events': detached.new_tensor(float(empty_cluster_count)),
        }

    def _maybe_refresh_semantic_anchor_cache(
        self,
        *,
        features: Optional[torch.Tensor],
        epoch: Optional[int],
        current_step: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        diagnostics: Dict[str, torch.Tensor] = {}
        if not self._should_recompute_semantic_anchors(epoch=epoch, current_step=current_step):
            diagnostics['semantic_recompute_triggered'] = self.semantic_cluster_counts_cache.new_zeros(())
            return diagnostics
        if features is None or not isinstance(features, torch.Tensor) or features.ndim != 2 or features.size(0) <= 0:
            diagnostics['semantic_recompute_triggered'] = self.semantic_cluster_counts_cache.new_zeros(())
            diagnostics['semantic_recompute_skipped_no_features'] = self.semantic_cluster_counts_cache.new_ones(())
            return diagnostics

        with torch.no_grad():
            recomputed = self._recompute_kmeans_anchors(
                features=features,
                num_clusters=self.prototype_bank.num_prototypes,
                max_iters=max(int(getattr(self.prototype_bank, 'init_max_iters', 15)), 1),
            )
            centers = recomputed['centers'].to(
                device=self.semantic_base_prototypes_cache.device,
                dtype=self.semantic_base_prototypes_cache.dtype,
            )
            counts = recomputed['counts'].to(
                device=self.semantic_cluster_counts_cache.device,
                dtype=self.semantic_cluster_counts_cache.dtype,
            )
            if centers.shape != self.semantic_base_prototypes_cache.shape:
                raise ValueError(
                    'Recomputed semantic anchors shape mismatch: '
                    f'expected {tuple(self.semantic_base_prototypes_cache.shape)} got {tuple(centers.shape)}.'
                )
            if counts.shape != self.semantic_cluster_counts_cache.shape:
                raise ValueError(
                    'Recomputed semantic cluster-count shape mismatch: '
                    f'expected {tuple(self.semantic_cluster_counts_cache.shape)} got {tuple(counts.shape)}.'
                )
            self.semantic_base_prototypes_cache.copy_(centers)
            self.semantic_cluster_counts_cache.copy_(counts)

        self._semantic_cache_initialized = True
        self._semantic_recompute_count += 1
        self._semantic_last_recompute_epoch = None if epoch is None else int(epoch)
        self._semantic_last_recompute_step = None if current_step is None else int(current_step)
        diagnostics.update(
            {
                'semantic_recompute_triggered': centers.new_ones(()),
                'semantic_recompute_count': centers.new_tensor(float(self._semantic_recompute_count)),
                'semantic_empty_cluster_count': recomputed['empty_cluster_count'].detach(),
                'semantic_empty_cluster_reseed_events': recomputed['empty_cluster_reseed_events'].detach(),
            }
        )
        return diagnostics

    def _compute_contextualized_from_base(
        self,
        base_prototypes: torch.Tensor,
        *,
        return_debug: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del return_debug
        contextualized, context_debug = self.contextualizer(base_prototypes, return_debug=True)
        return contextualized, context_debug

    def _prepare_semantic_anchor_features(
        self,
        anchor_prototypes: torch.Tensor,
        *,
        target_feature_dim: int,
    ) -> Tuple[torch.Tensor, str]:
        if anchor_prototypes.ndim != 2:
            return anchor_prototypes, 'invalid_rank'
        if anchor_prototypes.size(-1) == target_feature_dim and self.semantic_feature_space != 'prototype_projected':
            return anchor_prototypes, 'prototype_native'

        if self.semantic_feature_space == 'prototype_projected' or anchor_prototypes.size(-1) != target_feature_dim:
            image_projected = self.image_projector(anchor_prototypes, return_debug=False)
            text_projected = self.text_projector(anchor_prototypes, return_debug=False)
            projected = F.normalize(0.5 * (image_projected + text_projected), dim=-1)
            source = 'prototype_projected' if self.semantic_feature_space == 'prototype_projected' else 'auto_projected_dim_fix'
            return projected, source

        return anchor_prototypes, 'prototype_native'

    def get_prototype_context(
        self,
        return_debug: bool = False,
        *,
        epoch: Optional[int] = None,
        current_step: Optional[int] = None,
        semantic_recompute_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if hasattr(self.prototype_bank, 'is_initialized') and hasattr(self.prototype_bank, 'initialize_if_needed'):
            needs_init = not bool(self.prototype_bank.is_initialized())
            if needs_init:
                if not self.defer_prototype_init_until_semantic_start:
                    self.prototype_bank.initialize_if_needed()
                elif (epoch is not None or current_step is not None) and self._semantic_schedule_started(
                    epoch=epoch,
                    current_step=current_step,
                ):
                    self.prototype_bank.initialize_if_needed()
        legacy_prototypes, bank_debug = self.prototype_bank(return_debug=True)
        recompute_debug = self._maybe_refresh_semantic_anchor_cache(
            features=semantic_recompute_features,
            epoch=epoch,
            current_step=current_step,
        )
        semantic_mode = self._semantic_mode_enabled()
        use_recomputed_base = (
            semantic_mode
            and self.prototype_bank_source == 'recomputed_kmeans'
            and self._semantic_cache_initialized
        )
        base_prototypes = legacy_prototypes
        if use_recomputed_base:
            base_prototypes = self.semantic_base_prototypes_cache.to(
                device=legacy_prototypes.device,
                dtype=legacy_prototypes.dtype,
            )

        contextualized, contextual_debug = self._compute_contextualized_from_base(
            base_prototypes,
            return_debug=True,
        )
        routing_prototypes = contextualized

        outputs = {
            'legacy_prototypes': legacy_prototypes,
            'base_prototypes': base_prototypes,
            'prototypes': base_prototypes,
            'contextualized_prototypes': contextualized,
            'routing_prototypes': routing_prototypes,
            'bank_debug': bank_debug,
            'contextualizer_debug': contextual_debug,
            'prototype_source_type': 'recomputed_kmeans' if use_recomputed_base else 'learnable_legacy',
            'semantic_cluster_counts': self.semantic_cluster_counts_cache.to(
                device=legacy_prototypes.device,
                dtype=legacy_prototypes.dtype,
            ),
            'semantic_recompute_count': legacy_prototypes.new_tensor(float(self._semantic_recompute_count)),
            'semantic_recompute_last_epoch': legacy_prototypes.new_tensor(
                -1.0 if self._semantic_last_recompute_epoch is None else float(self._semantic_last_recompute_epoch)
            ),
            'semantic_recompute_last_step': legacy_prototypes.new_tensor(
                -1.0 if self._semantic_last_recompute_step is None else float(self._semantic_last_recompute_step)
            ),
            'semantic_mode_active': legacy_prototypes.new_tensor(float(semantic_mode)),
            'semantic_recompute_triggered': recompute_debug.get(
                'semantic_recompute_triggered',
                legacy_prototypes.new_zeros(()),
            ),
            'semantic_recompute_skipped_no_features': recompute_debug.get(
                'semantic_recompute_skipped_no_features',
                legacy_prototypes.new_zeros(()),
            ),
            'semantic_empty_cluster_reseed_events': recompute_debug.get(
                'semantic_empty_cluster_reseed_events',
                legacy_prototypes.new_zeros(()),
            ),
        }
        if return_debug:
            outputs['debug'] = {
                **bank_debug,
                **contextual_debug,
                **recompute_debug,
                'prototype_source_type_recomputed': legacy_prototypes.new_tensor(float(use_recomputed_base)),
                'prototype_source_type_legacy': legacy_prototypes.new_tensor(float(not use_recomputed_base)),
            }
        return outputs

    def _prepare_local_routing_tokens(
        self,
        image_embeddings: torch.Tensor,
        image_local_tokens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        debug_metrics: Dict[str, torch.Tensor] = {}
        if image_local_tokens is None:
            fallback_tokens = image_embeddings.unsqueeze(1)
            return self.local_routing_adapter(self.image_adapter(fallback_tokens)), debug_metrics

        if image_local_tokens.ndim != 3:
            raise ValueError('image_local_tokens must have shape [B, M, D] when routing_source=local_evidence.')
        if image_local_tokens.size(0) != image_embeddings.size(0):
            raise ValueError('image_local_tokens and image_embeddings must share the same batch size.')

        # Prefer patch/local evidence over CLS when available.
        local_tokens = image_local_tokens[:, 1:, :] if image_local_tokens.size(1) > 1 else image_local_tokens
        if local_tokens.size(1) <= 0:
            local_tokens = image_local_tokens
        debug_metrics['local_token_norm_mean'] = local_tokens.norm(dim=-1).mean().detach()
        local_tokens = self.image_adapter(local_tokens)
        local_tokens = self.local_routing_adapter(local_tokens)
        if local_tokens.size(-1) != self.prototype_dim:
            raise ValueError(
                'Local routing tokens must end in prototype_dim after adapters; '
                f'expected {self.prototype_dim}, got {local_tokens.size(-1)}.'
            )
        return local_tokens, debug_metrics

    def _compute_routing_weights(
        self,
        image_embeddings: torch.Tensor,
        image_features: torch.Tensor,
        contextualized_prototypes: torch.Tensor,
        image_local_tokens: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.routing_source == 'local_evidence':
            local_tokens, local_token_debug = self._prepare_local_routing_tokens(
                image_embeddings=image_embeddings,
                image_local_tokens=image_local_tokens,
            )
            routing_weights, routing_debug = self.router.route_from_local_evidence(
                local_embeddings=local_tokens,
                prototypes=contextualized_prototypes,
                pooling=self.local_routing_pooling,
                temperature=self.local_routing_temperature,
                normalize_inputs=self.local_routing_normalize_inputs,
                return_debug=True,
            )
            routing_debug.update(local_token_debug)
            routing_debug['routing_source_mode'] = routing_weights.new_tensor(1.0).detach()
            return {
                'routing_weights': routing_weights,
                'routing_debug': routing_debug,
            }

        routing_weights, routing_debug = self.router(image_features, contextualized_prototypes, return_debug=True)
        routing_debug['routing_source_mode'] = routing_weights.new_tensor(0.0).detach()
        routing_debug.setdefault('local_routing_entropy', routing_debug.get('prototype_assignment_entropy', routing_weights.new_zeros(())))
        routing_debug.setdefault('local_routing_max_mean', routing_debug.get('routing_max_prob', routing_weights.new_zeros(())))
        routing_debug.setdefault('local_routing_effective_support', routing_debug.get('routing_effective_support', routing_weights.new_zeros(())))
        return {
            'routing_weights': routing_weights,
            'routing_debug': routing_debug,
        }

    def encode_image_branch(
        self,
        image_embeddings: torch.Tensor,
        image_local_tokens: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None,
        contextualized_prototypes: Optional[torch.Tensor] = None,
        routing_prototypes: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Dict[str, object]:
        image_features = self.image_adapter(image_embeddings)
        context_debug = {}

        if prototypes is None or contextualized_prototypes is None or routing_prototypes is None:
            context = self.get_prototype_context(return_debug=return_debug)
            prototypes = context['base_prototypes']
            contextualized_prototypes = context['contextualized_prototypes']
            routing_prototypes = context['routing_prototypes']
            context_debug = {
                **context['bank_debug'],
                **context['contextualizer_debug'],
            }

        routing_outputs = self._compute_routing_weights(
            image_embeddings=image_embeddings,
            image_features=image_features,
            contextualized_prototypes=routing_prototypes,
            image_local_tokens=image_local_tokens,
        )
        routing_weights = routing_outputs['routing_weights']
        routing_debug = routing_outputs['routing_debug']
        summary, aggregator_debug = self.aggregator(routing_weights, routing_prototypes, return_debug=True)
        # Residual summary injection is disabled; keep image path anchored to adapted host image features.
        image_proxy_features = image_features
        image_projected, image_projector_debug = self.image_projector(image_proxy_features, return_debug=True)

        outputs = {
            'image_embedding': image_features,
            'image_proxy_features': image_proxy_features,
            'prototypes': prototypes,
            'base_prototypes': prototypes,
            'contextualized_prototypes': contextualized_prototypes,
            'routing_prototypes': routing_prototypes,
            'routing_weights': routing_weights,
            'summary': summary,
            'image_projected': image_projected,
            'image_projected_raw': image_projector_debug['projected_features_raw'],
            'router_debug': routing_debug,
            'aggregator_debug': aggregator_debug,
            'image_projector_debug': image_projector_debug,
        }
        if return_debug:
            outputs['debug'] = {
                **context_debug,
                **routing_debug,
                **aggregator_debug,
                **image_projector_debug,
            }
        return outputs

    def _prepare_text_inputs(
        self,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:
        text_features = self.text_adapter(text_token_states)
        token_keep_mask, mask_debug = self.token_mask_builder.build(
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
            return_debug=True,
        )
        return {
            'text_token_states': text_features,
            'token_keep_mask': token_keep_mask,
            'mask_debug': mask_debug,
        }

    def _build_text_pool_queries(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_image_conditioned_pooling:
            if summary is None:
                raise ValueError('summary must be provided when use_image_conditioned_pooling=true.')
            return summary
        return self.text_pool_query.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
    def pool_text_with_summary(
        self,
        summary: torch.Tensor,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
        return_debug: bool = False,
        disable_proxy_losses: bool = False,
        prepared_text: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        del pids
        text_inputs = prepared_text or self._prepare_text_inputs(
            text_token_states,
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
        )
        text_features = text_inputs['text_token_states']
        mask_debug = text_inputs['mask_debug']
        text_pool_queries = self._build_text_pool_queries(text_features.size(0), text_features.device, text_features.dtype, summary)
        token_scores, scorer_debug = self.token_scorer(text_pool_queries, text_features, return_debug=True)
        pooled_text, token_weights, pooler_debug = self.token_pooler(
            token_scores,
            text_features,
            mask_debug['token_keep_mask'],
            return_debug=True,
        )
        text_projected, text_projector_debug = self.text_projector(pooled_text, return_debug=True)

        outputs = {
            'text_token_states': text_features,
            'token_scores': token_scores,
            'token_valid_mask': mask_debug['token_valid_mask'],
            'token_keep_mask': mask_debug['token_keep_mask'],
            'valid_mask': mask_debug['token_keep_mask'],
            'token_weights': token_weights,
            'beta_logits_masked': pooler_debug['beta_logits_masked'],
            'pooled_text': pooled_text,
            'text_projected': text_projected,
            'text_projected_raw': text_projector_debug['projected_features_raw'],
            'special_token_positions': mask_debug['special_token_positions'],
            'scorer_debug': scorer_debug,
            'mask_debug': mask_debug,
            'pooler_debug': pooler_debug,
            'text_projector_debug': text_projector_debug,
        }
        if return_debug:
            outputs['debug'] = {
                **scorer_debug,
                **mask_debug,
                **pooler_debug,
                **text_projector_debug,
            }
        return outputs

    def build_text_basis_bank(
        self,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        contextualized_prototypes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
        return_debug: bool = False,
        prepared_text: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        if contextualized_prototypes.ndim != 2:
            raise ValueError('contextualized_prototypes must have shape [N, D].')

        text_inputs = prepared_text or self._prepare_text_inputs(
            text_token_states,
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
        )
        text_features = text_inputs['text_token_states']
        mask_debug = text_inputs['mask_debug']
        batch_size, seq_len, feature_dim = text_features.shape
        num_prototypes = contextualized_prototypes.size(0)

        expanded_queries = contextualized_prototypes.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * num_prototypes, feature_dim)
        expanded_tokens = text_features.unsqueeze(1).expand(batch_size, num_prototypes, seq_len, feature_dim).reshape(batch_size * num_prototypes, seq_len, feature_dim)
        expanded_keep_mask = mask_debug['token_keep_mask'].unsqueeze(1).expand(batch_size, num_prototypes, seq_len).reshape(batch_size * num_prototypes, seq_len)

        basis_scores = self.token_scorer(expanded_queries, expanded_tokens)
        pooled_basis, basis_weights, basis_pooler_debug = self.token_pooler(
            basis_scores,
            expanded_tokens,
            expanded_keep_mask,
            return_debug=True,
        )

        outputs = {
            'text_token_states': text_features,
            'token_valid_mask': mask_debug['token_valid_mask'],
            'token_keep_mask': mask_debug['token_keep_mask'],
            'special_token_positions': mask_debug['special_token_positions'],
            'basis_bank': pooled_basis.view(batch_size, num_prototypes, feature_dim),
        }
        if return_debug:
            outputs['basis_token_scores'] = basis_scores.view(batch_size, num_prototypes, seq_len)
            outputs['basis_token_weights'] = basis_weights.view(batch_size, num_prototypes, seq_len)
            outputs['basis_beta_logits_masked'] = basis_pooler_debug['beta_logits_masked'].view(batch_size, num_prototypes, seq_len)
            outputs['debug'] = {
                **mask_debug,
                'basis_token_scores': outputs['basis_token_scores'],
                'basis_token_weights': outputs['basis_token_weights'],
                'basis_beta_logits_masked': outputs['basis_beta_logits_masked'],
            }
        return outputs

    def reconstruct_surrogate_text(self, routing_weights: torch.Tensor, basis_bank: torch.Tensor) -> torch.Tensor:
        if routing_weights.ndim != 2:
            raise ValueError('routing_weights must have shape [B, N].')
        if basis_bank.ndim != 3:
            raise ValueError('basis_bank must have shape [B, N, D].')
        if routing_weights.shape[:2] != basis_bank.shape[:2]:
            raise ValueError('routing_weights and basis_bank must agree on [B, N].')
        surrogate = torch.einsum('bn,bnd->bd', routing_weights, basis_bank)
        if not torch.isfinite(surrogate).all():
            raise FloatingPointError('Surrogate text reconstruction produced NaN or Inf values.')
        return surrogate

    def compute_approximate_pairwise_similarity(
        self,
        image_projected: torch.Tensor,
        routing_weights: torch.Tensor,
        basis_bank: torch.Tensor,
        image_chunk_size: int = 32,
        text_chunk_size: int = 128,
    ) -> torch.Tensor:
        if image_projected.ndim != 2 or routing_weights.ndim != 2:
            raise ValueError('image_projected and routing_weights must have shape [N, D] and [N, P].')
        if basis_bank.ndim != 3:
            raise ValueError('basis_bank must have shape [B, P, D].')
        if image_projected.size(0) != routing_weights.size(0):
            raise ValueError('image_projected and routing_weights must have the same image batch dimension.')
        if routing_weights.size(1) != basis_bank.size(1):
            raise ValueError('routing_weights prototype dimension must match basis_bank prototype dimension.')

        image_chunk_size = max(int(image_chunk_size), 1)
        text_chunk_size = max(int(text_chunk_size), 1)
        num_text = basis_bank.size(0)
        num_image = image_projected.size(0)
        similarity = torch.empty(num_text, num_image, device=image_projected.device, dtype=image_projected.dtype)

        for image_start in range(0, num_image, image_chunk_size):
            image_end = min(image_start + image_chunk_size, num_image)
            image_projected_chunk = image_projected[image_start:image_end]
            routing_chunk = routing_weights[image_start:image_end]
            image_batch = image_projected_chunk.size(0)

            for text_start in range(0, num_text, text_chunk_size):
                text_end = min(text_start + text_chunk_size, num_text)
                basis_chunk = basis_bank[text_start:text_end]
                text_batch = basis_chunk.size(0)

                surrogate_chunk = torch.einsum('in,tnd->tid', routing_chunk, basis_chunk)
                projected_text = self.text_projector(surrogate_chunk.reshape(text_batch * image_batch, -1))
                expanded_image_projected = image_projected_chunk.unsqueeze(0).expand(text_batch, image_batch, -1).reshape(text_batch * image_batch, -1)
                block_similarity = self.losses.compute_paired_similarity(expanded_image_projected, projected_text)
                similarity[text_start:text_end, image_start:image_end] = block_similarity.view(text_batch, image_batch)

        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Approximate prototype retrieval similarity contains NaN or Inf values.')
        return similarity

    def _compute_exact_pairwise_similarity_logits(
        self,
        image_projected: torch.Tensor,
        summaries: torch.Tensor,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
        image_chunk_size: int = 32,
        text_chunk_size: int = 128,
        prepared_text: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        if image_projected.ndim != 2 or summaries.ndim != 2:
            raise ValueError('image_projected and summaries must have shape [N, D].')
        if text_token_states.ndim != 3 or token_ids.ndim != 2:
            raise ValueError('text_token_states must have shape [B, L, D] and token_ids must have shape [B, L].')
        if image_projected.size(0) != summaries.size(0):
            raise ValueError('image_projected and summaries must have the same batch dimension.')
        if text_token_states.size(0) != token_ids.size(0):
            raise ValueError('text_token_states and token_ids must have the same batch dimension.')

        image_chunk_size = max(int(image_chunk_size), 1)
        text_chunk_size = max(int(text_chunk_size), 1)
        text_inputs = prepared_text or self._prepare_text_inputs(
            text_token_states,
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
        )
        text_features = text_inputs['text_token_states']
        token_keep_mask = text_inputs['mask_debug']['token_keep_mask']

        num_text = text_features.size(0)
        num_image = summaries.size(0)
        similarity = torch.empty(num_text, num_image, device=image_projected.device, dtype=image_projected.dtype)
        for image_start in range(0, num_image, image_chunk_size):
            image_end = min(image_start + image_chunk_size, num_image)
            summary_chunk = summaries[image_start:image_end]
            image_projected_chunk = image_projected[image_start:image_end]
            image_batch = summary_chunk.size(0)

            for text_start in range(0, num_text, text_chunk_size):
                text_end = min(text_start + text_chunk_size, num_text)
                token_chunk = text_features[text_start:text_end]
                mask_chunk = token_keep_mask[text_start:text_end]
                text_batch = token_chunk.size(0)

                expanded_summary = summary_chunk[:, None, :].expand(image_batch, text_batch, -1).reshape(image_batch * text_batch, -1)
                text_pool_queries = self._build_text_pool_queries(image_batch * text_batch, image_projected.device, text_token_states.dtype, expanded_summary if self.use_image_conditioned_pooling else None)
                expanded_tokens = token_chunk[None, :, :, :].expand(image_batch, text_batch, -1, -1).reshape(image_batch * text_batch, token_chunk.size(1), token_chunk.size(2))
                expanded_mask = mask_chunk[None, :, :].expand(image_batch, text_batch, -1).reshape(image_batch * text_batch, mask_chunk.size(1))
                expanded_image_projected = image_projected_chunk[:, None, :].expand(image_batch, text_batch, -1).reshape(image_batch * text_batch, -1)

                token_scores = self.token_scorer(text_pool_queries, expanded_tokens)
                pooled_text, _ = self.token_pooler(token_scores, expanded_tokens, expanded_mask)
                projected_text = self.text_projector(pooled_text)
                block_similarity = self.losses.compute_paired_similarity(expanded_image_projected, projected_text)
                similarity[text_start:text_end, image_start:image_end] = block_similarity.view(image_batch, text_batch).t()

        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Prototype retrieval similarity contains NaN or Inf values.')
        return similarity


    def compute_surrogate_pairwise_logits(
        self,
        image_projected: torch.Tensor,
        routing_weights: torch.Tensor,
        basis_bank: torch.Tensor,
        image_chunk_size: int = 32,
        text_chunk_size: int = 128,
    ) -> torch.Tensor:
        similarity = self.compute_approximate_pairwise_similarity(
            image_projected=image_projected,
            routing_weights=routing_weights,
            basis_bank=basis_bank,
            image_chunk_size=image_chunk_size,
            text_chunk_size=text_chunk_size,
        )
        surrogate_pairwise_logits = similarity.t().contiguous()
        if surrogate_pairwise_logits.ndim != 2 or surrogate_pairwise_logits.shape != (image_projected.size(0), basis_bank.size(0)):
            raise ValueError(
                'compute_surrogate_pairwise_logits must return shape [B_image, B_text]; '
                f'got {tuple(surrogate_pairwise_logits.shape)}.'
            )
        if not torch.isfinite(surrogate_pairwise_logits).all():
            raise FloatingPointError('Surrogate pairwise logits contain NaN or Inf values.')
        return surrogate_pairwise_logits

    def compute_pairwise_similarity(
        self,
        image_projected: torch.Tensor,
        summaries: torch.Tensor,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
        image_chunk_size: int = 32,
        text_chunk_size: int = 128,
    ) -> torch.Tensor:
        del pids
        return self._compute_exact_pairwise_similarity_logits(
            image_projected=image_projected,
            summaries=summaries,
            text_token_states=text_token_states,
            token_ids=token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
            image_chunk_size=image_chunk_size,
            text_chunk_size=text_chunk_size,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        image_local_tokens: Optional[torch.Tensor] = None,
        pids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
        host_pairwise_logits: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        current_step: Optional[int] = None,
        return_debug: bool = False,
        disable_proxy_losses: bool = False,
    ) -> Dict[str, object]:
        semantic_recompute_features = self.image_adapter(image_embeddings.detach())
        context = self.get_prototype_context(
            return_debug=return_debug,
            epoch=epoch,
            current_step=current_step,
            semantic_recompute_features=semantic_recompute_features.detach(),
        )
        image_outputs = self.encode_image_branch(
            image_embeddings,
            image_local_tokens=image_local_tokens,
            prototypes=context['base_prototypes'],
            contextualized_prototypes=context['contextualized_prototypes'],
            routing_prototypes=context['routing_prototypes'],
            return_debug=return_debug,
        )
        prepared_text = self._prepare_text_inputs(
            text_token_states,
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
        )
        basis_outputs = self.build_text_basis_bank(
            text_token_states,
            token_ids,
            context['routing_prototypes'],
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
            return_debug=return_debug,
            prepared_text=prepared_text,
        )
        surrogate_pooled_text = self.reconstruct_surrogate_text(image_outputs['routing_weights'], basis_outputs['basis_bank'])
        surrogate_text_projected, surrogate_text_projector_debug = self.text_projector(surrogate_pooled_text, return_debug=True)
        exact_outputs = self.pool_text_with_summary(
            image_outputs['summary'],
            text_token_states,
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
            return_debug=return_debug,
            prepared_text=prepared_text,
        )
        collect_debug_diagnostics = self.collect_debug_metrics or bool(return_debug)
        scalar_metrics = {}
        if collect_debug_diagnostics:
            scalar_metrics = self._collect_scalar_metrics(
                prototypes=context['base_prototypes'],
                contextualized_prototypes=context['contextualized_prototypes'],
                routing_weights=image_outputs['routing_weights'],
                summary=image_outputs['summary'],
                exact_token_weights=exact_outputs['token_weights'],
                token_valid_mask=exact_outputs['token_valid_mask'],
                token_keep_mask=exact_outputs['token_keep_mask'],
                surrogate_pooled_text=surrogate_pooled_text,
                exact_pooled_text=exact_outputs['pooled_text'],
                image_features=image_outputs['image_embedding'],
                image_projector_debug=image_outputs['image_projector_debug'],
                surrogate_text_projector_debug=surrogate_text_projector_debug,
                exact_text_projector_debug=exact_outputs['text_projector_debug'],
                special_token_positions=exact_outputs['special_token_positions'],
                contextualizer_debug=context.get('contextualizer_debug'),
                router_debug=image_outputs['router_debug'],
                pooler_debug=exact_outputs['pooler_debug'],
            )
            scalar_metrics.update(
                self._compute_routing_certification_metrics(
                    routing_weights=image_outputs['routing_weights'],
                    basis_bank=basis_outputs['basis_bank'],
                    surrogate_text_projected=surrogate_text_projected,
                    exact_text_projected=exact_outputs['text_projected'],
                )
            )
        semantic_cluster_counts = context.get('semantic_cluster_counts')
        if isinstance(semantic_cluster_counts, torch.Tensor) and semantic_cluster_counts.numel() > 0:
            semantic_active_clusters = semantic_cluster_counts.gt(0).sum().to(dtype=image_outputs['image_projected'].dtype)
            semantic_empty_clusters = semantic_cluster_counts.eq(0).sum().to(dtype=image_outputs['image_projected'].dtype)
        else:
            semantic_active_clusters = image_outputs['image_projected'].new_zeros(())
            semantic_empty_clusters = image_outputs['image_projected'].new_zeros(())
        scalar_metrics.update(
            {
                'prototype_method_role_semantic_structure': image_outputs['image_projected'].new_tensor(
                    float(self.prototype_method_role == 'semantic_structure')
                ),
                'prototype_semantic_enabled': image_outputs['image_projected'].new_tensor(float(self.prototype_semantic_enabled)),
                'semantic_structure_enabled': image_outputs['image_projected'].new_tensor(float(self.semantic_structure_enabled)),
                'prototype_source_recomputed': image_outputs['image_projected'].new_tensor(
                    float(context.get('prototype_source_type') == 'recomputed_kmeans')
                ),
                'semantic_recompute_count': context.get('semantic_recompute_count', image_outputs['image_projected'].new_zeros(())),
                'semantic_recompute_last_epoch': context.get('semantic_recompute_last_epoch', image_outputs['image_projected'].new_full((), -1.0)),
                'semantic_recompute_last_step': context.get('semantic_recompute_last_step', image_outputs['image_projected'].new_full((), -1.0)),
                'semantic_recompute_triggered': context.get('semantic_recompute_triggered', image_outputs['image_projected'].new_zeros(())),
                'semantic_recompute_skipped_no_features': context.get('semantic_recompute_skipped_no_features', image_outputs['image_projected'].new_zeros(())),
                'semantic_empty_cluster_reseed_events': context.get('semantic_empty_cluster_reseed_events', image_outputs['image_projected'].new_zeros(())),
                'semantic_active_cluster_count': semantic_active_clusters.detach(),
                'semantic_empty_cluster_count': semantic_empty_clusters.detach(),
            }
        )
        prototype_usage_enabled = True
        if self.semantic_ramp_use_prototype and self._semantic_mode_enabled():
            prototype_usage_enabled = self._semantic_schedule_started(epoch=epoch, current_step=current_step)
        scalar_metrics['prototype_ramp_use_enabled'] = image_outputs['image_projected'].new_tensor(
            float(prototype_usage_enabled)
        ).detach()

        surrogate_pairwise_logits = None
        semantic_target_prototypes = (
            context['base_prototypes']
            if self.prototype_use_base_for_semantic_targets
            else context['contextualized_prototypes']
        )
        semantic_target_features, semantic_anchor_feature_source = self._prepare_semantic_anchor_features(
            semantic_target_prototypes,
            target_feature_dim=exact_outputs['text_projected'].size(-1),
        )
        scalar_metrics.update(
            {
                'semantic_anchor_input_dim': semantic_target_prototypes.new_tensor(float(semantic_target_prototypes.size(-1))).detach(),
                'semantic_anchor_feature_dim': semantic_target_features.new_tensor(float(semantic_target_features.size(-1))).detach(),
                'semantic_anchor_space_projected': semantic_target_features.new_tensor(
                    float(semantic_anchor_feature_source in {'prototype_projected', 'auto_projected_dim_fix'})
                ).detach(),
            }
        )
        ramp_scale = self._prototype_loss_scale(epoch=epoch, current_step=current_step)
        diag_loss_scale = ramp_scale if self.semantic_ramp_loss_diag else 1.0
        semantic_pbt_loss_scale = ramp_scale if self.semantic_ramp_loss_semantic_pbt else 1.0
        if not prototype_usage_enabled:
            ramp_scale = 0.0
            diag_loss_scale = 0.0
            semantic_pbt_loss_scale = 0.0
            loss_outputs = self._zero_loss_outputs(image_outputs['image_projected'])
        else:
            loss_outputs = self.losses(
                image_outputs['image_projected'],
                surrogate_text_projected,
                exact_outputs['text_projected'],
                pids=pids,
                prototypes=context['base_prototypes'],
                routing_weights=image_outputs['routing_weights'],
                surrogate_pairwise_logits=surrogate_pairwise_logits,
                host_pairwise_logits=host_pairwise_logits,
                semantic_image_student_embeddings=image_outputs['image_projected'],
                semantic_text_student_embeddings=surrogate_text_projected,
                semantic_text_teacher_embeddings=exact_outputs['text_projected'],
                semantic_base_prototypes=semantic_target_features,
                diag_loss_scale=diag_loss_scale,
                semantic_pbt_loss_scale=semantic_pbt_loss_scale,
                return_debug=return_debug,
                disable_proxy_losses=disable_proxy_losses,
            )
        if collect_debug_diagnostics:
            scalar_metrics.update(loss_outputs.get('debug_metrics', {}))

        outputs = {
            'image_embedding': image_outputs['image_embedding'],
            'text_token_states': prepared_text['text_token_states'],
            'legacy_prototypes': context['legacy_prototypes'],
            'base_prototypes': context['base_prototypes'],
            'prototypes': context['base_prototypes'],
            'contextualized_prototypes': context['contextualized_prototypes'],
            'routing_prototypes': context['routing_prototypes'],
            'routing_weights': image_outputs['routing_weights'],
            'summary': image_outputs['summary'],
            'token_valid_mask': exact_outputs['token_valid_mask'],
            'token_keep_mask': exact_outputs['token_keep_mask'],
            'valid_mask': exact_outputs['valid_mask'],
            'basis_bank': basis_outputs['basis_bank'],
            'exact_token_scores': exact_outputs['token_scores'],
            'exact_token_weights': exact_outputs['token_weights'],
            'beta_logits_masked': exact_outputs['beta_logits_masked'],
            'exact_pooled_text': exact_outputs['pooled_text'],
            'surrogate_pooled_text': surrogate_pooled_text,
            'image_projected': image_outputs['image_projected'],
            'image_projected_raw': image_outputs['image_projected_raw'],
            'image_proxy_features': image_outputs['image_proxy_features'],
            'semantic_image_feature': image_outputs['image_projected'],
            'surrogate_text_projected': surrogate_text_projected,
            'surrogate_text_projected_raw': surrogate_text_projector_debug['projected_features_raw'],
            'semantic_text_student_feature': surrogate_text_projected,
            'exact_text_projected': exact_outputs['text_projected'],
            'exact_text_projected_raw': exact_outputs['text_projected_raw'],
            'semantic_text_teacher_feature': exact_outputs['text_projected'],
            'alpha': image_outputs['routing_weights'],
            'beta': exact_outputs['token_weights'],
            'Q': image_outputs['summary'],
            'Theta_v': context['base_prototypes'],
            'Theta_tilde': context['contextualized_prototypes'],
            'T_pool': surrogate_pooled_text,
            'T_exact_pool': exact_outputs['pooled_text'],
            'T_hat_pool': surrogate_pooled_text,
            'Z_v': image_outputs['image_projected'],
            'Z_v_raw': image_outputs['image_projected_raw'],
            'Z_t': surrogate_text_projected,
            'Z_t_raw': surrogate_text_projector_debug['projected_features_raw'],
            'Z_t_exact': exact_outputs['text_projected'],
            'Z_t_exact_raw': exact_outputs['text_projected_raw'],
            'surrogate_pairwise_logits': surrogate_pairwise_logits,
            'prototype_source_type': context.get('prototype_source_type', 'learnable_legacy'),
            'semantic_cluster_counts': context.get('semantic_cluster_counts'),
            'prototype_loss_scale': image_outputs['image_projected'].new_tensor(float(ramp_scale)),
            'prototype_loss_ramp_scale': image_outputs['image_projected'].new_tensor(float(ramp_scale)),
            'loss_diag_scale': image_outputs['image_projected'].new_tensor(float(diag_loss_scale)),
            'loss_semantic_pbt_scale': image_outputs['image_projected'].new_tensor(float(semantic_pbt_loss_scale)),
            # Backward-compatible alias for existing logs/consumers.
            'semantic_loss_scale': image_outputs['image_projected'].new_tensor(float(semantic_pbt_loss_scale)),
            'losses': loss_outputs,
            'metrics': scalar_metrics,
        }
        outputs['debug'] = dict(scalar_metrics)
        if return_debug:
            outputs['debug'].update(
                {
                    **context['debug'],
                    **image_outputs['debug'],
                    **exact_outputs['debug'],
                    **basis_outputs.get('debug', {}),
                    'alpha': image_outputs['routing_weights'].detach(),
                    'beta': exact_outputs['token_weights'].detach(),
                    'Q': image_outputs['summary'].detach(),
                    'Theta_v': context['prototypes'].detach(),
                    'Theta_tilde': context['contextualized_prototypes'].detach(),
                    'T_pool': surrogate_pooled_text.detach(),
                    'T_exact_pool': exact_outputs['pooled_text'].detach(),
                    'T_hat_pool': surrogate_pooled_text.detach(),
                    'Z_v': image_outputs['image_projected'].detach(),
                    'Z_v_raw': image_outputs['image_projected_raw'].detach(),
                    'Z_t': surrogate_text_projected.detach(),
                    'Z_t_raw': surrogate_text_projector_debug['projected_features_raw'].detach(),
                    'Z_t_exact': exact_outputs['text_projected'].detach(),
                    'Z_t_exact_raw': exact_outputs['text_projected_raw'].detach(),
                    'token_valid_mask': exact_outputs['token_valid_mask'].detach(),
                    'token_keep_mask': exact_outputs['token_keep_mask'].detach(),
                    'beta_logits_masked': exact_outputs['beta_logits_masked'].detach(),
                    'basis_bank': basis_outputs['basis_bank'].detach(),
                }
            )
            if surrogate_pairwise_logits is not None:
                outputs['debug']['surrogate_pairwise_logits'] = surrogate_pairwise_logits.detach()
            if 'surrogate_retrieval_logits' in loss_outputs:
                outputs['debug']['surrogate_retrieval_logits'] = loss_outputs['surrogate_retrieval_logits'].detach()
            if 'basis_token_scores' in basis_outputs:
                outputs['debug']['basis_token_scores'] = basis_outputs['basis_token_scores'].detach()
                outputs['debug']['basis_token_weights'] = basis_outputs['basis_token_weights'].detach()
                outputs['debug']['basis_beta_logits_masked'] = basis_outputs['basis_beta_logits_masked'].detach()
            if 'image_proxy_logits' in loss_outputs:
                outputs['debug']['image_proxy_logits'] = loss_outputs['image_proxy_logits'].detach()
                outputs['debug']['text_proxy_logits'] = loss_outputs['text_proxy_logits'].detach()
                outputs['debug']['text_exact_proxy_logits'] = loss_outputs['text_exact_proxy_logits'].detach()
                outputs['debug']['class_proxies'] = loss_outputs['class_proxies']
        return outputs
