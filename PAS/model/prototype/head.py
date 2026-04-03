from typing import Dict, Optional

import torch
import torch.nn as nn

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
        num_classes: int = 0,
        proxy_temperature: float = 0.07,
        lambda_proxy: float = 1.0,
        use_loss_proxy_image: bool = True,
        use_loss_proxy_text: bool = True,
        use_loss_proxy_text_exact: bool = True,
        use_loss_align: bool = True,
        lambda_align: float = 1.0,
        use_loss_diag: bool = True,
        lambda_diag: float = 1.0,
        use_diversity_loss: bool = False,
        diversity_loss_weight: float = 0.0,
        use_balance_loss: bool = False,
        balance_loss_weight: float = 0.0,
        contrastive_temperature_init: float = 0.07,
        learnable_contrastive_temperature: bool = False,
        dead_prototype_threshold: float = 0.005,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.prototype_dim = int(prototype_dim)
        self.projector_hidden_dim = int(projector_hidden_dim or prototype_dim)
        self.projector_output_dim = int(projector_output_dim)
        self.dead_prototype_threshold = float(dead_prototype_threshold)

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
            lambda_proxy=lambda_proxy,
            use_loss_proxy_image=use_loss_proxy_image,
            use_loss_proxy_text=use_loss_proxy_text,
            use_loss_proxy_text_exact=use_loss_proxy_text_exact,
            use_loss_align=use_loss_align,
            lambda_align=lambda_align,
            use_loss_diag=use_loss_diag,
            lambda_diag=lambda_diag,
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
        return {
            'prototype_usage': usage.detach(),
            'prototype_usage_entropy': (-(usage * usage.clamp_min(1e-12).log()).sum()).detach(),
            'prototype_usage_max': usage.max().detach(),
            'prototype_dead_count': (usage < self.dead_prototype_threshold).sum().detach(),
            'routing_entropy': (-(routing_weights * routing_weights.clamp_min(1e-12).log()).sum(dim=-1).mean()).detach(),
            'routing_top1_histogram': top1_usage.detach(),
            'routing_top1_usage_entropy': (-(top1_usage * top1_usage.clamp_min(1e-12).log()).sum()).detach(),
            'routing_top1_usage_max': top1_usage.max().detach(),
            'routing_top1_dead_count': (top1_usage == 0).sum().detach(),
        }

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
        if pooler_debug:
            metrics['token_pool_entropy'] = pooler_debug['token_pool_entropy']
            metrics['beta_max_prob'] = pooler_debug['beta_max_prob']
        return metrics

    def get_prototype_context(self, return_debug: bool = False) -> Dict[str, object]:
        prototypes, bank_debug = self.prototype_bank(return_debug=True)
        contextualized, contextual_debug = self.contextualizer(prototypes, return_debug=True)
        outputs = {
            'prototypes': prototypes,
            'contextualized_prototypes': contextualized,
            'bank_debug': bank_debug,
            'contextualizer_debug': contextual_debug,
        }
        if return_debug:
            outputs['debug'] = {
                **bank_debug,
                **contextual_debug,
            }
        return outputs

    def encode_image_branch(
        self,
        image_embeddings: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
        contextualized_prototypes: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Dict[str, object]:
        image_features = self.image_adapter(image_embeddings)
        context_debug = {}

        if prototypes is None or contextualized_prototypes is None:
            context = self.get_prototype_context(return_debug=return_debug)
            prototypes = context['prototypes']
            contextualized_prototypes = context['contextualized_prototypes']
            context_debug = {
                **context['bank_debug'],
                **context['contextualizer_debug'],
            }

        routing_weights, routing_debug = self.router(image_features, contextualized_prototypes, return_debug=True)
        summary, aggregator_debug = self.aggregator(routing_weights, contextualized_prototypes, return_debug=True)
        image_proxy_features = image_features + summary
        image_projected, image_projector_debug = self.image_projector(image_proxy_features, return_debug=True)

        outputs = {
            'image_embedding': image_features,
            'image_proxy_features': image_proxy_features,
            'prototypes': prototypes,
            'contextualized_prototypes': contextualized_prototypes,
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

    def pool_text_with_summary(
        self,
        summary: torch.Tensor,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
        return_debug: bool = False,
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
        token_scores, scorer_debug = self.token_scorer(summary, text_features, return_debug=True)
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
        text_features = self.text_adapter(text_token_states)
        token_keep_mask = self.token_mask_builder.build(
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
            return_debug=False,
        )

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
                expanded_tokens = token_chunk[None, :, :, :].expand(image_batch, text_batch, -1, -1).reshape(image_batch * text_batch, token_chunk.size(1), token_chunk.size(2))
                expanded_mask = mask_chunk[None, :, :].expand(image_batch, text_batch, -1).reshape(image_batch * text_batch, mask_chunk.size(1))
                expanded_image_projected = image_projected_chunk[:, None, :].expand(image_batch, text_batch, -1).reshape(image_batch * text_batch, -1)

                token_scores = self.token_scorer(expanded_summary, expanded_tokens)
                pooled_text, _ = self.token_pooler(token_scores, expanded_tokens, expanded_mask)
                projected_text = self.text_projector(pooled_text)
                block_similarity = self.losses.compute_paired_similarity(expanded_image_projected, projected_text)
                block_similarity = block_similarity.view(image_batch, text_batch).t()
                similarity[text_start:text_end, image_start:image_end] = block_similarity

        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Prototype retrieval similarity contains NaN or Inf values.')
        return similarity

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
        return_debug: bool = False,
    ) -> Dict[str, object]:
        context = self.get_prototype_context(return_debug=return_debug)
        image_outputs = self.encode_image_branch(
            image_embeddings,
            prototypes=context['prototypes'],
            contextualized_prototypes=context['contextualized_prototypes'],
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
            context['contextualized_prototypes'],
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
        scalar_metrics = self._collect_scalar_metrics(
            prototypes=context['prototypes'],
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
        loss_outputs = self.losses(
            image_outputs['image_projected'],
            surrogate_text_projected,
            exact_outputs['text_projected'],
            pids=pids,
            prototypes=context['prototypes'],
            routing_weights=image_outputs['routing_weights'],
            return_debug=return_debug,
        )
        scalar_metrics.update(loss_outputs.get('debug_metrics', {}))

        outputs = {
            'image_embedding': image_outputs['image_embedding'],
            'text_token_states': prepared_text['text_token_states'],
            'prototypes': context['prototypes'],
            'contextualized_prototypes': context['contextualized_prototypes'],
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
            'surrogate_text_projected': surrogate_text_projected,
            'surrogate_text_projected_raw': surrogate_text_projector_debug['projected_features_raw'],
            'exact_text_projected': exact_outputs['text_projected'],
            'exact_text_projected_raw': exact_outputs['text_projected_raw'],
            'alpha': image_outputs['routing_weights'],
            'beta': exact_outputs['token_weights'],
            'Q': image_outputs['summary'],
            'Theta_v': context['prototypes'],
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



