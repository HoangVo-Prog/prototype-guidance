from typing import Dict, Optional

import torch
import torch.nn as nn

from .losses import PrototypeLosses
from .projector import MLPProjector
from .token_mask import TokenMaskBuilder
from .token_pooler import MaskedTokenPooler
from .token_scorer import TokenScorer


class DirectImageConditionedTextHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        prototype_dim: int,
        projector_output_dim: int,
        projector_hidden_dim: Optional[int] = None,
        projector_dropout: float = 0.0,
        projector_type: str = 'mlp2',
        image_adapter: Optional[nn.Module] = None,
        text_adapter: Optional[nn.Module] = None,
        token_scoring_type: str = 'cosine',
        token_temperature: float = 0.07,
        token_policy: str = 'content_only',
        special_token_ids: Optional[Dict[str, object]] = None,
        error_on_empty_kept_tokens: bool = True,
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
        contrastive_temperature_init: float = 0.07,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.prototype_dim = int(prototype_dim)
        self.projector_hidden_dim = int(projector_hidden_dim or prototype_dim)
        self.projector_output_dim = int(projector_output_dim)
        self.use_image_conditioned_pooling = bool(use_image_conditioned_pooling)
        self.uses_prototype_bank = False
        self.prototype_method_role = str(prototype_method_role).lower()
        self.prototype_semantic_enabled = bool(prototype_semantic_enabled)
        self.semantic_structure_enabled = bool(semantic_structure_enabled)
        self.semantic_feature_space = str(semantic_feature_space).lower()
        self.semantic_pbt_enabled = bool(semantic_pbt_enabled)

        self.image_adapter = image_adapter if image_adapter is not None else (nn.Identity() if self.input_dim == self.prototype_dim else nn.Linear(self.input_dim, self.prototype_dim))
        self.text_adapter = text_adapter if text_adapter is not None else (nn.Identity() if self.input_dim == self.prototype_dim else nn.Linear(self.input_dim, self.prototype_dim))
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
            learnable_temperature=False,
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
            use_loss_semantic_hosthard_weighted=use_loss_semantic_hosthard_weighted,
            lambda_semantic_hosthard_weighted=lambda_semantic_hosthard_weighted,
            semantic_hosthard_margin_ref=semantic_hosthard_margin_ref,
            semantic_hosthard_tau=semantic_hosthard_tau,
            semantic_hosthard_eps=semantic_hosthard_eps,
            semantic_hosthard_normalize_weights=semantic_hosthard_normalize_weights,
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
            use_diversity_loss=False,
            diversity_loss_weight=0.0,
            use_balance_loss=False,
            balance_loss_weight=0.0,
        )

    def _empty_context(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'prototypes': reference.new_empty((0, self.prototype_dim)),
            'contextualized_prototypes': reference.new_empty((0, self.prototype_dim)),
        }

    def _pooling_mode_metrics(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'direct_image_conditioned_pooling': reference.new_tensor(float(self.use_image_conditioned_pooling)),
            'direct_non_image_conditioned_pooling': reference.new_tensor(float(not self.use_image_conditioned_pooling)),
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

    def get_prototype_context(self, return_debug: bool = False) -> Dict[str, object]:
        outputs = {
            'prototypes': torch.empty(0, self.prototype_dim),
            'contextualized_prototypes': torch.empty(0, self.prototype_dim),
            'bank_debug': {},
            'contextualizer_debug': {},
        }
        if return_debug:
            outputs['debug'] = self._pooling_mode_metrics(torch.zeros((), dtype=torch.float32))
        return outputs

    def encode_image_branch(
        self,
        image_embeddings: torch.Tensor,
        image_local_tokens: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None,
        contextualized_prototypes: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Dict[str, object]:
        del image_local_tokens, prototypes, contextualized_prototypes
        image_features = self.image_adapter(image_embeddings)
        summary = image_features
        image_projected, image_projector_debug = self.image_projector(image_features, return_debug=True)
        empty_context = self._empty_context(image_features)
        routing_weights = image_features.new_empty((image_features.size(0), 0))
        outputs = {
            'image_embedding': image_features,
            'image_proxy_features': image_features,
            'prototypes': empty_context['prototypes'],
            'contextualized_prototypes': empty_context['contextualized_prototypes'],
            'routing_weights': routing_weights,
            'summary': summary,
            'image_projected': image_projected,
            'image_projected_raw': image_projector_debug['projected_features_raw'],
            'router_debug': {},
            'aggregator_debug': {},
            'image_projector_debug': image_projector_debug,
        }
        if return_debug:
            outputs['debug'] = self._pooling_mode_metrics(image_features)
        return outputs

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
        del pids, disable_proxy_losses
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
            'mask_debug': mask_debug,
            'scorer_debug': scorer_debug,
            'pooler_debug': pooler_debug,
            'text_projector_debug': text_projector_debug,
        }
        if return_debug:
            outputs['debug'] = {
                **mask_debug,
                **scorer_debug,
                **pooler_debug,
                **text_projector_debug,
                **self._pooling_mode_metrics(text_features),
            }
        return outputs

    def build_text_basis_bank(self, *args, **kwargs) -> Dict[str, object]:
        del args, kwargs
        raise RuntimeError(
            'Approximate prototype basis construction is unavailable when model.use_prototype_bank=false. '
            'Retrieval scoring is host-only exact.'
        )

    def compute_approximate_pairwise_similarity(self, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        raise RuntimeError(
            'Approximate prototype retrieval similarity is unavailable when model.use_prototype_bank=false. '
            'Retrieval scoring is host-only exact.'
        )

    def _compute_pairwise_similarity_logits(
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
                expanded_tokens = token_chunk[None, :, :, :].expand(image_batch, text_batch, -1, -1).reshape(image_batch * text_batch, token_chunk.size(1), token_chunk.size(2))
                expanded_mask = mask_chunk[None, :, :].expand(image_batch, text_batch, -1).reshape(image_batch * text_batch, mask_chunk.size(1))
                expanded_image_projected = image_projected_chunk[:, None, :].expand(image_batch, text_batch, -1).reshape(image_batch * text_batch, -1)
                text_pool_queries = self._build_text_pool_queries(image_batch * text_batch, expanded_tokens.device, expanded_tokens.dtype, expanded_summary if self.use_image_conditioned_pooling else None)

                token_scores = self.token_scorer(text_pool_queries, expanded_tokens)
                pooled_text, _ = self.token_pooler(token_scores, expanded_tokens, expanded_mask)
                projected_text = self.text_projector(pooled_text)
                block_similarity = self.losses.compute_paired_similarity(expanded_image_projected, projected_text)
                similarity[text_start:text_end, image_start:image_end] = block_similarity.view(image_batch, text_batch).t()

        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Direct retrieval similarity contains NaN or Inf values.')
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
        return self._compute_pairwise_similarity_logits(
            image_projected=image_projected,
            summaries=summaries,
            text_token_states=text_token_states,
            token_ids=token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
            image_chunk_size=image_chunk_size,
            text_chunk_size=text_chunk_size,
        )

    def _collect_scalar_metrics(
        self,
        image_outputs: Dict[str, object],
        exact_outputs: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        image_projector_debug = image_outputs['image_projector_debug']
        text_projector_debug = exact_outputs['text_projector_debug']
        image_embed_norms = image_projector_debug['projected_features_raw'].norm(dim=-1)
        image_embed_unit_norms = image_projector_debug['projected_features'].norm(dim=-1)
        text_embed_norms = text_projector_debug['projected_features_raw'].norm(dim=-1)
        text_embed_unit_norms = text_projector_debug['projected_features'].norm(dim=-1)
        metrics = {
            'prototype_method_role_semantic_structure': image_outputs['summary'].new_tensor(
                float(self.prototype_method_role == 'semantic_structure')
            ).detach(),
            'prototype_semantic_enabled': image_outputs['summary'].new_tensor(float(self.prototype_semantic_enabled)).detach(),
            'semantic_structure_enabled': image_outputs['summary'].new_tensor(float(self.semantic_structure_enabled)).detach(),
            'q_norm': image_outputs['summary'].norm(dim=-1).mean().detach(),
            't_pool_norm': exact_outputs['pooled_text'].norm(dim=-1).mean().detach(),
            'surrogate_t_pool_norm': exact_outputs['pooled_text'].norm(dim=-1).mean().detach(),
            'exact_t_pool_norm': exact_outputs['pooled_text'].norm(dim=-1).mean().detach(),
            'image_feature_norm': image_outputs['image_embedding'].norm(dim=-1).mean().detach(),
            'image_embed_norm': image_embed_norms.mean().detach(),
            'image_embed_norm_raw': image_embed_norms.mean().detach(),
            'image_embed_unit_norm': image_embed_unit_norms.mean().detach(),
            'image_embed_norm_std': image_embed_norms.std(unbiased=False).detach(),
            'image_embed_norm_min': image_embed_norms.min().detach(),
            'image_embed_norm_max': image_embed_norms.max().detach(),
            'text_embed_norm': text_embed_norms.mean().detach(),
            'text_embed_norm_raw': text_embed_norms.mean().detach(),
            'text_embed_unit_norm': text_embed_unit_norms.mean().detach(),
            'text_embed_norm_std': text_embed_norms.std(unbiased=False).detach(),
            'text_embed_norm_min': text_embed_norms.min().detach(),
            'text_embed_norm_max': text_embed_norms.max().detach(),
            'surrogate_text_embed_norm': text_embed_norms.mean().detach(),
            'surrogate_text_embed_norm_raw': text_embed_norms.mean().detach(),
            'surrogate_text_embed_unit_norm': text_embed_unit_norms.mean().detach(),
            'surrogate_text_embed_norm_std': text_embed_norms.std(unbiased=False).detach(),
            'surrogate_text_embed_norm_min': text_embed_norms.min().detach(),
            'surrogate_text_embed_norm_max': text_embed_norms.max().detach(),
            'exact_text_embed_norm': text_embed_norms.mean().detach(),
            'exact_text_embed_norm_raw': text_embed_norms.mean().detach(),
            'exact_text_embed_unit_norm': text_embed_unit_norms.mean().detach(),
            'exact_text_embed_norm_std': text_embed_norms.std(unbiased=False).detach(),
            'exact_text_embed_norm_min': text_embed_norms.min().detach(),
            'exact_text_embed_norm_max': text_embed_norms.max().detach(),
            'token_valid_fraction': exact_outputs['token_valid_mask'].float().mean().detach(),
            'valid_token_fraction': exact_outputs['token_keep_mask'].float().mean().detach(),
            'token_special_mass': self._compute_special_mass(exact_outputs['token_weights'], exact_outputs['special_token_positions']).detach(),
            'token_pool_entropy': exact_outputs['pooler_debug']['token_pool_entropy'],
            'beta_max_prob': exact_outputs['pooler_debug']['beta_max_prob'],
        }
        metrics.update(self._pooling_mode_metrics(image_outputs['image_embedding']))
        return metrics

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
        del epoch, current_step
        image_outputs = self.encode_image_branch(
            image_embeddings,
            image_local_tokens=image_local_tokens,
            return_debug=return_debug,
        )
        prepared_text = self._prepare_text_inputs(
            text_token_states,
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
        )
        exact_outputs = self.pool_text_with_summary(
            image_outputs['summary'],
            text_token_states,
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
            return_debug=return_debug,
            prepared_text=prepared_text,
        )
        surrogate_pairwise_logits = None
        loss_outputs = self.losses(
            image_outputs['image_projected'],
            exact_outputs['text_projected'],
            exact_outputs['text_projected'],
            pids=pids,
            prototypes=None,
            routing_weights=None,
            surrogate_pairwise_logits=surrogate_pairwise_logits,
            host_pairwise_logits=host_pairwise_logits,
            semantic_image_student_embeddings=image_outputs['image_projected'],
            semantic_text_student_embeddings=exact_outputs['text_projected'],
            semantic_text_teacher_embeddings=exact_outputs['text_projected'],
            semantic_base_prototypes=None,
            diag_loss_scale=None,
            semantic_pbt_loss_scale=None,
            return_debug=return_debug,
            disable_proxy_losses=disable_proxy_losses,
        )
        empty_context = self._empty_context(image_outputs['image_embedding'])
        basis_bank = image_outputs['image_embedding'].new_empty((image_outputs['image_embedding'].size(0), 0, self.prototype_dim))
        scalar_metrics = self._collect_scalar_metrics(image_outputs, exact_outputs)
        scalar_metrics.update(loss_outputs.get('debug_metrics', {}))

        outputs = {
            'image_embedding': image_outputs['image_embedding'],
            'text_token_states': prepared_text['text_token_states'],
            'prototypes': empty_context['prototypes'],
            'contextualized_prototypes': empty_context['contextualized_prototypes'],
            'routing_weights': image_outputs['routing_weights'],
            'summary': image_outputs['summary'],
            'token_valid_mask': exact_outputs['token_valid_mask'],
            'token_keep_mask': exact_outputs['token_keep_mask'],
            'valid_mask': exact_outputs['valid_mask'],
            'basis_bank': basis_bank,
            'exact_token_scores': exact_outputs['token_scores'],
            'exact_token_weights': exact_outputs['token_weights'],
            'beta_logits_masked': exact_outputs['beta_logits_masked'],
            'exact_pooled_text': exact_outputs['pooled_text'],
            'surrogate_pooled_text': exact_outputs['pooled_text'],
            'image_projected': image_outputs['image_projected'],
            'image_projected_raw': image_outputs['image_projected_raw'],
            'image_proxy_features': image_outputs['image_proxy_features'],
            'surrogate_text_projected': exact_outputs['text_projected'],
            'surrogate_text_projected_raw': exact_outputs['text_projected_raw'],
            'exact_text_projected': exact_outputs['text_projected'],
            'exact_text_projected_raw': exact_outputs['text_projected_raw'],
            'alpha': image_outputs['routing_weights'],
            'beta': exact_outputs['token_weights'],
            'Q': image_outputs['summary'],
            'Theta_v': empty_context['prototypes'],
            'Theta_tilde': empty_context['contextualized_prototypes'],
            'T_pool': exact_outputs['pooled_text'],
            'T_exact_pool': exact_outputs['pooled_text'],
            'T_hat_pool': exact_outputs['pooled_text'],
            'Z_v': image_outputs['image_projected'],
            'Z_v_raw': image_outputs['image_projected_raw'],
            'Z_t': exact_outputs['text_projected'],
            'Z_t_raw': exact_outputs['text_projected_raw'],
            'Z_t_exact': exact_outputs['text_projected'],
            'Z_t_exact_raw': exact_outputs['text_projected_raw'],
            'surrogate_pairwise_logits': surrogate_pairwise_logits,
            'losses': loss_outputs,
            'metrics': scalar_metrics,
        }
        outputs['debug'] = dict(scalar_metrics)
        if return_debug:
            outputs['debug'].update(
                {
                    **image_outputs.get('debug', {}),
                    **exact_outputs.get('debug', {}),
                    'alpha': image_outputs['routing_weights'].detach(),
                    'beta': exact_outputs['token_weights'].detach(),
                    'Q': image_outputs['summary'].detach(),
                    'Theta_v': empty_context['prototypes'].detach(),
                    'Theta_tilde': empty_context['contextualized_prototypes'].detach(),
                    'T_pool': exact_outputs['pooled_text'].detach(),
                    'T_exact_pool': exact_outputs['pooled_text'].detach(),
                    'T_hat_pool': exact_outputs['pooled_text'].detach(),
                    'Z_v': image_outputs['image_projected'].detach(),
                    'Z_v_raw': image_outputs['image_projected_raw'].detach(),
                    'Z_t': exact_outputs['text_projected'].detach(),
                    'Z_t_raw': exact_outputs['text_projected_raw'].detach(),
                    'Z_t_exact': exact_outputs['text_projected'].detach(),
                    'Z_t_exact_raw': exact_outputs['text_projected_raw'].detach(),
                    'token_valid_mask': exact_outputs['token_valid_mask'].detach(),
                    'token_keep_mask': exact_outputs['token_keep_mask'].detach(),
                    'beta_logits_masked': exact_outputs['beta_logits_masked'].detach(),
                    'basis_bank': basis_bank.detach(),
                }
            )
            if surrogate_pairwise_logits is not None:
                outputs['debug']['surrogate_pairwise_logits'] = surrogate_pairwise_logits.detach()
            if 'surrogate_retrieval_logits' in loss_outputs:
                outputs['debug']['surrogate_retrieval_logits'] = loss_outputs['surrogate_retrieval_logits'].detach()
            if 'image_proxy_logits' in loss_outputs:
                outputs['debug']['image_proxy_logits'] = loss_outputs['image_proxy_logits'].detach()
                outputs['debug']['text_proxy_logits'] = loss_outputs['text_proxy_logits'].detach()
                outputs['debug']['text_exact_proxy_logits'] = loss_outputs['text_exact_proxy_logits'].detach()
                outputs['debug']['class_proxies'] = loss_outputs['class_proxies']
        return outputs
