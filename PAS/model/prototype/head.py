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
        prototype_init: str = 'normalized_random',
        prototype_init_path: Optional[str] = None,
        routing_type: str = 'cosine',
        routing_temperature: float = 0.07,
        token_scoring_type: str = 'cosine',
        token_temperature: float = 0.07,
        token_policy: str = 'content_only',
        contextualization_enabled: bool = False,
        contextualization_type: str = 'none',
        contextualization_residual: bool = True,
        prototype_normalize: bool = True,
        use_diversity_loss: bool = False,
        diversity_loss_weight: float = 0.0,
        use_balance_loss: bool = False,
        balance_loss_weight: float = 0.0,
        contrastive_temperature_init: float = 0.07,
        learnable_contrastive_temperature: bool = True,
        dead_prototype_threshold: float = 0.005,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.prototype_dim = int(prototype_dim)
        self.projector_hidden_dim = int(projector_hidden_dim or prototype_dim)
        self.projector_output_dim = int(projector_output_dim)
        self.dead_prototype_threshold = float(dead_prototype_threshold)

        self.image_adapter = nn.Identity() if self.input_dim == self.prototype_dim else nn.Linear(self.input_dim, self.prototype_dim)
        self.text_adapter = nn.Identity() if self.input_dim == self.prototype_dim else nn.Linear(self.input_dim, self.prototype_dim)
        self.prototype_bank = PrototypeBank(
            num_prototypes=num_prototypes,
            prototype_dim=prototype_dim,
            init_mode=prototype_init,
            init_path=prototype_init_path,
            normalize_init=prototype_normalize,
        )
        self.contextualizer = PrototypeContextualizer(
            enabled=contextualization_enabled,
            contextualization_type=contextualization_type,
            residual=contextualization_residual,
            normalize=prototype_normalize,
        )
        self.router = Router(routing_type=routing_type, temperature=routing_temperature)
        self.aggregator = PrototypeAggregator()
        self.token_scorer = TokenScorer(scoring_type=token_scoring_type, temperature=token_temperature)
        self.token_mask_builder = TokenMaskBuilder(token_policy=token_policy)
        self.token_pooler = MaskedTokenPooler()
        self.image_projector = MLPProjector(
            input_dim=prototype_dim,
            hidden_dim=self.projector_hidden_dim,
            output_dim=self.projector_output_dim,
            dropout=projector_dropout,
            normalize_output=True,
        )
        self.text_projector = MLPProjector(
            input_dim=prototype_dim,
            hidden_dim=self.projector_hidden_dim,
            output_dim=self.projector_output_dim,
            dropout=projector_dropout,
            normalize_output=True,
        )
        self.losses = PrototypeLosses(
            temperature_init=contrastive_temperature_init,
            learnable_temperature=learnable_contrastive_temperature,
            use_diversity_loss=use_diversity_loss,
            diversity_loss_weight=diversity_loss_weight,
            use_balance_loss=use_balance_loss,
            balance_loss_weight=balance_loss_weight,
        )

    def _compute_special_mass(self, token_weights: torch.Tensor, special_token_positions: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_index = torch.arange(token_weights.size(0), device=token_weights.device)
        cls_mass = token_weights[batch_index, special_token_positions['cls']].mean()
        eos_mass = token_weights[batch_index, special_token_positions['eos']].mean()
        return cls_mass + eos_mass

    def _compute_usage_metrics(self, routing_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        usage = routing_weights.mean(dim=0)
        return {
            'prototype_usage': usage.detach(),
            'prototype_usage_entropy': (-(usage * usage.clamp_min(1e-12).log()).sum()).detach(),
            'prototype_usage_max': usage.max().detach(),
            'prototype_dead_count': (usage < self.dead_prototype_threshold).sum().detach(),
            'routing_entropy': (-(routing_weights * routing_weights.clamp_min(1e-12).log()).sum(dim=-1).mean()).detach(),
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
        token_weights: torch.Tensor,
        token_valid_mask: torch.Tensor,
        token_keep_mask: torch.Tensor,
        pooled_text: torch.Tensor,
        image_features: torch.Tensor,
        image_projector_debug: Dict[str, torch.Tensor],
        text_projector_debug: Dict[str, torch.Tensor],
        special_token_positions: Dict[str, torch.Tensor],
        contextualizer_debug: Optional[Dict[str, torch.Tensor]] = None,
        router_debug: Optional[Dict[str, torch.Tensor]] = None,
        pooler_debug: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        metrics = {}
        metrics.update(self._compute_usage_metrics(routing_weights))
        metrics.update(self._compute_pairwise_cosine_metrics('prototype', prototypes))
        metrics.update(self._compute_pairwise_cosine_metrics('contextualized_prototype', contextualized_prototypes))
        metrics.update(
            {
                'q_norm': summary.norm(dim=-1).mean().detach(),
                't_pool_norm': pooled_text.norm(dim=-1).mean().detach(),
                'image_feature_norm': image_features.norm(dim=-1).mean().detach(),
                'image_embed_norm': image_projector_debug['projected_features_raw'].norm(dim=-1).mean().detach(),
                'text_embed_norm': text_projector_debug['projected_features_raw'].norm(dim=-1).mean().detach(),
                'token_valid_fraction': token_valid_mask.float().mean().detach(),
                'valid_token_fraction': token_keep_mask.float().mean().detach(),
                'token_special_mass': self._compute_special_mass(token_weights, special_token_positions).detach(),
            }
        )
        if contextualizer_debug and 'prototype_contextualization_entropy' in contextualizer_debug:
            metrics['prototype_contextualization_entropy'] = contextualizer_debug['prototype_contextualization_entropy']
        if router_debug:
            metrics['routing_max_prob'] = router_debug['routing_max_prob']
            metrics['prototype_assignment_entropy'] = router_debug['prototype_assignment_entropy']
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
        image_projected, image_projector_debug = self.image_projector(image_features, return_debug=True)

        outputs = {
            'image_embedding': image_features,
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

    def pool_text_with_summary(
        self,
        summary: torch.Tensor,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
        return_debug: bool = False,
    ) -> Dict[str, object]:
        text_features = self.text_adapter(text_token_states)
        token_scores, scorer_debug = self.token_scorer(summary, text_features, return_debug=True)
        token_keep_mask, mask_debug = self.token_mask_builder.build(
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
            return_debug=True,
        )
        pooled_text, token_weights, pooler_debug = self.token_pooler(token_scores, text_features, token_keep_mask, return_debug=True)
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

    def compute_pairwise_similarity(
        self,
        image_projected: torch.Tensor,
        summaries: torch.Tensor,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
        image_chunk_size: int = 32,
        text_chunk_size: int = 128,
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
                block_similarity = (projected_text * expanded_image_projected).sum(dim=-1).view(image_batch, text_batch).t()
                similarity[text_start:text_end, image_start:image_end] = block_similarity

        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Prototype retrieval similarity contains NaN or Inf values.')
        return similarity

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
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
        text_outputs = self.pool_text_with_summary(
            image_outputs['summary'],
            text_token_states,
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
            return_debug=return_debug,
        )
        scalar_metrics = self._collect_scalar_metrics(
            prototypes=context['prototypes'],
            contextualized_prototypes=context['contextualized_prototypes'],
            routing_weights=image_outputs['routing_weights'],
            summary=image_outputs['summary'],
            token_weights=text_outputs['token_weights'],
            token_valid_mask=text_outputs['token_valid_mask'],
            token_keep_mask=text_outputs['token_keep_mask'],
            pooled_text=text_outputs['pooled_text'],
            image_features=image_outputs['image_embedding'],
            image_projector_debug=image_outputs['image_projector_debug'],
            text_projector_debug=text_outputs['text_projector_debug'],
            special_token_positions=text_outputs['special_token_positions'],
            contextualizer_debug=context.get('contextualizer_debug'),
            router_debug=image_outputs['router_debug'],
            pooler_debug=text_outputs['pooler_debug'],
        )
        loss_outputs = self.losses(
            image_outputs['image_projected'],
            text_outputs['text_projected'],
            prototypes=context['prototypes'],
            routing_weights=image_outputs['routing_weights'],
            return_debug=return_debug,
        )

        outputs = {
            'image_embedding': image_outputs['image_embedding'],
            'text_token_states': text_outputs['text_token_states'],
            'prototypes': context['prototypes'],
            'contextualized_prototypes': context['contextualized_prototypes'],
            'routing_weights': image_outputs['routing_weights'],
            'summary': image_outputs['summary'],
            'token_scores': text_outputs['token_scores'],
            'token_valid_mask': text_outputs['token_valid_mask'],
            'token_keep_mask': text_outputs['token_keep_mask'],
            'valid_mask': text_outputs['valid_mask'],
            'token_weights': text_outputs['token_weights'],
            'beta_logits_masked': text_outputs['beta_logits_masked'],
            'pooled_text': text_outputs['pooled_text'],
            'image_projected': image_outputs['image_projected'],
            'image_projected_raw': image_outputs['image_projected_raw'],
            'text_projected': text_outputs['text_projected'],
            'text_projected_raw': text_outputs['text_projected_raw'],
            'alpha': image_outputs['routing_weights'],
            'beta': text_outputs['token_weights'],
            'Q': image_outputs['summary'],
            'Theta_v': context['prototypes'],
            'Theta_tilde': context['contextualized_prototypes'],
            'S_t': text_outputs['token_scores'],
            'T_pool': text_outputs['pooled_text'],
            'Z_v': image_outputs['image_projected'],
            'Z_v_raw': image_outputs['image_projected_raw'],
            'Z_t': text_outputs['text_projected'],
            'Z_t_raw': text_outputs['text_projected_raw'],
            'losses': loss_outputs,
            'metrics': scalar_metrics,
        }
        outputs['debug'] = dict(scalar_metrics)
        if return_debug:
            outputs['debug'].update(
                {
                    **context['debug'],
                    **image_outputs['debug'],
                    **text_outputs['debug'],
                    'alpha': image_outputs['routing_weights'].detach(),
                    'beta': text_outputs['token_weights'].detach(),
                    'Q': image_outputs['summary'].detach(),
                    'Theta_v': context['prototypes'].detach(),
                    'Theta_tilde': context['contextualized_prototypes'].detach(),
                    'T_pool': text_outputs['pooled_text'].detach(),
                    'Z_v': image_outputs['image_projected'].detach(),
                    'Z_v_raw': image_outputs['image_projected_raw'].detach(),
                    'Z_t': text_outputs['text_projected'].detach(),
                    'Z_t_raw': text_outputs['text_projected_raw'].detach(),
                    'token_valid_mask': text_outputs['token_valid_mask'].detach(),
                    'token_keep_mask': text_outputs['token_keep_mask'].detach(),
                    'beta_logits_masked': text_outputs['beta_logits_masked'].detach(),
                }
            )
            if 'contrastive_logits' in loss_outputs:
                outputs['debug']['contrastive_logits'] = loss_outputs['contrastive_logits']
        return outputs
