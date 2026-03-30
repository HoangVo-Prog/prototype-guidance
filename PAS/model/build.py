from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn

from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .interfaces import EncoderOutput
from .prototype import build_prototype_head


class PASModel(nn.Module):
    def __init__(self, args, num_classes=0):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = int(base_cfg['embed_dim'])

        self.model_name = getattr(args, 'model_name', 'PAS')
        self.model_variant = getattr(args, 'model_variant', 'pas_v1')
        self.image_backbone = getattr(args, 'image_backbone', args.pretrain_choice)
        self.text_backbone = getattr(args, 'text_backbone', 'clip_text_transformer')
        self.projection_dim = getattr(args, 'projection_dim', self.embed_dim)
        self.prototype_dim = getattr(args, 'prototype_dim', self.embed_dim)
        self.return_debug_outputs = bool(getattr(args, 'return_debug_outputs', False))
        self.prototype_eval_image_chunk_size = int(getattr(args, 'prototype_eval_image_chunk_size', 32) or 32)
        self.prototype_eval_text_chunk_size = int(getattr(args, 'prototype_eval_text_chunk_size', 128) or 128)

        self._validate_configuration()
        self.prototype_head = build_prototype_head(args, input_dim=self.embed_dim)
        self._apply_freeze_policy()

    def _validate_configuration(self):
        if not bool(getattr(self.args, 'use_prototype_bank', True)):
            raise ValueError('Phase E requires model.use_prototype_bank=true.')
        if not bool(getattr(self.args, 'use_image_conditioned_pooling', True)):
            raise ValueError('Phase E requires model.use_image_conditioned_pooling=true.')
        if int(getattr(self.args, 'prototype_contextualization_num_layers', 1) or 1) != 1:
            raise NotImplementedError('Phase E supports exactly one parameter-free contextualization layer.')
        if bool(getattr(self.args, 'prototype_sparse_assignment', False)) or int(getattr(self.args, 'prototype_sparse_topk', 0) or 0) > 0:
            raise NotImplementedError('Sparse prototype routing is still disabled in Phase E.')
        if self.prototype_eval_image_chunk_size <= 0 or self.prototype_eval_text_chunk_size <= 0:
            raise ValueError('Prototype evaluation chunk sizes must be positive integers.')
        if self.prototype_dim != self.embed_dim:
            raise ValueError(
                'Minimal PAS v1 requires prototype_dim to match the backbone feature dimension so the method uses '
                'token-level text hidden states without an extra hidden-space redesign.'
            )
        token_policy = str(getattr(self.args, 'token_policy', 'content_only')).lower()
        exclude_special_tokens = bool(getattr(self.args, 'exclude_special_tokens', True))
        eos_as_only_token = bool(getattr(self.args, 'eos_as_only_token', False))
        mask_padding_tokens = bool(getattr(self.args, 'mask_padding_tokens', True))
        if not mask_padding_tokens:
            raise ValueError('Minimal PAS v1 requires padding tokens to be masked out.')
        if token_policy == 'content_only' and not exclude_special_tokens:
            raise ValueError('token_policy=content_only conflicts with exclude_special_tokens=false.')
        if token_policy == 'content_plus_special' and exclude_special_tokens:
            raise ValueError('token_policy=content_plus_special conflicts with exclude_special_tokens=true.')
        if token_policy != 'eos_only' and eos_as_only_token:
            raise ValueError('eos_as_only_token=true conflicts with the active token_policy.')

    def _freeze_module(self, module: nn.Module):
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _apply_freeze_policy(self):
        self.freeze_image_backbone = bool(getattr(self.args, 'freeze_image_backbone', True))
        self.freeze_text_backbone = bool(getattr(self.args, 'freeze_text_backbone', True))

        if self.freeze_image_backbone:
            self._freeze_module(self.base_model.visual)
        if self.freeze_text_backbone:
            self._freeze_module(self.base_model.transformer)
            self._freeze_module(self.base_model.token_embedding)
            self.base_model.positional_embedding.requires_grad = False
            self.base_model.ln_final.weight.requires_grad = False
            self.base_model.ln_final.bias.requires_grad = False
            self.base_model.text_projection.requires_grad = False

    def _get_special_token_positions(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = token_ids.size(0)
        device = token_ids.device
        return {
            'cls': torch.zeros(batch_size, dtype=torch.long, device=device),
            'eos': token_ids.argmax(dim=-1),
        }

    def _build_text_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.ne(0)

    def _resolve_text_states(self, text_output: EncoderOutput) -> torch.Tensor:
        if text_output.pre_projection_tokens is None:
            raise ValueError('The text encoder must expose last-layer token hidden states before CLIP pooling/projection.')
        if text_output.pre_projection_tokens.size(-1) != self.prototype_dim:
            raise ValueError(
                'Minimal PAS v1 requires text hidden-state dimension to match prototype_dim so routing scores are '
                'computed over token-level hidden states without a hidden-space detour.'
            )
        return text_output.pre_projection_tokens

    def extract_image_features(self, image: torch.Tensor) -> EncoderOutput:
        image_outputs = self.base_model.encode_image_intermediates(image, return_all=False, average_attn_weights=True)
        projected_tokens = image_outputs['projected_tokens'].float()
        pre_projection_tokens = image_outputs['pre_projection_tokens']
        if pre_projection_tokens is not None:
            pre_projection_tokens = pre_projection_tokens.float()
        image_global = projected_tokens[:, 0, :]
        image_pre_projection = None if pre_projection_tokens is None else pre_projection_tokens[:, 0, :]
        return EncoderOutput(
            tokens=projected_tokens,
            pooled=image_global,
            projected_tokens=projected_tokens,
            projected_pooled=image_global,
            pre_projection_tokens=pre_projection_tokens,
            pre_projection_pooled=image_pre_projection,
            attention_weights=None,
            token_mask=None,
            special_token_positions={},
            pooling_mode='cls',
            metadata={
                'encoder': 'image',
                'backbone': self.image_backbone,
            },
        )

    def extract_text_features(self, text: torch.Tensor) -> EncoderOutput:
        text_outputs = self.base_model.encode_text_intermediates(text.long(), return_all=False, average_attn_weights=True)
        projected_tokens = text_outputs['projected_tokens'].float()
        pre_projection_tokens = text_outputs['pre_projection_tokens']
        if pre_projection_tokens is not None:
            pre_projection_tokens = pre_projection_tokens.float()
        special_positions = self._get_special_token_positions(text)
        token_mask = self._build_text_mask(text)
        batch_indices = torch.arange(text.size(0), device=text.device)
        projected_pooled = projected_tokens[batch_indices, special_positions['eos']]
        pre_projection_pooled = None
        if pre_projection_tokens is not None:
            pre_projection_pooled = pre_projection_tokens[batch_indices, special_positions['eos']]
        return EncoderOutput(
            tokens=projected_tokens,
            pooled=projected_pooled,
            projected_tokens=projected_tokens,
            projected_pooled=projected_pooled,
            pre_projection_tokens=pre_projection_tokens,
            pre_projection_pooled=pre_projection_pooled,
            attention_weights=None,
            token_mask=token_mask,
            special_token_positions=special_positions,
            pooling_mode='image_conditioned',
            metadata={
                'encoder': 'text',
                'backbone': self.text_backbone,
            },
        )

    def encode_image_for_retrieval(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_output = self.extract_image_features(image)
        prototype_outputs = self.prototype_head.encode_image_branch(image_output.projected_pooled, return_debug=False)
        return {
            'image_projected': prototype_outputs['image_projected'].float(),
            'summary': prototype_outputs['summary'].float(),
        }

    def encode_text_for_retrieval(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_output = self.extract_text_features(text)
        return {
            'text_token_states': self._resolve_text_states(text_output).float(),
            'token_ids': text.long(),
            'attention_mask': text_output.token_mask,
            'special_token_positions': {key: value for key, value in text_output.special_token_positions.items()},
        }

    def compute_retrieval_similarity(self, image_features: Dict[str, torch.Tensor], text_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        similarity = self.prototype_head.compute_pairwise_similarity(
            image_projected=image_features['image_projected'],
            summaries=image_features['summary'],
            text_token_states=text_features['text_token_states'],
            token_ids=text_features['token_ids'],
            attention_mask=text_features.get('attention_mask'),
            special_token_positions=text_features.get('special_token_positions'),
            image_chunk_size=self.prototype_eval_image_chunk_size,
            text_chunk_size=self.prototype_eval_text_chunk_size,
        )
        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Retrieval similarity contains NaN or Inf values.')
        return similarity

    def _build_debug_outputs(
        self,
        image_output: EncoderOutput,
        text_output: EncoderOutput,
        prototype_outputs: Dict[str, object],
    ) -> Dict[str, object]:
        debug = dict(prototype_outputs.get('metrics', {}))
        debug.update(
            {
                'image_global': image_output.projected_pooled.detach(),
                'text_tokens': self._resolve_text_states(text_output).detach(),
                'token_mask': text_output.token_mask.detach(),
                'special_token_positions': {key: value.detach() for key, value in text_output.special_token_positions.items()},
                'alpha': prototype_outputs['routing_weights'].detach(),
                'beta': prototype_outputs['token_weights'].detach(),
                'Q': prototype_outputs['summary'].detach(),
                'Theta_v': prototype_outputs['prototypes'].detach(),
                'Theta_tilde': prototype_outputs['contextualized_prototypes'].detach(),
                'token_valid_mask': prototype_outputs['token_valid_mask'].detach(),
                'token_keep_mask': prototype_outputs['token_keep_mask'].detach(),
                'beta_logits_masked': prototype_outputs['beta_logits_masked'].detach(),
                'T_pool': prototype_outputs['pooled_text'].detach(),
                'Z_v': prototype_outputs['image_projected'].detach(),
                'Z_v_raw': prototype_outputs['image_projected_raw'].detach(),
                'Z_t': prototype_outputs['text_projected'].detach(),
                'Z_t_raw': prototype_outputs['text_projected_raw'].detach(),
            }
        )
        return debug

    def named_optimizer_groups(self) -> OrderedDict:
        groups = OrderedDict(
            prototype_bank=[],
            contextualizer=[],
            projectors=[],
            logit_scale=[],
            image_backbone=[],
            text_backbone=[],
            other=[],
        )
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('prototype_head.prototype_bank'):
                groups['prototype_bank'].append((name, parameter))
            elif name.startswith('prototype_head.contextualizer'):
                groups['contextualizer'].append((name, parameter))
            elif name.startswith('prototype_head.image_projector') or name.startswith('prototype_head.text_projector') or name.startswith('prototype_head.image_adapter') or name.startswith('prototype_head.text_adapter'):
                groups['projectors'].append((name, parameter))
            elif name.endswith('logit_scale'):
                groups['logit_scale'].append((name, parameter))
            elif name.startswith('base_model.visual'):
                groups['image_backbone'].append((name, parameter))
            elif name.startswith('base_model.transformer') or name.startswith('base_model.token_embedding') or name.startswith('base_model.positional_embedding') or name.startswith('base_model.ln_final') or name.startswith('base_model.text_projection'):
                groups['text_backbone'].append((name, parameter))
            else:
                groups['other'].append((name, parameter))
        return groups

    def forward(self, batch, epoch=None, current_step=None, return_debug: Optional[bool] = None):
        del epoch, current_step
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_output = self.extract_image_features(images)
        text_output = self.extract_text_features(caption_ids)
        should_return_debug = self.return_debug_outputs if return_debug is None else bool(return_debug)
        prototype_outputs = self.prototype_head(
            image_embeddings=image_output.projected_pooled,
            text_token_states=self._resolve_text_states(text_output),
            token_ids=caption_ids,
            attention_mask=text_output.token_mask,
            special_token_positions=text_output.special_token_positions,
            return_debug=should_return_debug,
        )

        losses = prototype_outputs['losses']
        if not torch.isfinite(losses['loss_total']):
            raise FloatingPointError('loss_total contains NaN or Inf values.')

        logit_scale = losses['logit_scale']
        outputs = {
            'loss_total': losses['loss_total'],
            'loss_infonce': losses['loss_infonce'],
            'loss_diversity': losses['loss_diversity'],
            'loss_balance': losses['loss_balance'],
            'loss_diversity_weighted': losses['loss_diversity_weighted'],
            'loss_balance_weighted': losses['loss_balance_weighted'],
            'lambda_div': losses['lambda_div'],
            'lambda_bal': losses['lambda_bal'],
            'temperature': torch.reciprocal(logit_scale.detach()),
            'logit_scale': logit_scale.detach(),
            'debug': dict(prototype_outputs.get('metrics', {})),
        }
        if 'contrastive_logits' in losses:
            outputs['logits'] = losses['contrastive_logits'].detach()
        if should_return_debug:
            outputs['debug'] = self._build_debug_outputs(image_output, text_output, prototype_outputs)
        return outputs


PrototypeGuidedRetrievalModel = PASModel
Model = PASModel


def build_model(args, num_classes=0):
    model = PASModel(args, num_classes=num_classes)
    convert_weights(model.base_model)
    model.prototype_head.float()
    return model
