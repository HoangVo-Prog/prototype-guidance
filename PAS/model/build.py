from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn

from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .interfaces import EncoderOutput
from .prototype import TokenMaskBuilder, build_prototype_head
from utils.precision import (
    canonicalize_amp_dtype,
    canonicalize_backbone_precision,
    canonicalize_prototype_precision,
    precision_to_torch_dtype,
)


SUPPORTED_PAS_CLIP_BACKBONES = (
    'ViT-B/16',
    'ViT-B/32',
    'ViT-L/14',
)


class PASModel(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._validate_pretrain_choice()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self._validate_backbone_contract(base_cfg)
        self.embed_dim = int(base_cfg['embed_dim'])

        self.model_name = getattr(args, 'model_name', 'PAS')
        self.model_variant = getattr(args, 'model_variant', 'pas_v1')
        self.image_backbone = getattr(args, 'image_backbone', args.pretrain_choice)
        self.text_backbone = getattr(args, 'text_backbone', 'clip_text_transformer')
        self.projection_dim = getattr(args, 'projection_dim', self.embed_dim)
        self.prototype_dim = getattr(args, 'prototype_dim', self.embed_dim)
        self.backbone_precision = canonicalize_backbone_precision(getattr(args, 'backbone_precision', 'fp16'))
        self.prototype_precision = canonicalize_prototype_precision(getattr(args, 'prototype_precision', 'fp32'))
        self.return_debug_outputs = bool(getattr(args, 'return_debug_outputs', False))
        self.prototype_eval_image_chunk_size = int(getattr(args, 'prototype_eval_image_chunk_size', 32) or 32)
        self.prototype_eval_text_chunk_size = int(getattr(args, 'prototype_eval_text_chunk_size', 128) or 128)

        self._validate_configuration()
        self.prototype_head = build_prototype_head(args, input_dim=self.embed_dim, num_classes=self.num_classes)
        self._apply_freeze_policy()

    def _validate_pretrain_choice(self):
        pretrain_choice = getattr(self.args, 'pretrain_choice', None)
        if pretrain_choice not in SUPPORTED_PAS_CLIP_BACKBONES:
            raise ValueError(
                'PAS currently supports only ViT CLIP backbones with the token-level runtime contract required by '
                f'prototype routing. Supported `pretrain_choice` values: {list(SUPPORTED_PAS_CLIP_BACKBONES)}. '
                f'Got {pretrain_choice!r}.'
            )

    def _validate_backbone_contract(self, base_cfg):
        vision_layers = base_cfg.get('vision_layers')
        if isinstance(vision_layers, (tuple, list)):
            raise ValueError(
                'PAS requires a ViT visual backbone that returns token sequences with a CLS slot; '
                f'got vision_layers={vision_layers!r} from pretrain_choice={getattr(self.args, "pretrain_choice", None)!r}.'
            )
        transformer_width = int(base_cfg.get('transformer_width', base_cfg['embed_dim']))
        embed_dim = int(base_cfg['embed_dim'])
        if transformer_width != embed_dim:
            raise ValueError(
                'PAS consumes text pre-projection token states, so it requires CLIP variants where '
                f'transformer_width == embed_dim. Got transformer_width={transformer_width} and embed_dim={embed_dim} '
                f'for pretrain_choice={getattr(self.args, "pretrain_choice", None)!r}.'
            )

    def _validate_configuration(self):
        if not bool(getattr(self.args, 'use_prototype_bank', True)):
            raise ValueError('PASModel requires model.use_prototype_bank=true because the active runtime is prototype-based.')
        if not bool(getattr(self.args, 'use_image_conditioned_pooling', True)):
            raise ValueError('PASModel requires model.use_image_conditioned_pooling=true because the active runtime scores text under image-conditioned pooling.')
        if not bool(getattr(self.args, 'normalize_projector_outputs', True)):
            raise ValueError(
                'PASModel requires model.normalize_projector_outputs=true in the active runtime so '
                'proxy supervision, fidelity/alignment losses, and retrieval scoring all operate on the same '
                'cosine-normalized embedding family.'
            )
        if self.num_classes <= 0:
            raise ValueError('PASModel requires num_classes > 0 so the active runtime can instantiate class proxies consistently for train and eval.')
        if self.prototype_eval_image_chunk_size <= 0 or self.prototype_eval_text_chunk_size <= 0:
            raise ValueError('Prototype evaluation chunk sizes must be positive integers.')
        configured_embedding_dim = getattr(self.args, 'embedding_dim', None)
        if configured_embedding_dim is not None and int(configured_embedding_dim) != self.embed_dim:
            raise ValueError(
                'model.embedding_dim must match the CLIP backbone feature dimension in PAS; '
                f'got embedding_dim={int(configured_embedding_dim)} and backbone feature_dim={self.embed_dim}.'
            )
        special_token_ids = getattr(self.args, 'special_token_ids', None)
        if special_token_ids is None:
            raise ValueError(
                'special_token_ids must be configured explicitly so token masking does not rely on hardcoded '
                'tokenizer assumptions.'
            )
        if self.backbone_precision == 'fp16' and bool(getattr(self.args, 'amp', False)):
            if canonicalize_amp_dtype(getattr(self.args, 'amp_dtype', 'fp16')) != 'fp16':
                raise ValueError('model.backbone_precision=fp16 requires training.amp_dtype=fp16 when AMP is enabled.')
        if self.prototype_precision == 'fp16' and bool(getattr(self.args, 'amp', False)):
            if canonicalize_amp_dtype(getattr(self.args, 'amp_dtype', 'fp16')) != 'fp16':
                raise ValueError('model.prototype_precision=fp16 requires training.amp_dtype=fp16 when AMP is enabled.')
        if bool(getattr(self.args, 'training', True)) and self.backbone_precision == 'fp16':
            if (not bool(getattr(self.args, 'freeze_image_backbone', True)) or not bool(getattr(self.args, 'freeze_text_backbone', True))) and not bool(getattr(self.args, 'amp', False)):
                raise ValueError('Unfrozen fp16 backbone training requires training.amp=true so the backbone is updated under proper AMP scaling.')
        if bool(getattr(self.args, 'training', True)) and self.prototype_precision == 'fp16' and not bool(getattr(self.args, 'amp', False)):
            raise ValueError('model.prototype_precision=fp16 requires training.amp=true so prototype modules are updated under proper AMP scaling.')
        TokenMaskBuilder(
            token_policy=str(getattr(self.args, 'token_policy', 'content_only')).lower(),
            special_token_ids=special_token_ids,
            error_on_empty_kept_tokens=bool(getattr(self.args, 'error_on_empty_kept_tokens', True)),
        )

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

    def _resolve_text_states(self, text_output: EncoderOutput) -> torch.Tensor:
        if text_output.pre_projection_tokens is None:
            raise ValueError('The text encoder must expose last-layer token hidden states before CLIP pooling/projection.')
        return text_output.pre_projection_tokens

    def _prototype_dtype(self) -> torch.dtype:
        return precision_to_torch_dtype(self.prototype_precision)

    def _cast_to_prototype_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=self._prototype_dtype())

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
                'backbone_precision': self.backbone_precision,
                'prototype_precision': self.prototype_precision,
            },
        )

    def extract_text_features(self, text: torch.Tensor) -> EncoderOutput:
        text_outputs = self.base_model.encode_text_intermediates(text.long(), return_all=False, average_attn_weights=True)
        projected_tokens = text_outputs['projected_tokens'].float()
        pre_projection_tokens = text_outputs['pre_projection_tokens']
        if pre_projection_tokens is not None:
            pre_projection_tokens = pre_projection_tokens.float()
        token_mask_builder = self.prototype_head.token_mask_builder
        token_mask = token_mask_builder.build_valid_mask(text.long())
        special_positions = token_mask_builder.get_special_token_positions(text.long(), attention_mask=token_mask)
        batch_indices = torch.arange(text.size(0), device=text.device)
        projected_pooled = None
        pre_projection_pooled = None
        eos_positions = special_positions.get('eos')
        if eos_positions is not None:
            projected_pooled = projected_tokens[batch_indices, eos_positions]
            if pre_projection_tokens is not None:
                pre_projection_pooled = pre_projection_tokens[batch_indices, eos_positions]
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
                'backbone_precision': self.backbone_precision,
                'prototype_precision': self.prototype_precision,
            },
        )

    def encode_image_for_retrieval(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_output = self.extract_image_features(image)
        prototype_outputs = self.prototype_head.encode_image_branch(self._cast_to_prototype_dtype(image_output.projected_pooled), return_debug=False)
        return {
            'image_projected': prototype_outputs['image_projected'],
            'summary': prototype_outputs['summary'],
            'routing_weights': prototype_outputs['routing_weights'],
        }

    def encode_text_for_retrieval(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_output = self.extract_text_features(text)
        return {
            'text_token_states': self._cast_to_prototype_dtype(self._resolve_text_states(text_output)),
            'token_ids': text.long(),
            'attention_mask': text_output.token_mask,
            'special_token_positions': {key: value for key, value in text_output.special_token_positions.items()},
        }

    def encode_text_basis_for_retrieval(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_output = self.extract_text_features(text)
        context = self.prototype_head.get_prototype_context(return_debug=False)
        basis_outputs = self.prototype_head.build_text_basis_bank(
            text_token_states=self._cast_to_prototype_dtype(self._resolve_text_states(text_output)),
            token_ids=text.long(),
            contextualized_prototypes=self._cast_to_prototype_dtype(context['contextualized_prototypes']),
            attention_mask=text_output.token_mask,
            special_token_positions=text_output.special_token_positions,
            return_debug=False,
        )
        return {
            'basis_bank': basis_outputs['basis_bank'],
        }

    def compute_retrieval_similarity(self, image_features: Dict[str, torch.Tensor], text_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        similarity = self.prototype_head.compute_pairwise_similarity(
            image_projected=self._cast_to_prototype_dtype(image_features['image_projected']),
            summaries=self._cast_to_prototype_dtype(image_features['summary']),
            text_token_states=self._cast_to_prototype_dtype(text_features['text_token_states']),
            token_ids=text_features['token_ids'],
            attention_mask=text_features.get('attention_mask'),
            special_token_positions=text_features.get('special_token_positions'),
            image_chunk_size=self.prototype_eval_image_chunk_size,
            text_chunk_size=self.prototype_eval_text_chunk_size,
        )
        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Retrieval similarity contains NaN or Inf values.')
        return similarity.float()

    def compute_approximate_retrieval_similarity(self, image_features: Dict[str, torch.Tensor], text_basis_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        similarity = self.prototype_head.compute_approximate_pairwise_similarity(
            image_projected=self._cast_to_prototype_dtype(image_features['image_projected']),
            routing_weights=self._cast_to_prototype_dtype(image_features['routing_weights']),
            basis_bank=self._cast_to_prototype_dtype(text_basis_features['basis_bank']),
            image_chunk_size=self.prototype_eval_image_chunk_size,
            text_chunk_size=self.prototype_eval_text_chunk_size,
        )
        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Approximate retrieval similarity contains NaN or Inf values.')
        return similarity.float()

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
                'beta': prototype_outputs['exact_token_weights'].detach(),
                'Q': prototype_outputs['summary'].detach(),
                'Theta_v': prototype_outputs['prototypes'].detach(),
                'Theta_tilde': prototype_outputs['contextualized_prototypes'].detach(),
                'token_valid_mask': prototype_outputs['token_valid_mask'].detach(),
                'token_keep_mask': prototype_outputs['token_keep_mask'].detach(),
                'beta_logits_masked': prototype_outputs['beta_logits_masked'].detach(),
                'basis_bank': prototype_outputs['basis_bank'].detach(),
                'T_pool': prototype_outputs['surrogate_pooled_text'].detach(),
                'T_exact_pool': prototype_outputs['exact_pooled_text'].detach(),
                'T_hat_pool': prototype_outputs['surrogate_pooled_text'].detach(),
                'Z_v': prototype_outputs['image_projected'].detach(),
                'Z_v_raw': prototype_outputs['image_projected_raw'].detach(),
                'Z_t': prototype_outputs['surrogate_text_projected'].detach(),
                'Z_t_raw': prototype_outputs['surrogate_text_projected_raw'].detach(),
                'Z_t_exact': prototype_outputs['exact_text_projected'].detach(),
                'Z_t_exact_raw': prototype_outputs['exact_text_projected_raw'].detach(),
            }
        )
        for key in ('basis_token_scores', 'basis_token_weights', 'basis_beta_logits_masked', 'image_proxy_logits', 'text_proxy_logits', 'text_exact_proxy_logits', 'class_proxies'):
            value = prototype_outputs.get('debug', {}).get(key)
            if isinstance(value, torch.Tensor):
                debug[key] = value.detach()
            elif value is not None:
                debug[key] = value
        return debug

    def named_optimizer_groups(self) -> OrderedDict:
        groups = OrderedDict(
            prototype_bank=[],
            projectors=[],
            class_proxies=[],
            image_backbone=[],
            text_backbone=[],
            other=[],
        )
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('prototype_head.prototype_bank'):
                groups['prototype_bank'].append((name, parameter))
            elif name.startswith('prototype_head.image_projector') or name.startswith('prototype_head.text_projector') or name.startswith('prototype_head.image_adapter') or name.startswith('prototype_head.text_adapter'):
                groups['projectors'].append((name, parameter))
            elif name.startswith('prototype_head.losses.class_proxies'):
                groups['class_proxies'].append((name, parameter))
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
        if 'pids' not in batch:
            raise KeyError("PASModel.forward requires batch['pids'] as class labels for the amortized proxy objective.")
        pids = batch['pids']
        image_output = self.extract_image_features(images)
        text_output = self.extract_text_features(caption_ids)
        should_return_debug = self.return_debug_outputs if return_debug is None else bool(return_debug)
        prototype_outputs = self.prototype_head(
            image_embeddings=self._cast_to_prototype_dtype(image_output.projected_pooled),
            text_token_states=self._cast_to_prototype_dtype(self._resolve_text_states(text_output)),
            token_ids=caption_ids,
            pids=pids,
            attention_mask=text_output.token_mask,
            special_token_positions=text_output.special_token_positions,
            return_debug=should_return_debug,
        )

        losses = prototype_outputs['losses']
        if not torch.isfinite(losses['loss_total']):
            raise FloatingPointError('loss_total contains NaN or Inf values.')

        outputs = {
            'loss_total': losses['loss_total'],
            'loss_proxy': losses['loss_proxy'],
            'loss_proxy_image': losses['loss_proxy_image'],
            'loss_proxy_text': losses['loss_proxy_text'],
            'loss_proxy_text_exact': losses['loss_proxy_text_exact'],
            'loss_align': losses['loss_align'],
            'loss_diag': losses['loss_diag'],
            'loss_diversity': losses['loss_diversity'],
            'loss_balance': losses['loss_balance'],
            'loss_proxy_weighted': losses['loss_proxy_weighted'],
            'loss_align_weighted': losses['loss_align_weighted'],
            'loss_diag_weighted': losses['loss_diag_weighted'],
            'loss_diversity_weighted': losses['loss_diversity_weighted'],
            'loss_balance_weighted': losses['loss_balance_weighted'],
            'lambda_proxy': losses['lambda_proxy'],
            'use_loss_proxy_text_exact': losses['use_loss_proxy_text_exact'],
            'lambda_align': losses['lambda_align'],
            'lambda_diag': losses['lambda_diag'],
            'lambda_div': losses['lambda_div'],
            'lambda_bal': losses['lambda_bal'],
            'proxy_temperature': losses['proxy_temperature'].detach(),
            'retrieval_temperature': losses['retrieval_temperature'].detach(),
            'logit_scale': losses['logit_scale'].detach(),
            'alpha': prototype_outputs['routing_weights'].detach(),
            'z_v': prototype_outputs['image_projected'],
            'z_t_hat_diag': prototype_outputs['surrogate_text_projected'],
            'z_t_exact_diag': prototype_outputs['exact_text_projected'],
            'debug': dict(prototype_outputs.get('metrics', {})),
        }
        for grad_tensor_key in ('z_v', 'z_t_hat_diag', 'z_t_exact_diag'):
            tensor = outputs.get(grad_tensor_key)
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.retain_grad()
        if should_return_debug:
            outputs['debug'] = self._build_debug_outputs(image_output, text_output, prototype_outputs)
        return outputs


PrototypeGuidedRetrievalModel = PASModel
Model = PASModel


def build_model(args, num_classes):
    model = PASModel(args, num_classes=num_classes)
    if model.backbone_precision == 'fp16':
        convert_weights(model.base_model)
    else:
        model.base_model.float()
    if model.prototype_precision == 'fp16':
        model.prototype_head.half()
    else:
        model.prototype_head.float()
    return model
