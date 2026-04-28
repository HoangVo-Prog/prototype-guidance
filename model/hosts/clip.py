"""Vanilla CLIP host implementation (bidirectional InfoNCE, no prototype branch)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.freeze_schedule import set_group_requires_grad

from .itself import get_original_itself_components


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class _Projector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        projector_type: str,
        normalize_output: bool,
        enabled: bool,
    ):
        super().__init__()
        self.normalize_output = bool(normalize_output)
        self.enabled = bool(enabled)
        if not self.enabled:
            self.net = nn.Identity()
            self.projector_type = 'identity'
        else:
            kind = str(projector_type).lower()
            if kind in {'linear'}:
                self.net = nn.Linear(input_dim, output_dim)
                self.projector_type = 'linear'
            elif kind in {'mlp', 'mlp2'}:
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_dim, output_dim),
                )
                self.projector_type = 'mlp2'
            else:
                raise ValueError(f'Unsupported projector_type={projector_type!r}.')

    def forward(self, inputs: torch.Tensor):
        raw = self.net(inputs)
        projected = F.normalize(raw, dim=-1) if self.normalize_output else raw
        return projected, raw


class _VanillaClipLoss(nn.Module):
    """Legacy-compatible CLIP loss with fixed logit scale from `model.temperature`."""

    def __init__(self, temperature: float, retrieval_mode: str, use_loss_ret: bool):
        super().__init__()
        if temperature <= 0:
            raise ValueError('model.temperature must be positive.')
        self.use_loss_ret = bool(use_loss_ret)
        self.retrieval_mode = str(retrieval_mode).lower()
        if self.retrieval_mode not in {'clip_bidirectional', 'surrogate_i2t'}:
            raise ValueError(
                f"Unsupported retrieval_mode={retrieval_mode!r}. "
                "Allowed values: ['surrogate_i2t', 'clip_bidirectional']."
            )
        self.register_buffer('logit_scale_log', torch.log(torch.tensor(1.0 / float(temperature), dtype=torch.float32)))

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale_log.exp().clamp(max=100.0)

    def get_retrieval_temperature(self) -> torch.Tensor:
        return torch.reciprocal(self.get_logit_scale())

    def logits_i2t(self, image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        image_embed = F.normalize(image_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        scale = self.get_logit_scale().to(device=image_embed.device, dtype=image_embed.dtype)
        return (image_embed @ text_embed.t()) * scale

    def similarity_t2i(self, image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        return self.logits_i2t(image_embed, text_embed).t().contiguous()

    def forward(self, image_embed: torch.Tensor, text_embed: torch.Tensor):
        if image_embed.ndim != 2 or text_embed.ndim != 2 or image_embed.shape != text_embed.shape:
            raise ValueError('Vanilla CLIP expects paired image/text embeddings with shape [B, D].')
        zero = image_embed.new_zeros(())
        logits = self.logits_i2t(image_embed, text_embed) if self.use_loss_ret else None
        if self.use_loss_ret:
            target = torch.arange(logits.size(0), device=logits.device)
            loss_i2t = F.cross_entropy(logits, target)
            if self.retrieval_mode == 'clip_bidirectional':
                loss_t2i = F.cross_entropy(logits.t(), target)
                loss_ret = 0.5 * (loss_i2t + loss_t2i)
            else:
                loss_t2i = zero
                loss_ret = loss_i2t
        else:
            loss_i2t = zero
            loss_t2i = zero
            loss_ret = zero
        return {
            'loss_total': loss_ret,
            'host_loss': loss_ret,
            'host_loss_i2t': loss_i2t,
            'host_loss_t2i': loss_t2i,
            'loss_ret': loss_ret,
            'loss_ret_i2t': loss_i2t,
            'loss_ret_t2i': loss_t2i,
            'loss_proxy': zero,
            'loss_proxy_image': zero,
            'loss_proxy_text': zero,
            'loss_proxy_text_exact': zero,
            'loss_align': zero,
            'loss_diag': zero,
            'loss_support': zero,
            'loss_diversity': zero,
            'loss_balance': zero,
            'loss_proxy_image_weighted': zero,
            'loss_proxy_text_weighted': zero,
            'loss_proxy_text_exact_weighted': zero,
            'loss_proxy_weighted': zero,
            'loss_ret_weighted': loss_ret,
            'loss_align_weighted': zero,
            'loss_diag_weighted': zero,
            'loss_support_weighted': zero,
            'loss_diversity_weighted': zero,
            'loss_balance_weighted': zero,
            'lambda_proxy': zero,
            'lambda_proxy_image': zero,
            'lambda_proxy_text': zero,
            'lambda_proxy_text_exact': zero,
            'use_loss_proxy_image': zero,
            'use_loss_proxy_text': zero,
            'use_loss_proxy_text_exact': zero,
            'use_loss_ret': image_embed.new_tensor(float(self.use_loss_ret)),
            'lambda_ret': image_embed.new_tensor(1.0),
            'use_loss_align': zero,
            'lambda_align': zero,
            'use_loss_diag': zero,
            'lambda_diag': zero,
            'use_loss_support': zero,
            'lambda_support': zero,
            'lambda_div': zero,
            'lambda_bal': zero,
            'proxy_temperature': zero,
            'retrieval_temperature': self.get_retrieval_temperature().to(device=image_embed.device, dtype=image_embed.dtype),
            'logit_scale': self.get_logit_scale().to(device=image_embed.device, dtype=image_embed.dtype),
            'surrogate_pairwise_logits': logits,
        }


class ClipHostModel(nn.Module):
    def __init__(self, args, num_classes, train_loader=None, **kwargs):
        super().__init__()
        del num_classes, train_loader, kwargs
        self.args = args
        self.return_debug_outputs = bool(getattr(args, 'return_debug_outputs', False))
        self.lambda_host = float(getattr(args, 'lambda_host', 1.0))
        self.prototype_head = None

        self._validate_config()
        self.base_model, base_cfg, self._convert_clip_weights = self._build_clip_backbone()
        self.embed_dim = int(base_cfg['embed_dim'])

        use_custom_projector = bool(getattr(args, 'use_custom_projector', False))
        projection_dim = int(getattr(args, 'projection_dim', self.embed_dim))
        hidden_dim = int(getattr(args, 'projector_hidden_dim', projection_dim))
        dropout = float(getattr(args, 'projector_dropout', 0.0))
        projector_type = str(getattr(args, 'projector_type', 'mlp2'))
        normalize_projector_outputs = bool(getattr(args, 'normalize_projector_outputs', True))

        self.host_head = nn.ModuleDict(
            {
                'image_projector': _Projector(
                    input_dim=self.embed_dim,
                    output_dim=projection_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    projector_type=projector_type,
                    normalize_output=normalize_projector_outputs,
                    enabled=use_custom_projector,
                ),
                'text_projector': _Projector(
                    input_dim=self.embed_dim,
                    output_dim=projection_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    projector_type=projector_type,
                    normalize_output=normalize_projector_outputs,
                    enabled=use_custom_projector,
                ),
            }
        )
        self.losses = _VanillaClipLoss(
            temperature=float(getattr(args, 'temperature', 0.07)),
            retrieval_mode=str(getattr(args, 'retrieval_mode', 'clip_bidirectional')),
            # Host-only CLIP objective is governed by `use_host_loss`.
            # `use_loss_ret` is reserved for prototype-side retrieval and must not gate host optimization.
            use_loss_ret=bool(getattr(args, 'use_host_loss', True)),
        )
        self._apply_precision_policy()
        self._apply_freeze_policy()

    def _validate_config(self):
        host_type = str(getattr(self.args, 'host_type', 'clip')).lower()
        if host_type != 'clip':
            raise ValueError(f'ClipHostModel requires host.type=clip, got {host_type!r}.')
        if bool(getattr(self.args, 'use_prototype_branch', False)):
            raise ValueError('clip host path currently supports only model.use_prototype_branch=false.')
        if bool(getattr(self.args, 'use_prototype_bank', False)):
            raise ValueError('clip host requires model.use_prototype_bank=false.')
        if bool(getattr(self.args, 'use_image_conditioned_pooling', False)):
            raise ValueError('clip host requires model.use_image_conditioned_pooling=false.')
        if str(getattr(self.args, 'token_policy', 'eos_only')).lower() != 'eos_only':
            raise ValueError('clip host requires text_pooling.token_policy=eos_only.')
        if not bool(getattr(self.args, 'use_host_loss', True)):
            raise ValueError('clip host requires objectives.use_host_loss=true.')

    def _build_clip_backbone(self) -> Tuple[nn.Module, Dict[str, object], object]:
        components = get_original_itself_components()
        model_build = components.model_build
        base_model, base_cfg = model_build.build_CLIP_from_openai_pretrained(
            self.args.pretrain_choice,
            self.args.img_size,
            self.args.stride_size,
        )
        return base_model, base_cfg, model_build.convert_weights

    def _apply_precision_policy(self):
        self._convert_clip_weights(self.base_model)
        self.host_head.half()
        self.losses.half()

    def _apply_freeze_policy(self):
        freeze_image_backbone = bool(getattr(self.args, 'freeze_image_backbone', True))
        freeze_text_backbone = bool(getattr(self.args, 'freeze_text_backbone', True))
        freeze_host_backbone = bool(
            getattr(
                self.args,
                'freeze_host_backbone',
                freeze_image_backbone and freeze_text_backbone,
            )
        )
        freeze_host_retrieval = bool(
            getattr(
                self.args,
                'freeze_host_retrieval',
                getattr(self.args, 'freeze_host_projectors', False),
            )
        )

        if freeze_host_backbone:
            set_group_requires_grad(self, 'host_backbone', False)
        else:
            if freeze_image_backbone:
                _freeze_module(self.base_model.visual)
            if freeze_text_backbone:
                _freeze_module(self.base_model.transformer)
                _freeze_module(self.base_model.token_embedding)
                self.base_model.positional_embedding.requires_grad = False
                self.base_model.ln_final.weight.requires_grad = False
                self.base_model.ln_final.bias.requires_grad = False
                self.base_model.text_projection.requires_grad = False

        if freeze_host_retrieval:
            set_group_requires_grad(self, 'host_retrieval', False)

    def _eos_indices(self, caption_ids: torch.Tensor) -> torch.Tensor:
        special = getattr(self.args, 'special_token_ids', None)
        eos_id = special.get('eos_token_id') if isinstance(special, dict) else None
        if eos_id is not None:
            eos_mask = caption_ids.eq(int(eos_id))
            if bool(eos_mask.any(dim=1).all()):
                return eos_mask.to(dtype=torch.int64).argmax(dim=1)
        return caption_ids.argmax(dim=-1)

    def _encode_image_global(self, images: torch.Tensor) -> torch.Tensor:
        encode_image_intermediates = getattr(self.base_model, 'encode_image_intermediates', None)
        if callable(encode_image_intermediates):
            x = encode_image_intermediates(images, return_all=False, average_attn_weights=True)['projected_tokens']
        else:
            x = self.base_model.encode_image(images)
            if isinstance(x, tuple):
                x = x[0]
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            return x[:, 0, :]
        raise ValueError(f'Unexpected image feature shape: {tuple(x.shape)}')

    def _encode_text_global(self, caption_ids: torch.Tensor) -> torch.Tensor:
        caption_ids = caption_ids.long()
        encode_text_intermediates = getattr(self.base_model, 'encode_text_intermediates', None)
        if callable(encode_text_intermediates):
            x = encode_text_intermediates(caption_ids, return_all=False, average_attn_weights=True)['projected_tokens']
        else:
            x = self.base_model.encode_text(caption_ids)
            if isinstance(x, tuple):
                x = x[0]
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            idx = self._eos_indices(caption_ids)
            return x[torch.arange(x.size(0), device=x.device), idx]
        raise ValueError(f'Unexpected text feature shape: {tuple(x.shape)}')

    def _project_image(self, image_global: torch.Tensor):
        return self.host_head['image_projector'](image_global)

    def _project_text(self, text_global: torch.Tensor):
        return self.host_head['text_projector'](text_global)

    def encode_image_for_retrieval(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_global = self._encode_image_global(images)
        image_projected, _ = self._project_image(image_global)
        return {
            'image_projected': image_projected,
            'host_image_projected': image_projected,
            'summary': image_global,
            'host_summary': image_global,
            'routing_weights': image_projected.new_empty((image_projected.size(0), 0)),
        }

    def encode_text_for_retrieval(self, caption_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_global = self._encode_text_global(caption_ids)
        text_projected, _ = self._project_text(text_global)
        return {
            'text_projected': text_projected,
            'host_text_projected': text_projected,
        }

    def encode_text_basis_for_retrieval(self, caption_ids: torch.Tensor):
        del caption_ids
        raise RuntimeError('encode_text_basis_for_retrieval is unavailable when use_prototype_bank=false.')

    def compute_retrieval_similarity(self, image_features: Dict[str, torch.Tensor], text_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_projected = image_features.get('host_image_projected', image_features['image_projected'])
        text_projected = text_features.get('host_text_projected', text_features['text_projected'])
        similarity = self.losses.similarity_t2i(image_projected, text_projected).float()
        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Retrieval similarity contains NaN or Inf values.')
        return similarity

    def compute_approximate_retrieval_similarity(self, image_features: Dict[str, torch.Tensor], text_basis_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        del image_features, text_basis_features
        raise RuntimeError('compute_approximate_retrieval_similarity is unavailable when use_prototype_bank=false.')

    def named_optimizer_groups(self) -> OrderedDict:
        groups = OrderedDict(
            prototype_bank=[],
            prototype_projectors=[],
            prototype_routing=[],
            prototype_pooling=[],
            prototype_contextualization=[],
            host_projectors=[],
            class_proxies=[],
            image_backbone=[],
            text_backbone=[],
            other=[],
        )
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('host_head'):
                groups['host_projectors'].append((name, parameter))
            elif name.startswith('base_model.visual'):
                groups['image_backbone'].append((name, parameter))
            elif (
                name.startswith('base_model.transformer')
                or name.startswith('base_model.token_embedding')
                or name.startswith('base_model.positional_embedding')
                or name.startswith('base_model.ln_final')
                or name.startswith('base_model.text_projection')
            ):
                groups['text_backbone'].append((name, parameter))
            else:
                groups['other'].append((name, parameter))
        return groups

    def forward(self, batch, epoch=None, current_step=None, return_debug: Optional[bool] = None, disable_proxy_losses: bool = False):
        del epoch, current_step, disable_proxy_losses
        should_return_debug = self.return_debug_outputs if return_debug is None else bool(return_debug)
        image_global = self._encode_image_global(batch['images'])
        text_global = self._encode_text_global(batch['caption_ids'])
        image_projected, image_projected_raw = self._project_image(image_global)
        text_projected, text_projected_raw = self._project_text(text_global)
        losses = self.losses(image_projected, text_projected)

        zero = losses['loss_total'].new_zeros(())
        host_loss_total = losses.get('host_loss', losses['loss_total'])
        loss_total = self.lambda_host * host_loss_total
        outputs = {
            'loss_total': loss_total,
            'loss_host': host_loss_total,
            'loss_host_ret': losses.get('host_loss', losses['loss_ret']),
            'loss_host_ret_i2t': losses.get('host_loss_i2t', losses['loss_ret_i2t']),
            'loss_host_ret_t2i': losses.get('host_loss_t2i', losses['loss_ret_t2i']),
            'loss_host_cid': zero,
            'loss_proto_total': zero,
            'loss_host_weighted': self.lambda_host * host_loss_total,
            'lambda_host': host_loss_total.new_tensor(self.lambda_host),
            'loss_proxy': losses['loss_proxy'],
            'loss_proxy_image': losses['loss_proxy_image'],
            'loss_proxy_text': losses['loss_proxy_text'],
            'loss_proxy_text_exact': losses['loss_proxy_text_exact'],
            # `loss_ret` is reserved for prototype-side retrieval; host-only CLIP training optimizes `loss_host`.
            'loss_ret': zero,
            'loss_align': losses['loss_align'],
            'loss_diag': losses['loss_diag'],
            'loss_support': losses['loss_support'],
            'loss_diversity': losses['loss_diversity'],
            'loss_balance': losses['loss_balance'],
            'loss_proxy_image_weighted': losses['loss_proxy_image_weighted'],
            'loss_proxy_text_weighted': losses['loss_proxy_text_weighted'],
            'loss_proxy_text_exact_weighted': losses['loss_proxy_text_exact_weighted'],
            'loss_proxy_weighted': losses['loss_proxy_weighted'],
            'loss_ret_weighted': zero,
            'loss_align_weighted': losses['loss_align_weighted'],
            'loss_diag_weighted': losses['loss_diag_weighted'],
            'loss_support_weighted': losses['loss_support_weighted'],
            'loss_diversity_weighted': losses['loss_diversity_weighted'],
            'loss_balance_weighted': losses['loss_balance_weighted'],
            'use_loss_proxy_text_exact': losses['use_loss_proxy_text_exact'],
            'use_loss_ret': zero,
            'lambda_ret': zero,
            'lambda_align': losses['lambda_align'],
            'lambda_diag': losses['lambda_diag'],
            'use_loss_support': losses['use_loss_support'],
            'lambda_support': losses['lambda_support'],
            'lambda_div': losses['lambda_div'],
            'lambda_bal': losses['lambda_bal'],
            'proxy_temperature': losses['proxy_temperature'].detach(),
            'retrieval_temperature': losses['retrieval_temperature'].detach(),
            'logit_scale': losses['logit_scale'].detach(),
            'host_retrieval_temperature': losses['retrieval_temperature'].detach(),
            'host_logit_scale': losses['logit_scale'].detach(),
            'alpha': image_projected.new_empty((image_projected.size(0), 0)),
            'z_v': image_projected,
            'z_t_hat_diag': text_projected,
            'z_t_exact_diag': text_projected,
            'surrogate_pairwise_logits': losses['surrogate_pairwise_logits'],
            'host_pairwise_logits': losses['surrogate_pairwise_logits'],
            'debug': {
                'vanilla_clip_mode': image_projected.new_tensor(1.0),
                'vanilla_clip_bidirectional': image_projected.new_tensor(float(self.losses.retrieval_mode == 'clip_bidirectional')),
                'host_loss_total': host_loss_total.detach(),
                'host_loss_ret': losses.get('host_loss', losses['loss_ret']).detach(),
            },
        }
        track_output_grads = bool(getattr(self.args, 'log_debug_metrics', True)) or should_return_debug
        if track_output_grads:
            for grad_key in ('z_v', 'z_t_hat_diag', 'z_t_exact_diag', 'surrogate_pairwise_logits'):
                tensor = outputs.get(grad_key)
                if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                    tensor.retain_grad()
        if should_return_debug:
            outputs['debug'].update(
                {
                    'image_embed_norm_raw': image_projected_raw.detach().norm(dim=-1).mean(),
                    'surrogate_text_embed_norm_raw': text_projected_raw.detach().norm(dim=-1).mean(),
                }
            )
        return outputs


def build_clip_host(args, num_classes, **kwargs):
    return ClipHostModel(args=args, num_classes=num_classes, **kwargs)
