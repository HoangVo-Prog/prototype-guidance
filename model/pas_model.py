from collections import OrderedDict
import logging
from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader

import torch
import torch.nn as nn

from .fusion import ResidualScoreFusion
from .hosts.itself import attach_itself_clip_text_intermediates, get_original_itself_components
from .interfaces import EncoderOutput
from .prototype import TokenMaskBuilder, build_prototype_head, init_mode_requires_data
from .host_heads import build_host_head
from utils.precision import (
    build_autocast_context,
    canonicalize_amp_dtype,
    canonicalize_backbone_precision,
    canonicalize_prototype_precision,
    precision_to_torch_dtype,
)


def build_CLIP_from_openai_pretrained(pretrain_choice, img_size, stride_size):
    components = get_original_itself_components()
    return components.model_build.build_CLIP_from_openai_pretrained(pretrain_choice, img_size, stride_size)


def convert_weights(model):
    components = get_original_itself_components()
    components.model_build.convert_weights(model)


SUPPORTED_PAS_CLIP_BACKBONES = (
    'ViT-B/16',
    'ViT-B/32',
    'ViT-L/14',
)


class PASModel(nn.Module):
    def __init__(self, args, num_classes, train_loader=None):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._validate_pretrain_choice()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self._validate_backbone_contract(base_cfg)
        self.embed_dim = int(base_cfg['embed_dim'])

        self.model_name = getattr(args, 'model_name', 'PAS')
        self.model_variant = getattr(args, 'model_variant', 'pas_v1')
        self.training_mode = str(getattr(args, 'training_mode', 'pas')).lower()
        self.host_type = str(getattr(args, 'host_type', 'clip')).lower()
        self.image_backbone = getattr(args, 'image_backbone', args.pretrain_choice)
        self.text_backbone = getattr(args, 'text_backbone', 'clip_text_transformer')
        self.projection_dim = getattr(args, 'projection_dim', self.embed_dim)
        self.prototype_dim = getattr(args, 'prototype_dim', self.embed_dim)
        self.use_custom_projector = bool(getattr(args, 'use_custom_projector', True))
        self.retrieval_mode = str(getattr(args, 'retrieval_mode', 'surrogate_i2t')).lower()
        self.backbone_precision = canonicalize_backbone_precision(getattr(args, 'backbone_precision', 'fp16'))
        self.prototype_precision = canonicalize_prototype_precision(getattr(args, 'prototype_precision', 'fp32'))
        self.return_debug_outputs = bool(getattr(args, 'return_debug_outputs', False))
        self.itself_return_all = bool(getattr(args, 'itself_return_all', False)) and self.host_type == 'itself'
        self.itself_average_attn_weights = bool(getattr(args, 'itself_average_attn_weights', True))
        self.prototype_eval_image_chunk_size = int(getattr(args, 'prototype_eval_image_chunk_size', 32) or 32)
        self.prototype_eval_text_chunk_size = int(getattr(args, 'prototype_eval_text_chunk_size', 128) or 128)
        self.use_host_loss = bool(getattr(args, 'use_host_loss', True))
        self.lambda_host = float(getattr(args, 'lambda_host', 1.0))
        default_use_prototype_branch = self.training_mode != 'vanilla_clip'
        self.use_prototype_branch = bool(getattr(args, 'use_prototype_branch', default_use_prototype_branch))
        self.use_prototype_bank = bool(getattr(args, 'use_prototype_bank', True)) if self.use_prototype_branch else False
        self.use_image_conditioned_pooling = bool(getattr(args, 'use_image_conditioned_pooling', True)) if self.use_prototype_branch else False
        default_fusion_enabled = self.use_prototype_branch
        self.fusion_enabled = bool(getattr(args, 'fusion_enabled', default_fusion_enabled)) and self.use_prototype_branch
        self.fusion_coefficient_source = str(getattr(args, 'fusion_coefficient_source', 'fixed')).lower()
        (
            self.fusion_lambda_host,
            self.fusion_lambda_prototype,
            self.fusion_legacy_coefficient_mode,
        ) = self._resolve_fusion_weights_from_args()
        # Backward-compatible diagnostic alias retained for existing logs.
        self.fusion_coefficient = self.fusion_lambda_prototype
        explicit_stage = getattr(args, 'training_stage', None)
        self.training_stage = 'joint' if explicit_stage is None else str(explicit_stage).lower()
        if self.host_type == 'itself':
            attach_itself_clip_text_intermediates(self.base_model)

        self._validate_configuration()
        self.token_mask_builder = TokenMaskBuilder(
            token_policy='eos_only',
            special_token_ids=getattr(args, 'special_token_ids', None),
            error_on_empty_kept_tokens=getattr(args, 'error_on_empty_kept_tokens', True),
        )
        self.host_head = build_host_head(
            args=args,
            input_dim=self.embed_dim,
            num_classes=self.num_classes,
        )
        self.prototype_head = None
        if self.use_prototype_branch:
            image_adapter = nn.Identity() if self.embed_dim == self.prototype_dim else nn.Linear(self.embed_dim, self.prototype_dim)
            text_adapter = nn.Identity() if self.embed_dim == self.prototype_dim else nn.Linear(self.embed_dim, self.prototype_dim)
            prototype_init_features = self._maybe_build_prototype_init_features(train_loader=train_loader, image_adapter=image_adapter)
            self.prototype_head = build_prototype_head(
                args,
                input_dim=self.embed_dim,
                num_classes=self.num_classes,
                image_adapter=image_adapter,
                text_adapter=text_adapter,
                prototype_init_features=prototype_init_features,
            )
        self.fusion_module = ResidualScoreFusion(
            enabled=self.fusion_enabled,
            lambda_host=self.fusion_lambda_host,
            lambda_prototype=self.fusion_lambda_prototype,
            coefficient_source=self.fusion_coefficient_source,
        )
        self._log_runtime_mode_summary()
        self._apply_freeze_policy()

    def _validate_pretrain_choice(self):
        pretrain_choice = getattr(self.args, 'pretrain_choice', None)
        if pretrain_choice not in SUPPORTED_PAS_CLIP_BACKBONES:
            raise ValueError(
                'PAS currently supports only ViT CLIP backbones with the token-level runtime contract required by '
                f'prototype routing. Supported `pretrain_choice` values: {list(SUPPORTED_PAS_CLIP_BACKBONES)}. '
                f'Got {pretrain_choice!r}.'
            )

    def _resolve_fusion_weights_from_args(self) -> Tuple[float, float, bool]:
        raw_lambda_host = getattr(self.args, 'fusion_lambda_host', None)
        raw_lambda_prototype = getattr(self.args, 'fusion_lambda_prototype', None)
        raw_legacy_coefficient = getattr(self.args, 'fusion_coefficient', None)

        explicit_lambda_host = raw_lambda_host is not None
        explicit_lambda_prototype = raw_lambda_prototype is not None
        if explicit_lambda_host or explicit_lambda_prototype:
            lambda_host = float(raw_lambda_host) if explicit_lambda_host else (1.0 - float(raw_lambda_prototype))
            lambda_prototype = float(raw_lambda_prototype) if explicit_lambda_prototype else (1.0 - lambda_host)
            return lambda_host, lambda_prototype, False

        if raw_legacy_coefficient is None:
            raw_legacy_coefficient = 1.0 if self.use_prototype_branch else 0.0
        return 1.0, float(raw_legacy_coefficient), True

    def _validate_fusion_weights(self) -> None:
        for field_name, value in (
            ('fusion.lambda_host', self.fusion_lambda_host),
            ('fusion.lambda_prototype', self.fusion_lambda_prototype),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f'{field_name} must be within [0, 1], got {value}.')

        if not self.fusion_legacy_coefficient_mode:
            pair_sum = self.fusion_lambda_host + self.fusion_lambda_prototype
            if abs(pair_sum - 1.0) > 1e-6:
                raise ValueError(
                    'fusion.lambda_host + fusion.lambda_prototype must equal 1.0 '
                    f'(tolerance=1e-6), got {pair_sum}.'
                )

    def _log_runtime_mode_summary(self):
        logger = logging.getLogger('pas.model')
        logger.info(
            'Runtime mode: training_mode=%s host_type=%s host_loss=%s prototype_branch=%s prototype_bank=%s image_conditioned_pooling=%s fusion_enabled=%s fusion_lambda_host=%s fusion_lambda_prototype=%s fusion_source=%s legacy_fusion_coefficient_mode=%s',
            self.training_mode,
            self.host_type,
            self.use_host_loss,
            self.use_prototype_branch,
            self.use_prototype_bank,
            self.use_image_conditioned_pooling,
            self.fusion_enabled,
            self.fusion_lambda_host,
            self.fusion_lambda_prototype,
            self.fusion_coefficient_source,
            self.fusion_legacy_coefficient_mode,
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
        model_logger = logging.getLogger('pas.model')
        self._validate_fusion_weights()
        if self.training_mode not in {'pas', 'vanilla_clip'}:
            raise ValueError(f"model.training_mode must be one of ['pas', 'vanilla_clip']; got {self.training_mode!r}.")
        if self.host_type not in {'clip', 'itself'}:
            raise ValueError(f"host.type must be one of ['clip', 'itself']; got {self.host_type!r}.")
        if self.training_mode == 'vanilla_clip' and self.host_type != 'clip':
            raise ValueError('model.training_mode=vanilla_clip requires host.type=clip.')
        if self.training_mode == 'vanilla_clip':
            if self.use_prototype_branch:
                raise ValueError('model.training_mode=vanilla_clip requires model.use_prototype_branch=false.')
            if self.use_prototype_bank:
                raise ValueError('model.training_mode=vanilla_clip requires model.use_prototype_bank=false.')
            if self.use_image_conditioned_pooling:
                raise ValueError('model.training_mode=vanilla_clip requires model.use_image_conditioned_pooling=false.')
            if str(getattr(self.args, 'token_policy', 'eos_only')).lower() != 'eos_only':
                raise ValueError('model.training_mode=vanilla_clip requires text_pooling.token_policy=eos_only.')
            if self.retrieval_mode != 'clip_bidirectional':
                raise ValueError('model.training_mode=vanilla_clip requires loss.retrieval_mode=clip_bidirectional.')
            if not self.use_host_loss:
                raise ValueError('model.training_mode=vanilla_clip requires objectives.use_host_loss=true.')
            if str(getattr(self.args, 'retrieval_scorer', 'exact')).lower() != 'exact':
                raise ValueError('model.training_mode=vanilla_clip requires evaluation.retrieval_scorer=exact.')
            incompatible_flags = {
                'loss.use_loss_proxy_image': bool(getattr(self.args, 'use_loss_proxy_image', False)),
                'loss.use_loss_proxy_text': bool(getattr(self.args, 'use_loss_proxy_text', False)),
                'loss.use_loss_proxy_text_exact': bool(getattr(self.args, 'use_loss_proxy_text_exact', False)),
                'loss.use_loss_align': bool(getattr(self.args, 'use_loss_align', False)),
                'loss.use_loss_dir': bool(getattr(self.args, 'use_loss_dir', False)),
                'loss.use_loss_gap': bool(getattr(self.args, 'use_loss_gap', False)),
                'loss.use_loss_sup': bool(getattr(self.args, 'use_loss_sup', False)),
                'loss.use_loss_diag': bool(getattr(self.args, 'use_loss_diag', False)),
                'loss.use_loss_support': bool(getattr(self.args, 'use_loss_support', False)),
                'loss.use_balancing_loss': bool(getattr(self.args, 'use_balancing_loss', False)),
                'loss.use_diversity_loss': bool(getattr(self.args, 'use_diversity_loss', False)),
            }
            enabled_incompatible = sorted(name for name, enabled in incompatible_flags.items() if enabled)
            if enabled_incompatible:
                raise ValueError(
                    'model.training_mode=vanilla_clip does not support prototype/auxiliary losses. '
                    f'Disable: {enabled_incompatible}.'
                )
        elif self.retrieval_mode == 'clip_bidirectional':
            raise ValueError('loss.retrieval_mode=clip_bidirectional is only supported when model.training_mode=vanilla_clip.')
        if self.use_prototype_branch and self.use_prototype_bank and not self.use_image_conditioned_pooling:
            raise ValueError(
                'use_prototype_bank=true requires use_image_conditioned_pooling=true. '
                'Prototype-routed training with text-only pooling is no longer supported.'
            )

        if self.training_stage == 'stage0' and self.use_prototype_branch:
            raise ValueError('training.stage=stage0 is reserved for host-only baselines and requires model.use_prototype_branch=false.')
        retrieval_scorer = str(getattr(self.args, 'retrieval_scorer', 'exact')).lower()
        if (not self.use_prototype_branch or not self.use_prototype_bank) and retrieval_scorer == 'approximate':
            model_logger.warning(
                'evaluation.retrieval_scorer=approximate requires a live prototype bank; evaluation may fall back to exact scoring.'
            )
        if not bool(getattr(self.args, 'normalize_projector_outputs', True)):
            model_logger.warning(
                'model.normalize_projector_outputs=false is accepted for ablation bookkeeping. The runtime will continue '
                'without the previous hard constraint, so train/eval geometry may differ from the canonical PAS setup.'
            )
        proxy_losses_requested = any(
            bool(getattr(self.args, attr_name, False))
            for attr_name in ('use_loss_proxy_image', 'use_loss_proxy_text', 'use_loss_proxy_text_exact')
        )
        if self.use_prototype_branch and proxy_losses_requested and self.num_classes <= 0:
            raise ValueError('PASModel requires num_classes > 0 when prototype proxy-supervised training is active.')
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
        legacy_retrieval_flags = {
            'use_loss_ret_exact': 'Legacy exact retrieval training is removed. Use loss.use_loss_ret for row-wise surrogate image-to-text retrieval only.',
            'use_loss_ret_exact_image': 'Legacy exact image-side retrieval training is removed. The only valid retrieval loss is row-wise surrogate image-to-text retrieval.',
            'use_loss_ret_exact_text': 'Legacy text-to-image retrieval training is invalid because surrogate text embeddings are image-conditioned.',
            'lambda_ret_exact': 'Legacy exact retrieval weighting is removed. Use loss.lambda_ret for surrogate row-wise retrieval.',
            'lambda_ret_exact_image': 'Legacy exact image-side retrieval weighting is removed. Use loss.lambda_ret for surrogate row-wise retrieval.',
            'lambda_ret_exact_text': 'Legacy text-side retrieval weighting is removed. Column-wise text retrieval is invalid for image-conditioned text embeddings.',
            'ret_exact_temperature': 'Legacy exact retrieval temperature is removed. model.temperature defines the surrogate retrieval temperature.',
            'freeze_prototype': 'training.freeze_prototype was replaced by training.freeze_prototype_side.',
            'freeze_proxy': 'training.freeze_proxy was replaced by training.freeze_prototype_side.',
        }
        for attr_name, message in legacy_retrieval_flags.items():
            if not hasattr(self.args, attr_name):
                continue
            value = getattr(self.args, attr_name)
            if value in (None, False, 0, 0.0, ''):
                continue
            raise ValueError(message)

        allowed_training_stages = {'stage0', 'stage1', 'stage2', 'stage3', 'joint'}
        explicit_stage = getattr(self.args, 'training_stage', None)
        if explicit_stage is None:
            self.training_stage = 'joint'
        else:
            self.training_stage = str(explicit_stage).lower()
        if self.training_stage not in allowed_training_stages:
            raise ValueError(f"training.stage must be one of ['stage0', 'stage1', 'stage2', 'stage3', 'joint']; got {self.training_stage!r}.")
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
        self.freeze_host_projectors = bool(getattr(self.args, 'freeze_host_projectors', False))
        self.freeze_prototype_side = bool(getattr(self.args, 'freeze_prototype_side', False))

        if self.freeze_image_backbone:
            self._freeze_module(self.base_model.visual)
        if self.freeze_text_backbone:
            self._freeze_module(self.base_model.transformer)
            self._freeze_module(self.base_model.token_embedding)
            self.base_model.positional_embedding.requires_grad = False
            self.base_model.ln_final.weight.requires_grad = False
            self.base_model.ln_final.bias.requires_grad = False
            self.base_model.text_projection.requires_grad = False
        if self.freeze_host_projectors:
            if hasattr(self.host_head, 'freeze_trainable_head'):
                self.host_head.freeze_trainable_head()
            else:
                self._freeze_module(self.host_head)
        if self.freeze_prototype_side and self.prototype_head is not None:
            self._freeze_module(self.prototype_head)

    def _encode_image_intermediates(self, image: torch.Tensor, return_all: bool, average_attn_weights: bool) -> Dict[str, Optional[torch.Tensor]]:
        if hasattr(self.base_model, 'encode_image_intermediates'):
            return self.base_model.encode_image_intermediates(
                image,
                return_all=return_all,
                average_attn_weights=average_attn_weights,
            )

        if return_all and hasattr(self.base_model, 'encode_image_all_atten'):
            projected_tokens, attention_weights = self.base_model.encode_image_all_atten(
                image,
                average_attn_weights=average_attn_weights,
            )
        else:
            projected_tokens, attention_weights = self.base_model.encode_image(image)

        if torch.is_tensor(projected_tokens) and projected_tokens.ndim == 2:
            projected_tokens = projected_tokens.unsqueeze(1)
        return {
            'projected_tokens': projected_tokens,
            'pre_projection_tokens': None,
            'attention_weights': attention_weights,
        }

    def _encode_text_intermediates(self, text: torch.Tensor, return_all: bool, average_attn_weights: bool) -> Dict[str, Optional[torch.Tensor]]:
        if hasattr(self.base_model, 'encode_text_intermediates'):
            return self.base_model.encode_text_intermediates(
                text,
                return_all=return_all,
                average_attn_weights=average_attn_weights,
            )

        if return_all and hasattr(self.base_model, 'encode_text_all_atten'):
            projected_tokens, attention_weights = self.base_model.encode_text_all_atten(
                text,
                average_attn_weights=average_attn_weights,
            )
        else:
            projected_tokens, attention_weights = self.base_model.encode_text(text)

        if torch.is_tensor(projected_tokens) and projected_tokens.ndim == 2:
            projected_tokens = projected_tokens.unsqueeze(1)
        return {
            'projected_tokens': projected_tokens,
            'pre_projection_tokens': None,
            'attention_weights': attention_weights,
        }


    def _collect_train_image_records(self, train_loader) -> list:
        train_dataset = getattr(train_loader, 'dataset', None)
        raw_records = getattr(train_dataset, 'dataset', None)
        if raw_records is None:
            raise ValueError(
                'Automatic prototype init fallback expected `train_loader.dataset.dataset` to expose raw training '
                'records of the form `(pid, image_id, img_path, caption)`, but that structure was not found. '
                'Provide `prototype_init_path` or update the training dataset wrapper so the raw train split is accessible.'
            )

        unique_records = []
        seen_image_keys = set()
        for record_index, record in enumerate(raw_records):
            if not isinstance(record, (tuple, list)) or len(record) < 3:
                raise ValueError(
                    'Automatic prototype init fallback expected each raw train record to look like '
                    f'`(pid, image_id, img_path, caption)`, but found `{record!r}` at index {record_index}. '
                    'Provide `prototype_init_path` or update the dataset contract.'
                )
            pid = int(record[0])
            image_key = record[1]
            img_path = record[2]
            try:
                is_duplicate = image_key in seen_image_keys
            except TypeError:
                image_key = img_path
                is_duplicate = image_key in seen_image_keys
            if is_duplicate:
                continue
            seen_image_keys.add(image_key)
            unique_records.append((pid, img_path))

        if not unique_records:
            raise ValueError(
                'Automatic prototype init fallback found no unique training images. '
                'Provide `prototype_init_path` or verify that the train split is populated.'
            )
        return unique_records

    def _extract_train_image_embeddings(self, train_loader, image_adapter: nn.Module) -> torch.Tensor:
        from datasets.bases import ImageDataset
        from datasets.build import build_transforms

        logger = logging.getLogger('pas.prototype_init')
        unique_records = self._collect_train_image_records(train_loader)
        transform = build_transforms(img_size=getattr(self.args, 'img_size', (384, 128)), aug=False, is_train=False)
        image_dataset = ImageDataset(
            image_pids=[pid for pid, _ in unique_records],
            img_paths=[img_path for _, img_path in unique_records],
            transform=transform,
        )
        batch_size = max(1, min(int(getattr(self.args, 'batch_size', 32) or 32), len(image_dataset)))
        num_workers = max(int(getattr(self.args, 'num_workers', 0) or 0), 0)
        image_loader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        extraction_device = torch.device(getattr(self.args, 'device', 'cpu'))
        if extraction_device.type == 'cuda' and not torch.cuda.is_available():
            extraction_device = torch.device('cpu')

        self.base_model.to(extraction_device)
        image_adapter.to(extraction_device)
        base_model_was_training = self.base_model.training
        adapter_was_training = image_adapter.training
        self.base_model.eval()
        image_adapter.eval()

        feature_batches = []
        adapter_parameter = next(image_adapter.parameters(), None)
        adapter_dtype = adapter_parameter.dtype if adapter_parameter is not None else torch.float32
        try:
            with torch.no_grad():
                with build_autocast_context(self.args, extraction_device):
                    for _, images in image_loader:
                        images = images.to(extraction_device)
                        image_outputs = self._encode_image_intermediates(
                            images,
                            return_all=False,
                            average_attn_weights=True,
                        )
                        projected_tokens = image_outputs.get('projected_tokens')
                        if not torch.is_tensor(projected_tokens) or projected_tokens.ndim != 3:
                            raise ValueError(
                                'Automatic prototype init fallback expected the image encoder to return '
                                '`projected_tokens` with shape [B, T, D], but that representation was unavailable. '
                                'Update the image encoder contract or provide `prototype_init_path`.'
                            )
                        projected_pooled = projected_tokens[:, 0, :].float()
                        if projected_pooled.size(-1) != self.embed_dim:
                            raise ValueError(
                                'Automatic prototype init fallback expected CLS-pooled image embeddings with '
                                f'dim {self.embed_dim}, but found dim {projected_pooled.size(-1)}. '
                                'Update the image encoder contract or provide `prototype_init_path`.'
                            )
                        prototype_features = image_adapter(projected_pooled.to(dtype=adapter_dtype, device=extraction_device))
                        if prototype_features.ndim != 2:
                            raise ValueError(
                                'Automatic prototype init fallback expected prototype-space image embeddings '
                                f'with shape [B, D], but found shape {tuple(prototype_features.shape)} after `image_adapter`. '
                                'Update the prototype projection path or provide `prototype_init_path`.'
                            )
                        if prototype_features.size(-1) != self.prototype_dim:
                            raise ValueError(
                                'Automatic prototype init fallback expected prototype-space image embeddings '
                                f'with dim {self.prototype_dim}, but found dim {prototype_features.size(-1)} after `image_adapter`. '
                                'Update the adapter/prototype configuration or provide `prototype_init_path`.'
                            )
                        feature_batches.append(prototype_features.detach().float().cpu())
        finally:
            self.base_model.train(base_model_was_training)
            image_adapter.train(adapter_was_training)

        if not feature_batches:
            raise ValueError(
                'Automatic prototype init fallback did not extract any train image embeddings. '
                'Provide `prototype_init_path` or verify the train split and image loader.'
            )
        features = torch.cat(feature_batches, dim=0)
        logger.info(
            'Built prototype init features from train split unique_images=%s embedding_shape=%s representation=%s',
            features.size(0),
            tuple(features.shape),
            'image.projected_pooled->prototype_head.image_adapter',
        )
        return features

    def _broadcast_train_image_embeddings(
        self,
        features: Optional[torch.Tensor],
        extraction_error: Optional[Exception] = None,
    ) -> torch.Tensor:
        logger = logging.getLogger('pas.prototype_init')
        if not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            if extraction_error is not None:
                raise RuntimeError(f'Automatic prototype init fallback failed: {extraction_error}') from extraction_error
            if features is None:
                raise ValueError('Distributed prototype init broadcast expected local features, but none were provided.')
            return features.cpu()

        rank = torch.distributed.get_rank()
        broadcast_device = torch.device(getattr(self.args, 'device', 'cpu'))
        if broadcast_device.type != 'cuda':
            raise ValueError(
                'Automatic prototype init fallback under distributed training expected a CUDA device so ranks can '
                f'broadcast image embeddings consistently, but got device={broadcast_device}. '
                'Provide `prototype_init_path` or run distributed training on CUDA.'
            )

        payload = [None]
        if rank == 0:
            if extraction_error is not None:
                payload[0] = {
                    'ok': False,
                    'error': str(extraction_error),
                }
            elif features is None:
                payload[0] = {
                    'ok': False,
                    'error': 'Rank 0 did not produce train image embeddings for prototype initialization.',
                }
            else:
                payload[0] = {
                    'ok': True,
                    'shape': tuple(features.shape),
                }
        torch.distributed.broadcast_object_list(payload, src=0)
        metadata = payload[0]
        if not metadata.get('ok', False):
            raise RuntimeError(
                'Automatic prototype init fallback failed on rank 0: '
                f"{metadata.get('error', 'unknown error')}"
            )

        if rank == 0:
            broadcast_tensor = features.to(device=broadcast_device, dtype=torch.float32)
        else:
            broadcast_tensor = torch.empty(metadata['shape'], device=broadcast_device, dtype=torch.float32)
        torch.distributed.broadcast(broadcast_tensor, src=0)
        logger.info(
            'Received broadcast prototype init features from rank 0 embedding_shape=%s',
            tuple(metadata['shape']),
        )
        return broadcast_tensor.cpu()

    def _maybe_build_prototype_init_features(self, train_loader, image_adapter: nn.Module) -> Optional[torch.Tensor]:
        logger = logging.getLogger('pas.prototype_init')
        if not self.use_prototype_branch or not self.use_prototype_bank:
            logger.info('Skipping prototype initialization because the prototype bank is disabled.')
            return None
        init_mode = getattr(self.args, 'prototype_init', 'normalized_random')
        init_path = getattr(self.args, 'prototype_init_path', None)
        requires_data = init_mode_requires_data(init_mode)
        logger.info(
            'Prototype init selection mode=%s path_provided=%s requires_data=%s',
            init_mode,
            bool(init_path),
            requires_data,
        )
        if not requires_data:
            return None
        if init_path:
            logger.info('Prototype init path provided; preserving existing path-based initialization behavior.')
            return None
        if not bool(getattr(self.args, 'training', True)):
            raise ValueError(
                f'Prototype init mode `{init_mode}` requires train image embeddings when `prototype_init_path` is missing, '
                'but the model is not in training mode so no train split is available. Provide `prototype_init_path`.'
            )
        if train_loader is None:
            raise ValueError(
                f'Prototype init mode `{init_mode}` requires train image embeddings when `prototype_init_path` is missing. '
                'Build the model with `train_loader=...` or provide `prototype_init_path`.'
            )
        logger.info('Triggering automatic train-image-embedding fallback for prototype init mode=%s', init_mode)
        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        local_features = None
        extraction_error = None
        if rank == 0:
            try:
                local_features = self._extract_train_image_embeddings(train_loader, image_adapter)
            except Exception as exc:  # pragma: no cover - distributed safety path
                extraction_error = exc
        return self._broadcast_train_image_embeddings(local_features, extraction_error=extraction_error)

    def _resolve_text_states(self, text_output: EncoderOutput) -> torch.Tensor:
        if text_output.pre_projection_tokens is not None:
            return text_output.pre_projection_tokens
        if self.host_type == 'itself':
            raise RuntimeError(
                'ITSELF host requires text pre-projection token states for prototype routing, but the current '
                'CLIP runtime did not expose them. Ensure the original ITSELF CLIP internals are available and '
                '`encode_text_intermediates` returns `pre_projection_tokens` before `text_projection`.'
            )
        if not getattr(self, '_warned_projected_text_state_fallback', False):
            logging.getLogger('pas.model').warning(
                'Text pre-projection token states are unavailable from the current CLIP runtime; '
                'falling back to projected text tokens for prototype routing.'
            )
            self._warned_projected_text_state_fallback = True
        return text_output.projected_tokens

    def _prototype_dtype(self) -> torch.dtype:
        return precision_to_torch_dtype(self.prototype_precision)

    def _cast_to_prototype_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=self._prototype_dtype())

    def _encode_host_image_branch(self, image_output: EncoderOutput, return_debug: bool = False) -> Dict[str, object]:
        return self.host_head.encode_image_branch(
            image_output,
            return_debug=return_debug,
        )

    def _encode_host_text_branch(self, text_output: EncoderOutput, return_debug: bool = False) -> Dict[str, object]:
        raise RuntimeError('_encode_host_text_branch requires token ids; call host_head.encode_text_branch directly.')

    def _zero_loss_outputs(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        return {
            'loss_total': zero,
            'loss_proto': zero,
            'loss_proxy': zero,
            'loss_proxy_image': zero,
            'loss_proxy_text': zero,
            'loss_proxy_text_exact': zero,
            'loss_ret': zero,
            'loss_align': zero,
            'loss_dir': zero,
            'loss_gap': zero,
            'loss_sup': zero,
            'loss_diag': zero,
            'loss_support': zero,
            'loss_diversity': zero,
            'loss_balance': zero,
            'loss_proxy_image_weighted': zero,
            'loss_proxy_text_weighted': zero,
            'loss_proxy_text_exact_weighted': zero,
            'loss_proxy_weighted': zero,
            'loss_ret_weighted': zero,
            'loss_weight_ret': zero,
            'loss_weight_ret_weighted': zero,
            'loss_align_weighted': zero,
            'loss_dir_weighted': zero,
            'loss_gap_weighted': zero,
            'loss_sup_weighted': zero,
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
            'use_loss_ret': zero,
            'lambda_ret': zero,
            'use_loss_weight_ret': zero,
            'lambda_weight_ret': zero,
            'weight_ret_margin_delta': zero,
            'weight_ret_tau': zero,
            'weight_ret_detach_host': zero,
            'weight_ret_normalize_mean_one': zero,
            'use_loss_align': zero,
            'lambda_align': zero,
            'use_loss_dir': zero,
            'lambda_dir': zero,
            'use_loss_gap': zero,
            'lambda_gap': zero,
            'use_loss_sup': zero,
            'lambda_sup': zero,
            'use_loss_diag': zero,
            'lambda_diag': zero,
            'use_loss_support': zero,
            'lambda_support': zero,
            'lambda_div': zero,
            'lambda_bal': zero,
            'prototype_gap_margin': zero,
            'prototype_support_target': zero,
            'proxy_temperature': zero,
            'diag_temperature': zero,
            'retrieval_temperature': zero,
            'logit_scale': zero,
            'debug_metrics': {},
        }

    def extract_image_features(self, image: torch.Tensor) -> EncoderOutput:
        image_outputs = self._encode_image_intermediates(
            image,
            return_all=self.itself_return_all,
            average_attn_weights=self.itself_average_attn_weights,
        )
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
            attention_weights=image_outputs.get('attention_weights'),
            token_mask=None,
            special_token_positions={},
            pooling_mode='cls',
            metadata={
                'encoder': 'image',
                'backbone': self.image_backbone,
                'backbone_precision': self.backbone_precision,
                'prototype_precision': self.prototype_precision,
                'host_type': self.host_type,
            },
        )

    def extract_text_features(self, text: torch.Tensor) -> EncoderOutput:
        text_outputs = self._encode_text_intermediates(
            text.long(),
            return_all=self.itself_return_all,
            average_attn_weights=self.itself_average_attn_weights,
        )
        projected_tokens = text_outputs['projected_tokens'].float()
        pre_projection_tokens = text_outputs['pre_projection_tokens']
        if pre_projection_tokens is not None:
            pre_projection_tokens = pre_projection_tokens.float()
        token_mask = self.token_mask_builder.build_valid_mask(text.long())
        special_positions = self.token_mask_builder.get_special_token_positions(text.long(), attention_mask=token_mask)
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
            attention_weights=text_outputs.get('attention_weights'),
            token_mask=token_mask,
            special_token_positions=special_positions,
            pooling_mode='eos_only' if not self.use_prototype_branch else ('image_conditioned' if self.use_image_conditioned_pooling else 'text_only'),
            metadata={
                'encoder': 'text',
                'backbone': self.text_backbone,
                'backbone_precision': self.backbone_precision,
                'prototype_precision': self.prototype_precision,
                'host_type': self.host_type,
            },
        )

    def encode_image_for_retrieval(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_output = self.extract_image_features(image)
        host_image = self.host_head.encode_image_branch(image_output, return_debug=False)
        summary = host_image['summary']
        routing_weights = host_image['routing_weights']
        if self.prototype_head is not None:
            prototype_image = self.prototype_head.encode_image_branch(
                self._cast_to_prototype_dtype(image_output.projected_pooled),
                return_debug=False,
            )
            summary = prototype_image['summary']
            routing_weights = prototype_image['routing_weights']
        outputs = dict(host_image)
        outputs.update(
            {
                'host_image_projected': host_image['image_projected'],
                'host_summary': host_image['summary'],
                'summary': summary,
                'routing_weights': routing_weights,
            }
        )
        if self.prototype_head is not None:
            outputs['prototype_image_projected'] = prototype_image['image_projected']
            outputs['prototype_summary'] = prototype_image['summary']
        return outputs

    def encode_text_for_retrieval(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_output = self.extract_text_features(text)
        host_text = self.host_head.encode_text_branch(text_output, text.long(), return_debug=False)
        outputs = dict(host_text)
        outputs['host_text_projected'] = host_text['text_projected']
        if self.prototype_head is not None:
            outputs.update(
                {
                    'text_token_states': self._cast_to_prototype_dtype(self._resolve_text_states(text_output)),
                    'token_ids': text.long(),
                    'attention_mask': text_output.token_mask,
                    'special_token_positions': {key: value for key, value in text_output.special_token_positions.items()},
                }
            )
        return outputs

    def encode_text_basis_for_retrieval(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.prototype_head is None or not self.use_prototype_bank:
            raise RuntimeError('encode_text_basis_for_retrieval is unavailable when model.use_prototype_bank=false. Use evaluation.retrieval_scorer=exact.')
        text_output = self.extract_text_features(text)
        host_text = self.host_head.encode_text_branch(text_output, text.long(), return_debug=False)
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
            'host_text_features': host_text,
            'host_text_projected': host_text['text_projected'],
            'basis_bank': basis_outputs['basis_bank'],
        }

    def _compute_host_similarity(self, image_features: Dict[str, torch.Tensor], text_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        similarity = self.host_head.compute_similarity_matrix(image_features, text_features)
        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Host retrieval similarity contains NaN or Inf values.')
        return similarity.float()

    def fuse_retrieval_similarity(
        self,
        host_similarity: torch.Tensor,
        prototype_similarity: Optional[torch.Tensor] = None,
        lambda_host: Optional[float] = None,
        lambda_prototype: Optional[float] = None,
    ) -> torch.Tensor:
        similarity = self.fusion_module(
            host_similarity=host_similarity,
            prototype_similarity=prototype_similarity,
            lambda_host=lambda_host,
            lambda_prototype=lambda_prototype,
        )
        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Retrieval similarity contains NaN or Inf values.')
        return similarity.float()

    def compute_retrieval_similarity_components(
        self,
        image_features: Dict[str, torch.Tensor],
        text_features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        host_similarity = self._compute_host_similarity(image_features, text_features)
        prototype_similarity = None
        if self.prototype_head is not None and 'text_token_states' in text_features:
            prototype_similarity = self.prototype_head.compute_pairwise_similarity(
                image_projected=self._cast_to_prototype_dtype(image_features.get('prototype_image_projected', image_features.get('host_image_projected', image_features['image_projected']))),
                summaries=self._cast_to_prototype_dtype(image_features.get('prototype_summary', image_features['summary'])),
                text_token_states=self._cast_to_prototype_dtype(text_features['text_token_states']),
                token_ids=text_features['token_ids'],
                attention_mask=text_features.get('attention_mask'),
                special_token_positions=text_features.get('special_token_positions'),
                image_chunk_size=self.prototype_eval_image_chunk_size,
                text_chunk_size=self.prototype_eval_text_chunk_size,
            ).float()
        return host_similarity.float(), prototype_similarity.float() if isinstance(prototype_similarity, torch.Tensor) else None

    def compute_retrieval_similarity(self, image_features: Dict[str, torch.Tensor], text_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        host_similarity, prototype_similarity = self.compute_retrieval_similarity_components(
            image_features=image_features,
            text_features=text_features,
        )
        return self.fuse_retrieval_similarity(host_similarity, prototype_similarity)

    def compute_approximate_retrieval_similarity_components(
        self,
        image_features: Dict[str, torch.Tensor],
        text_basis_features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prototype_head is None or not self.use_prototype_bank:
            raise RuntimeError('compute_approximate_retrieval_similarity is unavailable when model.use_prototype_bank=false. Use evaluation.retrieval_scorer=exact.')
        host_similarity = self._compute_host_similarity(
            image_features,
            text_basis_features.get('host_text_features', {
                'text_projected': text_basis_features['host_text_projected'],
                'host_text_projected': text_basis_features['host_text_projected'],
            }),
        )
        prototype_similarity = self.prototype_head.compute_approximate_pairwise_similarity(
            image_projected=self._cast_to_prototype_dtype(image_features.get('prototype_image_projected', image_features.get('host_image_projected', image_features['image_projected']))),
            routing_weights=self._cast_to_prototype_dtype(image_features['routing_weights']),
            basis_bank=self._cast_to_prototype_dtype(text_basis_features['basis_bank']),
            image_chunk_size=self.prototype_eval_image_chunk_size,
            text_chunk_size=self.prototype_eval_text_chunk_size,
        ).float()
        return host_similarity.float(), prototype_similarity.float()

    def compute_approximate_retrieval_similarity(self, image_features: Dict[str, torch.Tensor], text_basis_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        host_similarity, prototype_similarity = self.compute_approximate_retrieval_similarity_components(
            image_features=image_features,
            text_basis_features=text_basis_features,
        )
        similarity = self.fuse_retrieval_similarity(host_similarity, prototype_similarity)
        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Approximate retrieval similarity contains NaN or Inf values.')
        return similarity.float()

    def _build_debug_outputs(
        self,
        image_output: EncoderOutput,
        text_output: EncoderOutput,
        host_outputs: Dict[str, object],
        prototype_outputs: Dict[str, object],
    ) -> Dict[str, object]:
        debug = dict(host_outputs.get('metrics', {}))
        debug.update(prototype_outputs.get('metrics', {}))
        debug.update(
            {
                'fusion_coefficient': torch.tensor(self.fusion_coefficient, device=host_outputs['image_projected'].device, dtype=host_outputs['image_projected'].dtype),
                'fusion_lambda_host': torch.tensor(self.fusion_lambda_host, device=host_outputs['image_projected'].device, dtype=host_outputs['image_projected'].dtype),
                'fusion_lambda_prototype': torch.tensor(self.fusion_lambda_prototype, device=host_outputs['image_projected'].device, dtype=host_outputs['image_projected'].dtype),
                'fusion_enabled': torch.tensor(float(self.fusion_enabled), device=host_outputs['image_projected'].device, dtype=host_outputs['image_projected'].dtype),
                'image_global': image_output.projected_pooled.detach(),
                'text_tokens': self._resolve_text_states(text_output).detach(),
                'token_mask': text_output.token_mask.detach(),
                'special_token_positions': {key: value.detach() for key, value in text_output.special_token_positions.items()},
                'host_image_projected': host_outputs['image_projected'].detach(),
                'host_text_projected': host_outputs['surrogate_text_projected'].detach(),
                'host_pairwise_logits': host_outputs.get('surrogate_pairwise_logits').detach() if isinstance(host_outputs.get('surrogate_pairwise_logits'), torch.Tensor) else None,
                'alpha': prototype_outputs['routing_weights'].detach(),
                'Q': prototype_outputs['summary'].detach(),
                'Theta_v': prototype_outputs['prototypes'].detach(),
                'Theta_tilde': prototype_outputs['contextualized_prototypes'].detach(),
                'basis_bank': prototype_outputs['basis_bank'].detach(),
                'Z_v': prototype_outputs.get('image_projected', host_outputs['image_projected']).detach(),
                'Z_v_raw': prototype_outputs.get('image_projected_raw', host_outputs['image_projected_raw']).detach(),
                'Z_t': prototype_outputs['surrogate_text_projected'].detach(),
                'Z_t_raw': prototype_outputs['surrogate_text_projected_raw'].detach(),
                'Z_t_exact': prototype_outputs['exact_text_projected'].detach(),
                'Z_t_exact_raw': prototype_outputs['exact_text_projected_raw'].detach(),
            }
        )
        for key in ('exact_token_weights', 'token_valid_mask', 'token_keep_mask', 'beta_logits_masked', 'surrogate_pooled_text', 'exact_pooled_text'):
            value = prototype_outputs.get(key)
            if isinstance(value, torch.Tensor):
                debug[key] = value.detach()
        if 'exact_token_weights' in prototype_outputs:
            debug['beta'] = prototype_outputs['exact_token_weights'].detach()
        if 'surrogate_pooled_text' in prototype_outputs:
            debug['T_pool'] = prototype_outputs['surrogate_pooled_text'].detach()
            debug['T_hat_pool'] = prototype_outputs['surrogate_pooled_text'].detach()
        if 'exact_pooled_text' in prototype_outputs:
            debug['T_exact_pool'] = prototype_outputs['exact_pooled_text'].detach()
        for key in ('basis_token_scores', 'basis_token_weights', 'basis_beta_logits_masked', 'image_proxy_logits', 'text_proxy_logits', 'text_exact_proxy_logits', 'surrogate_retrieval_logits', 'surrogate_pairwise_logits', 'class_proxies'):
            value = prototype_outputs.get('debug', {}).get(key)
            if isinstance(value, torch.Tensor):
                debug[key] = value.detach()
            elif value is not None:
                debug[key] = value
        return debug

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
            if name.startswith('prototype_head.losses.class_proxies'):
                groups['class_proxies'].append((name, parameter))
            elif name.startswith('prototype_head.prototype_bank'):
                groups['prototype_bank'].append((name, parameter))
            elif name.startswith('prototype_head.contextualizer'):
                groups['prototype_contextualization'].append((name, parameter))
            elif name.startswith('prototype_head.router'):
                groups['prototype_routing'].append((name, parameter))
            elif (
                name.startswith('prototype_head.text_pool_query')
                or name.startswith('prototype_head.token_pooler')
                or name.startswith('prototype_head.token_scorer')
                or name.startswith('prototype_head.token_mask_builder')
            ):
                groups['prototype_pooling'].append((name, parameter))
            elif (
                name.startswith('prototype_head.image_projector')
                or name.startswith('prototype_head.text_projector')
                or name.startswith('prototype_head.proto_query_proj')
                or name.startswith('prototype_head.image_adapter')
                or name.startswith('prototype_head.text_adapter')
            ):
                groups['prototype_projectors'].append((name, parameter))
            elif name.startswith('host_head'):
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
        del epoch
        images = batch['images']
        caption_ids = batch['caption_ids']
        pids = batch.get('pids')
        proxy_losses_requested = self.prototype_head is not None and any(
            bool(getattr(self.args, attr_name, False))
            for attr_name in ('use_loss_proxy_image', 'use_loss_proxy_text', 'use_loss_proxy_text_exact')
        )
        if self.prototype_head is not None and proxy_losses_requested and pids is None:
            raise KeyError("PASModel.forward requires batch['pids'] when prototype proxy losses are enabled.")
        if pids is not None:
            for label_key in ('image_pids', 'caption_pids'):
                paired_labels = batch.get(label_key)
                if paired_labels is None:
                    continue
                paired_labels = paired_labels.to(device=pids.device, dtype=pids.dtype)
                if paired_labels.shape != pids.shape or not torch.equal(paired_labels, pids):
                    raise ValueError(
                        f"Batch label mismatch: batch['pids'] and batch['{label_key}'] must match exactly for paired training samples."
                    )

        image_output = self.extract_image_features(images)
        text_output = self.extract_text_features(caption_ids)
        should_return_debug = self.return_debug_outputs if return_debug is None else bool(return_debug)

        host_outputs = self.host_head(
            image_output,
            text_output,
            caption_ids,
            pids=pids,
            return_debug=should_return_debug,
            current_step=current_step,
        )
        host_losses = host_outputs['losses']

        if self.prototype_head is None:
            empty_alpha = host_outputs['routing_weights']
            empty_bank = empty_alpha.new_empty((text_output.projected_pooled.size(0), 0, self.prototype_dim))
            prototype_outputs = {
                'routing_weights': empty_alpha,
                'summary': self._cast_to_prototype_dtype(image_output.projected_pooled),
                'prototypes': empty_alpha.new_empty((0, self.prototype_dim)),
                'contextualized_prototypes': empty_alpha.new_empty((0, self.prototype_dim)),
                'basis_bank': empty_bank,
                'token_valid_mask': text_output.token_mask,
                'token_keep_mask': text_output.token_mask,
                'beta_logits_masked': empty_alpha.new_empty((text_output.projected_pooled.size(0), text_output.projected_tokens.size(1))),
                'exact_token_weights': empty_alpha.new_empty((text_output.projected_pooled.size(0), text_output.projected_tokens.size(1))),
                'surrogate_pooled_text': self._cast_to_prototype_dtype(text_output.projected_pooled),
                'exact_pooled_text': self._cast_to_prototype_dtype(text_output.projected_pooled),
                'surrogate_text_projected': host_outputs['surrogate_text_projected'],
                'surrogate_text_projected_raw': host_outputs['surrogate_text_projected_raw'],
                'exact_text_projected': host_outputs['exact_text_projected'],
                'exact_text_projected_raw': host_outputs['exact_text_projected_raw'],
                'image_projected': host_outputs['image_projected'],
                'image_projected_raw': host_outputs['image_projected_raw'],
                'losses': self._zero_loss_outputs(host_outputs['image_projected']),
                'metrics': {'prototype_branch_disabled': host_outputs['image_projected'].new_tensor(1.0)},
                'debug': {},
                'surrogate_pairwise_logits': None,
            }
        else:
            prototype_outputs = self.prototype_head(
                image_embeddings=self._cast_to_prototype_dtype(image_output.projected_pooled),
                text_token_states=self._cast_to_prototype_dtype(self._resolve_text_states(text_output)),
                token_ids=caption_ids,
                pids=pids,
                attention_mask=text_output.token_mask,
                special_token_positions=text_output.special_token_positions,
                host_pairwise_logits=host_outputs.get('surrogate_pairwise_logits'),
                return_debug=should_return_debug,
                disable_proxy_losses=disable_proxy_losses,
            )

        prototype_losses = prototype_outputs['losses']
        metric_losses = prototype_losses if self.prototype_head is not None else host_losses
        loss_total = (self.lambda_host * host_losses['loss_total']) + prototype_losses['loss_total']
        if not torch.isfinite(loss_total):
            raise FloatingPointError('loss_total contains NaN or Inf values.')

        outputs = {
            'loss_total': loss_total,
            'loss_host': host_losses['loss_total'],
            'loss_host_ret': host_losses['loss_ret'],
            'loss_host_ret_i2t': host_losses['loss_ret_i2t'],
            'loss_host_ret_t2i': host_losses['loss_ret_t2i'],
            'loss_host_cid': host_losses.get('loss_cid', host_losses['loss_total'].new_zeros(())),
            'loss_proto_total': prototype_losses['loss_total'],
            'loss_proto': prototype_losses['loss_total'],
            'loss_proxy': prototype_losses['loss_proxy'],
            'loss_proxy_image': prototype_losses['loss_proxy_image'],
            'loss_proxy_text': prototype_losses['loss_proxy_text'],
            'loss_proxy_text_exact': prototype_losses['loss_proxy_text_exact'],
            'loss_ret': prototype_losses['loss_ret'],
            'loss_align': prototype_losses['loss_align'],
            'loss_dir': prototype_losses['loss_dir'],
            'loss_gap': prototype_losses['loss_gap'],
            'loss_sup': prototype_losses['loss_sup'],
            'loss_diag': prototype_losses['loss_diag'],
            'loss_support': prototype_losses['loss_support'],
            'loss_diversity': prototype_losses['loss_diversity'],
            'loss_balance': prototype_losses['loss_balance'],
            'loss_proxy_image_weighted': prototype_losses['loss_proxy_image_weighted'],
            'loss_proxy_text_weighted': prototype_losses['loss_proxy_text_weighted'],
            'loss_proxy_text_exact_weighted': prototype_losses['loss_proxy_text_exact_weighted'],
            'loss_proxy_weighted': prototype_losses['loss_proxy_weighted'],
            'loss_ret_weighted': prototype_losses['loss_ret_weighted'],
            'loss_weight_ret': prototype_losses['loss_weight_ret'],
            'loss_weight_ret_weighted': prototype_losses['loss_weight_ret_weighted'],
            'loss_align_weighted': prototype_losses['loss_align_weighted'],
            'loss_dir_weighted': prototype_losses['loss_dir_weighted'],
            'loss_gap_weighted': prototype_losses['loss_gap_weighted'],
            'loss_sup_weighted': prototype_losses['loss_sup_weighted'],
            'loss_diag_weighted': prototype_losses['loss_diag_weighted'],
            'loss_support_weighted': prototype_losses['loss_support_weighted'],
            'loss_diversity_weighted': prototype_losses['loss_diversity_weighted'],
            'loss_balance_weighted': prototype_losses['loss_balance_weighted'],
            'loss_host_weighted': self.lambda_host * host_losses['loss_total'],
            'lambda_host': host_losses['loss_total'].new_tensor(self.lambda_host),
            'lambda_proxy': prototype_losses['lambda_proxy'],
            'lambda_proxy_image': prototype_losses['lambda_proxy_image'],
            'lambda_proxy_text': prototype_losses['lambda_proxy_text'],
            'lambda_proxy_text_exact': prototype_losses['lambda_proxy_text_exact'],
            'use_loss_proxy_text_exact': prototype_losses['use_loss_proxy_text_exact'],
            'use_loss_ret': metric_losses['use_loss_ret'],
            'lambda_ret': metric_losses['lambda_ret'],
            'use_loss_weight_ret': prototype_losses['use_loss_weight_ret'],
            'lambda_weight_ret': prototype_losses['lambda_weight_ret'],
            'weight_ret_margin_delta': prototype_losses['weight_ret_margin_delta'],
            'weight_ret_tau': prototype_losses['weight_ret_tau'],
            'weight_ret_detach_host': prototype_losses['weight_ret_detach_host'],
            'weight_ret_normalize_mean_one': prototype_losses['weight_ret_normalize_mean_one'],
            'lambda_align': prototype_losses['lambda_align'],
            'use_loss_dir': prototype_losses['use_loss_dir'],
            'lambda_dir': prototype_losses['lambda_dir'],
            'use_loss_gap': prototype_losses['use_loss_gap'],
            'lambda_gap': prototype_losses['lambda_gap'],
            'use_loss_sup': prototype_losses['use_loss_sup'],
            'lambda_sup': prototype_losses['lambda_sup'],
            'lambda_diag': prototype_losses['lambda_diag'],
            'use_loss_support': prototype_losses['use_loss_support'],
            'lambda_support': prototype_losses['lambda_support'],
            'lambda_div': prototype_losses['lambda_div'],
            'lambda_bal': prototype_losses['lambda_bal'],
            'prototype_gap_margin': prototype_losses['prototype_gap_margin'],
            'prototype_support_target': prototype_losses['prototype_support_target'],
            'proxy_temperature': metric_losses['proxy_temperature'].detach(),
            'diag_temperature': prototype_losses['diag_temperature'].detach(),
            'retrieval_temperature': metric_losses['retrieval_temperature'].detach(),
            'logit_scale': metric_losses['logit_scale'].detach(),
            'host_retrieval_temperature': host_losses['retrieval_temperature'].detach(),
            'host_logit_scale': host_losses['logit_scale'].detach(),
            'fusion_coefficient': host_losses['loss_total'].new_tensor(self.fusion_coefficient),
            'fusion_lambda_host': host_losses['loss_total'].new_tensor(self.fusion_lambda_host),
            'fusion_lambda_prototype': host_losses['loss_total'].new_tensor(self.fusion_lambda_prototype),
            'alpha': prototype_outputs['routing_weights'].detach(),
            'z_v': prototype_outputs.get('image_projected', host_outputs['image_projected']),
            'z_t_hat_diag': prototype_outputs['surrogate_text_projected'],
            'z_t_exact_diag': prototype_outputs['exact_text_projected'],
            'surrogate_pairwise_logits': prototype_outputs.get('surrogate_pairwise_logits'),
            'host_pairwise_logits': host_outputs.get('surrogate_pairwise_logits'),
            'debug': dict(host_outputs.get('metrics', {})),
        }
        outputs['debug'].update(prototype_outputs.get('metrics', {}))
        outputs['debug']['fusion_coefficient'] = self.fusion_coefficient
        outputs['debug']['fusion_lambda_host'] = self.fusion_lambda_host
        outputs['debug']['fusion_lambda_prototype'] = self.fusion_lambda_prototype
        outputs['debug']['host_loss_total'] = host_losses['loss_total'].detach()
        outputs['debug']['host_loss_ret'] = host_losses['loss_ret'].detach()
        outputs['debug']['host_loss_cid'] = host_losses.get('loss_cid', host_losses['loss_total'].new_zeros(())).detach()
        track_output_grads = bool(getattr(self.args, 'log_debug_metrics', True)) or should_return_debug
        if track_output_grads:
            for grad_tensor_key in ('z_v', 'z_t_hat_diag', 'z_t_exact_diag', 'surrogate_pairwise_logits'):
                tensor = outputs.get(grad_tensor_key)
                if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                    tensor.retain_grad()
        if should_return_debug:
            outputs['debug'] = self._build_debug_outputs(image_output, text_output, host_outputs, prototype_outputs)
        return outputs

PrototypeGuidedRetrievalModel = PASModel
Model = PASModel


def build_model(args, num_classes, train_loader=None):
    model = PASModel(args, num_classes=num_classes, train_loader=train_loader)
    if model.backbone_precision == 'fp16':
        convert_weights(model.base_model)
    else:
        model.base_model.float()
    if model.prototype_precision == 'fp16':
        model.host_head.half()
        if model.prototype_head is not None:
            model.prototype_head.half()
    else:
        model.host_head.float()
        if model.prototype_head is not None:
            model.prototype_head.float()
    return model
