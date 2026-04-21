from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .interface_contract import (
    HOST_EXPORT_INTERFACE_SUPPORTED_VERSIONS,
    HOST_EXPORT_INTERFACE_VERSION,
    HostExportPolicy,
    HostPluginInterface,
    build_host_plugin_interface,
)
from .interfaces import EncoderOutput
from .runtime_modes import (
    RUNTIME_MODE_HOST_ONLY,
    RUNTIME_MODE_JOINT_TRAINING,
    normalize_runtime_mode,
    resolve_runtime_mode_from_args,
)
from . import pas_model as legacy_pas
from utils.precision import precision_to_torch_dtype


class HostCore(nn.Module):
    def __init__(
        self,
        *,
        args,
        base_model: nn.Module,
        host_head: nn.Module,
        token_mask_builder: nn.Module,
        host_type: str,
        image_backbone: str,
        text_backbone: str,
        backbone_precision: str,
        prototype_precision: str,
        use_prototype_branch: bool,
        use_image_conditioned_pooling: bool,
        itself_return_all: bool,
        itself_average_attn_weights: bool,
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.host_head = host_head
        self.token_mask_builder = token_mask_builder
        self.host_type = str(host_type).lower()
        self.image_backbone = str(image_backbone)
        self.text_backbone = str(text_backbone)
        self.backbone_precision = str(backbone_precision)
        self.prototype_precision = str(prototype_precision)
        self.use_prototype_branch = bool(use_prototype_branch)
        self.use_image_conditioned_pooling = bool(use_image_conditioned_pooling)
        self.itself_return_all = bool(itself_return_all)
        self.itself_average_attn_weights = bool(itself_average_attn_weights)
        self._warned_projected_text_state_fallback = False

    def _encode_image_intermediates(
        self,
        image: torch.Tensor,
        return_all: bool,
        average_attn_weights: bool,
    ) -> Dict[str, Optional[torch.Tensor]]:
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

    def _encode_text_intermediates(
        self,
        text: torch.Tensor,
        return_all: bool,
        average_attn_weights: bool,
    ) -> Dict[str, Optional[torch.Tensor]]:
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
        text = text.long()
        text_outputs = self._encode_text_intermediates(
            text,
            return_all=self.itself_return_all,
            average_attn_weights=self.itself_average_attn_weights,
        )
        projected_tokens = text_outputs['projected_tokens'].float()
        pre_projection_tokens = text_outputs['pre_projection_tokens']
        if pre_projection_tokens is not None:
            pre_projection_tokens = pre_projection_tokens.float()
        token_mask = self.token_mask_builder.build_valid_mask(text)
        special_positions = self.token_mask_builder.get_special_token_positions(text, attention_mask=token_mask)
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

    def resolve_text_states(self, text_output: EncoderOutput) -> torch.Tensor:
        if text_output.pre_projection_tokens is not None:
            return text_output.pre_projection_tokens
        if self.host_type == 'itself':
            raise RuntimeError(
                'ITSELF host requires text pre-projection token states for prototype routing, but the current '
                'runtime did not expose them.'
            )
        if not self._warned_projected_text_state_fallback:
            logging.getLogger('pas.model').warning(
                'Text pre-projection token states are unavailable; falling back to projected text tokens.'
            )
            self._warned_projected_text_state_fallback = True
        return text_output.projected_tokens

    def encode_image_branch(self, image_output: EncoderOutput, return_debug: bool = False) -> Dict[str, object]:
        return self.host_head.encode_image_branch(
            image_output,
            return_debug=return_debug,
        )

    def encode_text_branch(
        self,
        text_output: EncoderOutput,
        token_ids: torch.Tensor,
        return_debug: bool = False,
        current_step: Optional[int] = None,
    ) -> Dict[str, object]:
        return self.host_head.encode_text_branch(
            text_output,
            token_ids.long(),
            return_debug=return_debug,
            current_step=current_step,
        )

    def compute_similarity_matrix(self, image_features: Dict[str, torch.Tensor], text_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        similarity = self.host_head.compute_similarity_matrix(image_features, text_features)
        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Host retrieval similarity contains NaN or Inf values.')
        return similarity.float()

    def forward_host(
        self,
        *,
        batch: Dict[str, torch.Tensor],
        current_step: Optional[int],
        total_steps: Optional[int],
        return_debug: bool,
    ) -> Tuple[Dict[str, object], EncoderOutput, EncoderOutput]:
        images = batch['images']
        caption_ids = batch['caption_ids']
        pids = batch.get('pids')
        image_output = self.extract_image_features(images)
        text_output = self.extract_text_features(caption_ids)
        host_outputs = self.host_head(
            image_output,
            text_output,
            caption_ids.long(),
            pids=pids,
            return_debug=return_debug,
            current_step=current_step,
            total_steps=total_steps,
        )
        return host_outputs, image_output, text_output

    def build_plugin_interface(
        self,
        *,
        image_output: EncoderOutput,
        text_output: EncoderOutput,
        token_ids: torch.Tensor,
        host_pairwise_logits: Optional[torch.Tensor],
        policy: HostExportPolicy,
        metadata: Optional[Dict[str, object]] = None,
    ) -> HostPluginInterface:
        return build_host_plugin_interface(
            image_embeddings=image_output.projected_pooled,
            text_token_states=self.resolve_text_states(text_output),
            token_ids=token_ids.long(),
            attention_mask=text_output.token_mask,
            special_token_positions=text_output.special_token_positions,
            image_local_tokens=image_output.projected_tokens,
            host_pairwise_logits=host_pairwise_logits,
            policy=policy,
            metadata=metadata,
        )


class PrototypePlugin(nn.Module):
    def __init__(
        self,
        *,
        prototype_head: nn.Module,
        prototype_precision: str,
        eval_image_chunk_size: int,
        eval_text_chunk_size: int,
        use_prototype_bank: bool,
        accepted_host_interface_versions=None,
    ):
        super().__init__()
        self.prototype_head = prototype_head
        self.prototype_precision = str(prototype_precision)
        self.eval_image_chunk_size = int(eval_image_chunk_size)
        self.eval_text_chunk_size = int(eval_text_chunk_size)
        self.use_prototype_bank = bool(use_prototype_bank)
        accepted = accepted_host_interface_versions
        if accepted is None:
            accepted = HOST_EXPORT_INTERFACE_SUPPORTED_VERSIONS
        self.accepted_host_interface_versions = {str(item) for item in accepted}
        self.schema_version = 'prototype_plugin_v1'

    def prototype_dtype(self) -> torch.dtype:
        return precision_to_torch_dtype(self.prototype_precision)

    def cast(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=self.prototype_dtype())

    def forward_from_interface(
        self,
        *,
        interface: HostPluginInterface,
        pids: Optional[torch.Tensor],
        epoch: Optional[int],
        current_step: Optional[int],
        return_debug: bool,
        disable_proxy_losses: bool,
    ) -> Dict[str, object]:
        interface.validate()
        if str(interface.version) not in self.accepted_host_interface_versions:
            raise ValueError(
                f'PrototypePlugin interface version mismatch: got {interface.version!r}, '
                f'expected one of {sorted(self.accepted_host_interface_versions)!r}.'
            )
        return self.prototype_head(
            image_embeddings=self.cast(interface.image_embeddings),
            image_local_tokens=None if interface.image_local_tokens is None else self.cast(interface.image_local_tokens),
            text_token_states=self.cast(interface.text_token_states),
            token_ids=interface.token_ids.long(),
            pids=pids,
            attention_mask=interface.attention_mask,
            special_token_positions=interface.special_token_positions,
            host_pairwise_logits=interface.host_pairwise_logits,
            epoch=epoch,
            current_step=current_step,
            return_debug=return_debug,
            disable_proxy_losses=disable_proxy_losses,
        )

    def encode_image_branch(self, *, image_embeddings: torch.Tensor, image_local_tokens: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.prototype_head.encode_image_branch(
            self.cast(image_embeddings),
            image_local_tokens=None if image_local_tokens is None else self.cast(image_local_tokens),
            return_debug=False,
        )

    def build_text_basis_bank(
        self,
        *,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        special_token_positions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        context = self.prototype_head.get_prototype_context(return_debug=False)
        return self.prototype_head.build_text_basis_bank(
            text_token_states=self.cast(text_token_states),
            token_ids=token_ids.long(),
            contextualized_prototypes=self.cast(context.get('routing_prototypes', context['contextualized_prototypes'])),
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
            return_debug=False,
        )

    def compute_exact_similarity(
        self,
        *,
        image_projected: torch.Tensor,
        summaries: torch.Tensor,
        text_token_states: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        special_token_positions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.prototype_head.compute_pairwise_similarity(
            image_projected=self.cast(image_projected),
            summaries=self.cast(summaries),
            text_token_states=self.cast(text_token_states),
            token_ids=token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
            image_chunk_size=self.eval_image_chunk_size,
            text_chunk_size=self.eval_text_chunk_size,
        ).float()

    def compute_approximate_similarity(
        self,
        *,
        image_projected: torch.Tensor,
        routing_weights: torch.Tensor,
        basis_bank: torch.Tensor,
    ) -> torch.Tensor:
        return self.prototype_head.compute_approximate_pairwise_similarity(
            image_projected=self.cast(image_projected),
            routing_weights=self.cast(routing_weights),
            basis_bank=self.cast(basis_bank),
            image_chunk_size=self.eval_image_chunk_size,
            text_chunk_size=self.eval_text_chunk_size,
        ).float()


class PASRuntimeModel(nn.Module):
    def __init__(self, args, num_classes, train_loader=None):
        super().__init__()
        self.args = args
        self.num_classes = int(num_classes)
        legacy_model = legacy_pas.build_model(args=args, num_classes=num_classes, train_loader=train_loader)

        self.host_type = str(getattr(legacy_model, 'host_type', getattr(args, 'host_type', 'clip'))).lower()
        self.lambda_host = float(getattr(legacy_model, 'lambda_host', getattr(args, 'lambda_host', 1.0)))
        self.use_host_loss = bool(getattr(legacy_model, 'use_host_loss', getattr(args, 'use_host_loss', True)))
        self.use_prototype_branch = bool(getattr(legacy_model, 'use_prototype_branch', getattr(args, 'use_prototype_branch', False)))
        self.use_prototype_bank = bool(getattr(legacy_model, 'use_prototype_bank', getattr(args, 'use_prototype_bank', False)))
        self.use_image_conditioned_pooling = bool(
            getattr(legacy_model, 'use_image_conditioned_pooling', getattr(args, 'use_image_conditioned_pooling', False))
        )
        self.return_debug_outputs = bool(getattr(legacy_model, 'return_debug_outputs', getattr(args, 'return_debug_outputs', False)))
        self.prototype_dim = int(getattr(legacy_model, 'prototype_dim', getattr(args, 'prototype_dim', 0) or 0))
        self.prototype_method_role = str(getattr(args, 'prototype_method_role', getattr(legacy_model, 'prototype_method_role', 'semantic_structure'))).lower()
        self.prototype_semantic_enabled = bool(
            getattr(args, 'prototype_semantic_enabled', getattr(legacy_model, 'prototype_semantic_enabled', self.prototype_method_role == 'semantic_structure'))
        )
        self.semantic_structure_enabled = bool(
            getattr(args, 'semantic_structure_enabled', getattr(legacy_model, 'semantic_structure_enabled', self.prototype_semantic_enabled))
        )
        self.semantic_ramp_use_prototype = bool(getattr(args, 'semantic_ramp_use_prototype', False))
        self.semantic_recompute_start_epoch = max(int(getattr(args, 'semantic_recompute_start_epoch', 0)), 0)
        self.semantic_recompute_start_step = max(int(getattr(args, 'semantic_recompute_start_step', 0)), 0)
        self.prototype_inference_mode = 'host_only'
        self.host_export_interface_version = HOST_EXPORT_INTERFACE_VERSION
        self.host_component_schema_version = 'host_core_v1'

        self.host_core = HostCore(
            args=args,
            base_model=legacy_model.base_model,
            host_head=legacy_model.host_head,
            token_mask_builder=legacy_model.token_mask_builder,
            host_type=self.host_type,
            image_backbone=getattr(legacy_model, 'image_backbone', getattr(args, 'image_backbone', args.pretrain_choice)),
            text_backbone=getattr(legacy_model, 'text_backbone', getattr(args, 'text_backbone', 'clip_text_transformer')),
            backbone_precision=str(getattr(legacy_model, 'backbone_precision', getattr(args, 'backbone_precision', 'fp16'))),
            prototype_precision=str(getattr(legacy_model, 'prototype_precision', getattr(args, 'prototype_precision', 'fp32'))),
            use_prototype_branch=self.use_prototype_branch,
            use_image_conditioned_pooling=self.use_image_conditioned_pooling,
            itself_return_all=bool(getattr(legacy_model, 'itself_return_all', False)),
            itself_average_attn_weights=bool(getattr(legacy_model, 'itself_average_attn_weights', True)),
        )
        self.prototype_plugin = None
        if getattr(legacy_model, 'prototype_head', None) is not None:
            self.prototype_plugin = PrototypePlugin(
                prototype_head=legacy_model.prototype_head,
                prototype_precision=str(getattr(legacy_model, 'prototype_precision', getattr(args, 'prototype_precision', 'fp32'))),
                eval_image_chunk_size=int(getattr(legacy_model, 'prototype_eval_image_chunk_size', getattr(args, 'prototype_eval_image_chunk_size', 32))),
                eval_text_chunk_size=int(getattr(legacy_model, 'prototype_eval_text_chunk_size', getattr(args, 'prototype_eval_text_chunk_size', 128))),
                use_prototype_bank=self.use_prototype_bank,
                accepted_host_interface_versions=HOST_EXPORT_INTERFACE_SUPPORTED_VERSIONS,
            )

        initial_mode = resolve_runtime_mode_from_args(args, for_training=bool(getattr(args, 'training', True)))
        self.runtime_mode = normalize_runtime_mode(initial_mode)
        if self.runtime_mode == RUNTIME_MODE_HOST_ONLY and self.prototype_plugin is not None:
            logging.getLogger('pas.model').info('Runtime mode host_only selected: prototype plugin remains attached but inactive.')
        if self.runtime_mode != RUNTIME_MODE_HOST_ONLY and self.prototype_plugin is None:
            raise ValueError(
                f'runtime_mode={self.runtime_mode!r} requires an active prototype branch, '
                'but prototype modules are disabled.'
            )
        self.set_runtime_mode(self.runtime_mode)

    @property
    def base_model(self) -> nn.Module:
        return self.host_core.base_model

    @property
    def host_head(self) -> nn.Module:
        return self.host_core.host_head

    @property
    def prototype_head(self) -> Optional[nn.Module]:
        if self.prototype_plugin is None:
            return None
        return self.prototype_plugin.prototype_head

    def set_runtime_mode(self, mode: str) -> None:
        normalized = normalize_runtime_mode(mode)
        if normalized != RUNTIME_MODE_HOST_ONLY and self.prototype_plugin is None:
            raise ValueError(
                f'Cannot switch runtime_mode to {normalized!r} because prototype branch is unavailable.'
            )
        self.runtime_mode = normalized

    @staticmethod
    def _compute_pairwise_logits_from_outputs(outputs: Dict[str, object], *, output_name: str) -> torch.Tensor:
        pairwise_logits = outputs.get('surrogate_pairwise_logits')
        if isinstance(pairwise_logits, torch.Tensor):
            return pairwise_logits.float()
        image_projected = outputs.get('image_projected')
        text_projected = outputs.get('surrogate_text_projected', outputs.get('exact_text_projected'))
        if isinstance(image_projected, torch.Tensor) and isinstance(text_projected, torch.Tensor):
            return (image_projected.float() @ text_projected.float().t()).float()
        raise RuntimeError(
            f'{output_name} must provide surrogate_pairwise_logits or '
            'image_projected/surrogate_text_projected tensors.'
        )

    def get_group_checkpoint_compatibility(self, group_name: str) -> Dict[str, object]:
        group = str(group_name)
        all_modes = (
            RUNTIME_MODE_HOST_ONLY,
            RUNTIME_MODE_JOINT_TRAINING,
        )
        common = {
            'runtime_mode': str(self.runtime_mode),
            'host_export_interface_version': self.host_export_interface_version,
        }
        if group == 'host':
            return {
                **common,
                'component_name': 'HostCore',
                'component_schema_version': self.host_component_schema_version,
                'compatible_runtime_modes': list(all_modes),
            }
        if group in {'prototype_bank', 'prototype_projector'}:
            return {
                **common,
                'component_name': 'PrototypePlugin',
                'component_schema_version': self.prototype_plugin.schema_version if self.prototype_plugin is not None else 'prototype_plugin_unavailable',
                'accepted_host_interface_versions': sorted(
                    self.prototype_plugin.accepted_host_interface_versions if self.prototype_plugin is not None else []
                ),
                'compatible_runtime_modes': [RUNTIME_MODE_JOINT_TRAINING],
            }
        return dict(common)

    def _current_runtime_mode(self) -> str:
        mode = normalize_runtime_mode(self.runtime_mode)
        if mode == RUNTIME_MODE_JOINT_TRAINING and not self.training:
            return RUNTIME_MODE_HOST_ONLY
        return mode

    def _semantic_schedule_started(self, *, epoch: Optional[int], current_step: Optional[int]) -> bool:
        if epoch is not None and int(epoch) < int(self.semantic_recompute_start_epoch):
            return False
        if current_step is not None and int(current_step) < int(self.semantic_recompute_start_step):
            return False
        return True

    def _prototype_branch_active_for_step(
        self,
        *,
        mode: str,
        epoch: Optional[int],
        current_step: Optional[int],
    ) -> bool:
        if mode != RUNTIME_MODE_JOINT_TRAINING or self.prototype_plugin is None:
            return False
        if not self.semantic_ramp_use_prototype:
            return True
        return self._semantic_schedule_started(epoch=epoch, current_step=current_step)

    def _policy_for_runtime_mode(self, mode: str) -> HostExportPolicy:
        if mode == RUNTIME_MODE_JOINT_TRAINING:
            return HostExportPolicy(
                detach=False,
                allow_host_pairwise_logits=True,
                include_image_local_tokens=True,
            )
        return HostExportPolicy(
            detach=True,
            allow_host_pairwise_logits=False,
            include_image_local_tokens=False,
        )

    def _zero_loss_outputs(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        return {
            'loss_total': zero,
            'loss_proto': zero,
            'loss_semantic_pbt': zero,
            'loss_semantic_hardneg_margin': zero,
            'loss_semantic_hardneg_margin_image': zero,
            'loss_semantic_hardneg_margin_text': zero,
            'loss_semantic_hosthard_weighted': zero,
            'loss_semantic_hosthard_weighted_image': zero,
            'loss_semantic_hosthard_weighted_text': zero,
            'loss_diag': zero,
            'loss_diversity': zero,
            'loss_balance': zero,
            'loss_semantic_pbt_weighted': zero,
            'loss_semantic_hardneg_margin_weighted': zero,
            'loss_semantic_hosthard_weighted_weighted': zero,
            'loss_diag_weighted': zero,
            'loss_diversity_weighted': zero,
            'loss_balance_weighted': zero,
            'use_loss_semantic_pbt': zero,
            'lambda_semantic_pbt': zero,
            'use_loss_semantic_hardneg_margin': zero,
            'lambda_semantic_hardneg_margin': zero,
            'semantic_hardneg_margin': zero,
            'semantic_hardneg_eps': zero,
            'use_loss_semantic_hosthard_weighted': zero,
            'lambda_semantic_hosthard_weighted': zero,
            'semantic_hosthard_margin_ref': zero,
            'semantic_hosthard_tau': zero,
            'semantic_hosthard_eps': zero,
            'semantic_hosthard_normalize_weights': zero,
            'prototype_loss_scale': zero,
            'prototype_loss_ramp_scale': zero,
            'loss_diag_scale': zero,
            'loss_semantic_pbt_scale': zero,
            'loss_semantic_hosthard_weighted_scale': zero,
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

    def _empty_prototype_outputs(
        self,
        *,
        host_outputs: Dict[str, object],
        text_output: EncoderOutput,
        image_output: EncoderOutput,
    ) -> Dict[str, object]:
        empty_alpha = host_outputs['routing_weights']
        empty_bank = empty_alpha.new_empty((text_output.projected_pooled.size(0), 0, self.prototype_dim))
        return {
            'routing_weights': empty_alpha,
            'summary': image_output.projected_pooled,
            'prototypes': empty_alpha.new_empty((0, self.prototype_dim)),
            'contextualized_prototypes': empty_alpha.new_empty((0, self.prototype_dim)),
            'basis_bank': empty_bank,
            'token_valid_mask': text_output.token_mask,
            'token_keep_mask': text_output.token_mask,
            'beta_logits_masked': empty_alpha.new_empty((text_output.projected_pooled.size(0), text_output.projected_tokens.size(1))),
            'exact_token_weights': empty_alpha.new_empty((text_output.projected_pooled.size(0), text_output.projected_tokens.size(1))),
            'surrogate_pooled_text': text_output.projected_pooled,
            'exact_pooled_text': text_output.projected_pooled,
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

    def encode_image_for_retrieval(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_output = self.host_core.extract_image_features(image)
        host_image = self.host_core.encode_image_branch(image_output, return_debug=False)
        summary = host_image['summary']
        routing_weights = host_image['routing_weights']
        outputs = dict(host_image)
        outputs.update(
            {
                'host_image_projected': host_image['image_projected'],
                'host_summary': host_image['summary'],
                'summary': summary,
                'routing_weights': routing_weights,
            }
        )
        if self.prototype_plugin is not None:
            prototype_image = self.prototype_plugin.encode_image_branch(
                image_embeddings=image_output.projected_pooled,
                image_local_tokens=image_output.projected_tokens,
            )
            outputs['summary'] = prototype_image['summary']
            outputs['routing_weights'] = prototype_image['routing_weights']
            outputs['prototype_image_projected'] = prototype_image['image_projected']
            outputs['prototype_summary'] = prototype_image['summary']
        return outputs

    def encode_text_for_retrieval(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_output = self.host_core.extract_text_features(text)
        host_text = self.host_core.encode_text_branch(text_output, text.long(), return_debug=False)
        outputs = dict(host_text)
        outputs['host_text_projected'] = host_text['text_projected']
        if self.prototype_plugin is not None:
            outputs.update(
                {
                    'text_token_states': self.host_core.resolve_text_states(text_output),
                    'token_ids': text.long(),
                    'attention_mask': text_output.token_mask,
                    'special_token_positions': {key: value for key, value in text_output.special_token_positions.items()},
                }
            )
        return outputs

    def encode_text_basis_for_retrieval(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise RuntimeError(
            'encode_text_basis_for_retrieval is removed. Retrieval scoring is host-only exact.'
        )

    def _semantic_inference_active(self) -> bool:
        return bool(
            self.prototype_method_role == 'semantic_structure'
            and self.prototype_semantic_enabled
            and self.semantic_structure_enabled
        )

    def compute_retrieval_similarity(self, image_features: Dict[str, torch.Tensor], text_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.host_core.compute_similarity_matrix(image_features, text_features).float()

    def compute_approximate_retrieval_similarity_components(
        self,
        image_features: Dict[str, torch.Tensor],
        text_basis_features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError('Approximate prototype retrieval scoring is removed. Retrieval is host-only exact.')

    def compute_approximate_retrieval_similarity(self, image_features: Dict[str, torch.Tensor], text_basis_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise RuntimeError('Approximate prototype retrieval scoring is removed. Retrieval is host-only exact.')

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
            if name.startswith('prototype_plugin.prototype_head.losses.class_proxies'):
                groups['class_proxies'].append((name, parameter))
            elif name.startswith('prototype_plugin.prototype_head.prototype_bank'):
                groups['prototype_bank'].append((name, parameter))
            elif name.startswith('prototype_plugin.prototype_head.contextualizer'):
                groups['prototype_contextualization'].append((name, parameter))
            elif name.startswith('prototype_plugin.prototype_head.router') or name.startswith('prototype_plugin.prototype_head.local_routing_adapter'):
                groups['prototype_routing'].append((name, parameter))
            elif (
                name.startswith('prototype_plugin.prototype_head.text_pool_query')
                or name.startswith('prototype_plugin.prototype_head.token_pooler')
                or name.startswith('prototype_plugin.prototype_head.token_scorer')
                or name.startswith('prototype_plugin.prototype_head.token_mask_builder')
            ):
                groups['prototype_pooling'].append((name, parameter))
            elif (
                name.startswith('prototype_plugin.prototype_head.image_projector')
                or name.startswith('prototype_plugin.prototype_head.text_projector')
                or name.startswith('prototype_plugin.prototype_head.proto_query_proj')
                or name.startswith('prototype_plugin.prototype_head.image_adapter')
                or name.startswith('prototype_plugin.prototype_head.text_adapter')
            ):
                groups['prototype_projectors'].append((name, parameter))
            elif name.startswith('host_core.host_head'):
                groups['host_projectors'].append((name, parameter))
            elif name.startswith('host_core.base_model.visual'):
                groups['image_backbone'].append((name, parameter))
            elif (
                name.startswith('host_core.base_model.transformer')
                or name.startswith('host_core.base_model.token_embedding')
                or name.startswith('host_core.base_model.positional_embedding')
                or name.startswith('host_core.base_model.ln_final')
                or name.startswith('host_core.base_model.text_projection')
            ):
                groups['text_backbone'].append((name, parameter))
            else:
                groups['other'].append((name, parameter))
        return groups

    def forward(
        self,
        batch,
        epoch=None,
        current_step=None,
        total_steps=None,
        return_debug: Optional[bool] = None,
        disable_proxy_losses: bool = False,
    ):
        mode = self._current_runtime_mode()
        pids = batch.get('pids')
        should_return_debug = self.return_debug_outputs if return_debug is None else bool(return_debug)
        host_outputs, image_output, text_output = self.host_core.forward_host(
            batch=batch,
            current_step=current_step,
            total_steps=total_steps,
            return_debug=should_return_debug,
        )
        host_losses = host_outputs['losses']
        prototype_outputs = None
        prototype_active = self._prototype_branch_active_for_step(
            mode=mode,
            epoch=epoch,
            current_step=current_step,
        )
        if not prototype_active:
            prototype_outputs = self._empty_prototype_outputs(
                host_outputs=host_outputs,
                text_output=text_output,
                image_output=image_output,
            )
        else:
            policy = self._policy_for_runtime_mode(mode)
            host_pairwise_logits_for_plugin = (
                host_outputs.get('surrogate_pairwise_logits') if policy.allow_host_pairwise_logits else None
            )
            if isinstance(host_pairwise_logits_for_plugin, torch.Tensor) and self.host_type == 'itself':
                # ITSELF host similarity is returned as [text, image]; semantic hard-neg mining expects [image, text].
                host_pairwise_logits_for_plugin = host_pairwise_logits_for_plugin.t().contiguous()
            interface = self.host_core.build_plugin_interface(
                image_output=image_output,
                text_output=text_output,
                token_ids=batch['caption_ids'],
                host_pairwise_logits=host_pairwise_logits_for_plugin,
                policy=policy,
                metadata={
                    'runtime_mode': mode,
                    'host_type': self.host_type,
                    'policy_detach': bool(policy.detach),
                    'allow_host_pairwise_logits': bool(policy.allow_host_pairwise_logits),
                },
            )
            prototype_outputs = self.prototype_plugin.forward_from_interface(
                interface=interface,
                pids=pids,
                epoch=epoch,
                current_step=current_step,
                return_debug=should_return_debug,
                disable_proxy_losses=disable_proxy_losses,
            )

        prototype_losses = prototype_outputs['losses']
        metric_losses = prototype_losses if prototype_active else host_losses
        if mode == RUNTIME_MODE_JOINT_TRAINING:
            loss_total = (self.lambda_host * host_losses['loss_total']) + prototype_losses['loss_total']
        elif mode == RUNTIME_MODE_HOST_ONLY:
            loss_total = self.lambda_host * host_losses['loss_total']
        else:
            raise ValueError(f'Unsupported runtime mode after refactor: {mode!r}')
        if not torch.isfinite(loss_total):
            raise FloatingPointError('loss_total contains NaN or Inf values.')

        metric_zero = host_losses['loss_total'].new_zeros(())
        metric_proxy_temperature = metric_losses.get('proxy_temperature', metric_zero)
        metric_retrieval_temperature = metric_losses.get(
            'retrieval_temperature',
            host_losses.get('retrieval_temperature', metric_zero),
        )
        metric_logit_scale = metric_losses.get(
            'logit_scale',
            host_losses.get('logit_scale', metric_zero),
        )

        outputs = {
            'loss_total': loss_total,
            'loss_host': host_losses['loss_total'],
            'loss_host_ret': host_losses['loss_ret'],
            'loss_host_ret_i2t': host_losses['loss_ret_i2t'],
            'loss_host_ret_t2i': host_losses['loss_ret_t2i'],
            'loss_host_cid': host_losses.get('loss_cid', host_losses['loss_total'].new_zeros(())),
            'loss_proto_total': prototype_losses['loss_total'],
            'loss_proto': prototype_losses['loss_total'],
            'loss_semantic_pbt': prototype_losses.get('loss_semantic_pbt', metric_zero),
            'loss_semantic_hardneg_margin': prototype_losses.get('loss_semantic_hardneg_margin', metric_zero),
            'loss_semantic_hardneg_margin_image': prototype_losses.get('loss_semantic_hardneg_margin_image', metric_zero),
            'loss_semantic_hardneg_margin_text': prototype_losses.get('loss_semantic_hardneg_margin_text', metric_zero),
            'loss_semantic_hosthard_weighted': prototype_losses.get('loss_semantic_hosthard_weighted', metric_zero),
            'loss_semantic_hosthard_weighted_image': prototype_losses.get('loss_semantic_hosthard_weighted_image', metric_zero),
            'loss_semantic_hosthard_weighted_text': prototype_losses.get('loss_semantic_hosthard_weighted_text', metric_zero),
            'loss_diag': prototype_losses['loss_diag'],
            'loss_diversity': prototype_losses['loss_diversity'],
            'loss_balance': prototype_losses['loss_balance'],
            'loss_semantic_pbt_weighted': prototype_losses.get('loss_semantic_pbt_weighted', metric_zero),
            'loss_semantic_hardneg_margin_weighted': prototype_losses.get('loss_semantic_hardneg_margin_weighted', metric_zero),
            'loss_semantic_hosthard_weighted_weighted': prototype_losses.get(
                'loss_semantic_hosthard_weighted_weighted',
                metric_zero,
            ),
            'loss_diag_weighted': prototype_losses['loss_diag_weighted'],
            'loss_diversity_weighted': prototype_losses['loss_diversity_weighted'],
            'loss_balance_weighted': prototype_losses['loss_balance_weighted'],
            'loss_host_weighted': self.lambda_host * host_losses['loss_total'],
            'lambda_host': host_losses['loss_total'].new_tensor(self.lambda_host),
            'use_loss_semantic_pbt': prototype_losses.get('use_loss_semantic_pbt', metric_zero),
            'lambda_semantic_pbt': prototype_losses.get('lambda_semantic_pbt', metric_zero),
            'use_loss_semantic_hardneg_margin': prototype_losses.get('use_loss_semantic_hardneg_margin', metric_zero),
            'lambda_semantic_hardneg_margin': prototype_losses.get('lambda_semantic_hardneg_margin', metric_zero),
            'semantic_hardneg_margin': prototype_losses.get('semantic_hardneg_margin', metric_zero),
            'semantic_hardneg_eps': prototype_losses.get('semantic_hardneg_eps', metric_zero),
            'use_loss_semantic_hosthard_weighted': prototype_losses.get(
                'use_loss_semantic_hosthard_weighted',
                metric_zero,
            ),
            'lambda_semantic_hosthard_weighted': prototype_losses.get(
                'lambda_semantic_hosthard_weighted',
                metric_zero,
            ),
            'semantic_hosthard_margin_ref': prototype_losses.get('semantic_hosthard_margin_ref', metric_zero),
            'semantic_hosthard_tau': prototype_losses.get('semantic_hosthard_tau', metric_zero),
            'semantic_hosthard_eps': prototype_losses.get('semantic_hosthard_eps', metric_zero),
            'semantic_hosthard_normalize_weights': prototype_losses.get(
                'semantic_hosthard_normalize_weights',
                metric_zero,
            ),
            'prototype_loss_scale': prototype_losses.get(
                'prototype_loss_scale',
                prototype_losses.get('semantic_loss_scale', metric_zero),
            ),
            'prototype_loss_ramp_scale': prototype_losses.get(
                'prototype_loss_ramp_scale',
                prototype_losses.get('prototype_loss_scale', metric_zero),
            ),
            'loss_diag_scale': prototype_losses.get('loss_diag_scale', metric_zero.new_ones(())),
            'loss_semantic_pbt_scale': prototype_losses.get(
                'loss_semantic_pbt_scale',
                prototype_losses.get('semantic_loss_scale', metric_zero.new_ones(())),
            ),
            'loss_semantic_hosthard_weighted_scale': prototype_losses.get(
                'loss_semantic_hosthard_weighted_scale',
                prototype_losses.get('semantic_loss_scale', metric_zero.new_ones(())),
            ),
            'semantic_loss_scale': prototype_losses.get('semantic_loss_scale', metric_zero),
            'lambda_diag': prototype_losses['lambda_diag'],
            'lambda_div': prototype_losses['lambda_div'],
            'lambda_bal': prototype_losses['lambda_bal'],
            'proxy_temperature': metric_proxy_temperature.detach(),
            'diag_temperature': prototype_losses['diag_temperature'].detach(),
            'retrieval_temperature': metric_retrieval_temperature.detach(),
            'logit_scale': metric_logit_scale.detach(),
            'host_retrieval_temperature': host_losses['retrieval_temperature'].detach(),
            'host_logit_scale': host_losses['logit_scale'].detach(),
            'alpha': prototype_outputs['routing_weights'].detach(),
            'z_v': prototype_outputs.get('image_projected', host_outputs['image_projected']),
            'z_t_hat_diag': prototype_outputs['surrogate_text_projected'],
            'z_t_exact_diag': prototype_outputs['exact_text_projected'],
            'surrogate_pairwise_logits': prototype_outputs.get('surrogate_pairwise_logits'),
            'host_pairwise_logits': host_outputs.get('surrogate_pairwise_logits'),
            'debug': dict(host_outputs.get('metrics', {})),
        }
        outputs['debug'].update(prototype_outputs.get('metrics', {}))
        outputs['debug']['runtime_mode'] = mode
        outputs['debug']['prototype_branch_active'] = float(prototype_active)
        outputs['debug']['prototype_method_role_semantic_structure'] = float(self.prototype_method_role == 'semantic_structure')
        outputs['debug']['prototype_semantic_enabled'] = float(self.prototype_semantic_enabled)
        outputs['debug']['semantic_structure_enabled'] = float(self.semantic_structure_enabled)
        outputs['debug']['prototype_inference_host_only'] = float(self.prototype_inference_mode == 'host_only')
        outputs['debug']['host_loss_total'] = host_losses['loss_total'].detach()
        outputs['debug']['host_loss_ret'] = host_losses['loss_ret'].detach()
        outputs['debug']['host_loss_cid'] = host_losses.get('loss_cid', host_losses['loss_total'].new_zeros(())).detach()
        track_output_grads = bool(getattr(self.args, 'log_debug_metrics', True)) or should_return_debug
        if track_output_grads:
            for grad_tensor_key in ('z_v', 'z_t_hat_diag', 'z_t_exact_diag', 'surrogate_pairwise_logits'):
                tensor = outputs.get(grad_tensor_key)
                if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                    tensor.retain_grad()
        return outputs


def build_structural_split_model(args, num_classes, train_loader=None):
    return PASRuntimeModel(args=args, num_classes=num_classes, train_loader=train_loader)
