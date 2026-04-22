from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import torch


HOST_EXPORT_INTERFACE_VERSION = 'host_export_v1'
HOST_EXPORT_INTERFACE_SUPPORTED_VERSIONS: Set[str] = {HOST_EXPORT_INTERFACE_VERSION}
HOST_SCORE_SCHEMA_VERSION = 'host_score_v1'


@dataclass(frozen=True)
class HostExportPolicy:
    detach: bool = True
    allow_host_pairwise_logits: bool = False
    include_image_local_tokens: bool = True


@dataclass(frozen=True)
class HostPluginInterface:
    image_embeddings: torch.Tensor
    text_token_states: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    special_token_positions: Dict[str, torch.Tensor]
    image_local_tokens: Optional[torch.Tensor] = None
    host_pairwise_logits: Optional[torch.Tensor] = None
    host_pairwise_logits_global: Optional[torch.Tensor] = None
    host_pairwise_logits_local: Optional[torch.Tensor] = None
    version: str = HOST_EXPORT_INTERFACE_VERSION
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.image_embeddings, torch.Tensor) or self.image_embeddings.ndim != 2:
            raise ValueError('HostPluginInterface.image_embeddings must have shape [B, D].')
        if not isinstance(self.text_token_states, torch.Tensor) or self.text_token_states.ndim != 3:
            raise ValueError('HostPluginInterface.text_token_states must have shape [B, L, D].')
        if not isinstance(self.token_ids, torch.Tensor) or self.token_ids.ndim != 2:
            raise ValueError('HostPluginInterface.token_ids must have shape [B, L].')
        if self.text_token_states.size(0) != self.token_ids.size(0):
            raise ValueError('HostPluginInterface text batch dimensions must match.')
        if self.image_embeddings.size(0) != self.token_ids.size(0):
            raise ValueError('HostPluginInterface image/text batch dimensions must match.')
        if self.attention_mask is not None:
            if not isinstance(self.attention_mask, torch.Tensor) or self.attention_mask.ndim != 2:
                raise ValueError('HostPluginInterface.attention_mask must have shape [B, L] when provided.')
            if self.attention_mask.shape != self.token_ids.shape:
                raise ValueError('HostPluginInterface attention_mask must match token_ids shape.')
        if self.image_local_tokens is not None:
            if not isinstance(self.image_local_tokens, torch.Tensor) or self.image_local_tokens.ndim != 3:
                raise ValueError('HostPluginInterface.image_local_tokens must have shape [B, T, D] when provided.')
            if self.image_local_tokens.size(0) != self.image_embeddings.size(0):
                raise ValueError('HostPluginInterface.image_local_tokens batch dimension must match image_embeddings.')
        batch_size = self.image_embeddings.size(0)
        for field_name in (
            'host_pairwise_logits',
            'host_pairwise_logits_global',
            'host_pairwise_logits_local',
        ):
            field_value = getattr(self, field_name)
            if field_value is None:
                continue
            if not isinstance(field_value, torch.Tensor) or field_value.ndim != 2:
                raise ValueError(f'HostPluginInterface.{field_name} must have shape [B, B] when provided.')
            if tuple(field_value.shape) != (batch_size, batch_size):
                raise ValueError(
                    f'HostPluginInterface.{field_name} must have shape [B, B] where B matches image/text batch.'
                )
        version = str(self.version).strip()
        if version not in HOST_EXPORT_INTERFACE_SUPPORTED_VERSIONS:
            raise ValueError(
                f'Unsupported HostPluginInterface.version={self.version!r}. '
                f'Expected one of {sorted(HOST_EXPORT_INTERFACE_SUPPORTED_VERSIONS)!r}.'
            )
        if 'eos' not in self.special_token_positions:
            raise ValueError('HostPluginInterface.special_token_positions must include `eos`.')

    def detached_copy(self) -> 'HostPluginInterface':
        return HostPluginInterface(
            image_embeddings=self.image_embeddings.detach(),
            text_token_states=self.text_token_states.detach(),
            token_ids=self.token_ids.detach(),
            attention_mask=None if self.attention_mask is None else self.attention_mask.detach(),
            special_token_positions={key: value.detach() for key, value in self.special_token_positions.items()},
            image_local_tokens=None if self.image_local_tokens is None else self.image_local_tokens.detach(),
            host_pairwise_logits=None if self.host_pairwise_logits is None else self.host_pairwise_logits.detach(),
            host_pairwise_logits_global=(
                None if self.host_pairwise_logits_global is None else self.host_pairwise_logits_global.detach()
            ),
            host_pairwise_logits_local=(
                None if self.host_pairwise_logits_local is None else self.host_pairwise_logits_local.detach()
            ),
            version=self.version,
            metadata=dict(self.metadata),
        )


def build_host_plugin_interface(
    *,
    image_embeddings: torch.Tensor,
    text_token_states: torch.Tensor,
    token_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    special_token_positions: Dict[str, torch.Tensor],
    image_local_tokens: Optional[torch.Tensor],
    host_pairwise_logits: Optional[torch.Tensor],
    host_pairwise_logits_global: Optional[torch.Tensor],
    host_pairwise_logits_local: Optional[torch.Tensor],
    policy: HostExportPolicy,
    metadata: Optional[Dict[str, object]] = None,
) -> HostPluginInterface:
    if (
        (not bool(policy.allow_host_pairwise_logits))
        and (
            host_pairwise_logits is not None
            or host_pairwise_logits_global is not None
            or host_pairwise_logits_local is not None
        )
    ):
        raise ValueError(
            'HostExportPolicy forbids exporting host_pairwise_logits in this mode, '
            'but host pairwise logits were provided.'
        )
    exported_logits = host_pairwise_logits if bool(policy.allow_host_pairwise_logits) else None
    exported_logits_global = host_pairwise_logits_global if bool(policy.allow_host_pairwise_logits) else None
    exported_logits_local = host_pairwise_logits_local if bool(policy.allow_host_pairwise_logits) else None
    exported_local_tokens = image_local_tokens if bool(policy.include_image_local_tokens) else None
    interface = HostPluginInterface(
        image_embeddings=image_embeddings,
        text_token_states=text_token_states,
        token_ids=token_ids,
        attention_mask=attention_mask,
        special_token_positions=dict(special_token_positions),
        image_local_tokens=exported_local_tokens,
        host_pairwise_logits=exported_logits,
        host_pairwise_logits_global=exported_logits_global,
        host_pairwise_logits_local=exported_logits_local,
        metadata=dict(metadata or {}),
    )
    interface.validate()
    if bool(policy.detach):
        interface = interface.detached_copy()
        interface.validate()
    return interface
