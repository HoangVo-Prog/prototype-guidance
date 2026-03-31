from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


INIT_MODE_ALIASES = {
    'normalized_random': 'normalized_random',
    'sampled_image_embeddings': 'external_embeddings',
    'kmeans_centroids': 'external_embeddings',
}


class PrototypeBank(nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        prototype_dim: int,
        init_mode: str = 'normalized_random',
        init_path: Optional[str] = None,
        normalize_init: bool = True,
        init_scale: float = 0.02,
    ):
        super().__init__()
        if num_prototypes <= 0:
            raise ValueError('num_prototypes must be positive.')
        if prototype_dim <= 0:
            raise ValueError('prototype_dim must be positive.')
        canonical_init_mode = INIT_MODE_ALIASES.get(str(init_mode).lower())
        if canonical_init_mode is None:
            raise ValueError(f'Unsupported prototype init mode: {init_mode}')
        self.num_prototypes = int(num_prototypes)
        self.prototype_dim = int(prototype_dim)
        self.init_mode = canonical_init_mode
        self.requested_init_mode = str(init_mode).lower()
        self.init_path = init_path
        self.normalize_init = bool(normalize_init)
        self.init_scale = float(init_scale)
        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, self.prototype_dim))
        self.reset_parameters()

    def _normalize_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, dim=-1)

    def _load_external_prototypes(self) -> torch.Tensor:
        if not self.init_path:
            raise ValueError(
                f'Prototype init mode `{self.requested_init_mode}` requires `prototype_init_path` '
                'to point to sampled embeddings or k-means centroids.'
            )
        loaded = torch.load(self.init_path, map_location='cpu')
        if isinstance(loaded, dict):
            if 'prototypes' in loaded:
                loaded = loaded['prototypes']
            elif 'state_dict' in loaded and 'prototypes' in loaded['state_dict']:
                loaded = loaded['state_dict']['prototypes']
        loaded = torch.as_tensor(loaded, dtype=self.prototypes.dtype)
        if loaded.shape != self.prototypes.shape:
            raise ValueError(
                f'Loaded prototypes have shape {tuple(loaded.shape)} but expected {tuple(self.prototypes.shape)}.'
            )
        return loaded

    def reset_parameters(self) -> None:
        if self.init_mode == 'external_embeddings':
            value = self._load_external_prototypes()
        elif self.init_mode == 'normalized_random':
            value = torch.randn_like(self.prototypes) * self.init_scale
        else:
            raise ValueError(f'Unsupported canonical prototype init mode: {self.init_mode}')

        if self.normalize_init:
            value = self._normalize_rows(value)
        with torch.no_grad():
            self.prototypes.copy_(value)

    def get_prototypes(self) -> torch.Tensor:
        return self.prototypes

    def forward(self, return_debug: bool = False):
        prototypes = self.get_prototypes()
        if not return_debug:
            return prototypes
        debug = {
            'raw_prototypes': prototypes,
            'prototype_init_mode': self.requested_init_mode,
            'prototype_norm_mean': prototypes.norm(dim=-1).mean().detach(),
            'prototype_norm_std': prototypes.norm(dim=-1).std(unbiased=False).detach(),
        }
        return prototypes, debug
