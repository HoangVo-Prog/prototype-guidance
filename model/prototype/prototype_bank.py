import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)
_NORMALIZATION_EPS = 1e-12
_HYBRID_DUPLICATE_COSINE_THRESHOLD = 0.98
_HYBRID_DUPLICATE_MAX_ATTEMPTS = 4

INIT_MODE_ALIASES = {
    'normalized_random': 'normalized_random',
    'sampled_image_embeddings': 'sampled_image_embeddings',
    'kmeans_centroids': 'kmeans_centroids',
    'orthogonal_normalized_random': 'orthogonal_normalized_random',
    'spherical_kmeans_centroids': 'spherical_kmeans_centroids',
    'hybrid_spherical_kmeans_random': 'hybrid_spherical_kmeans_random',
}
DATA_DRIVEN_INIT_MODES = {
    'sampled_image_embeddings',
    'kmeans_centroids',
    'spherical_kmeans_centroids',
    'hybrid_spherical_kmeans_random',
}


def init_mode_requires_data(init_mode: str) -> bool:
    canonical_init_mode = INIT_MODE_ALIASES.get(str(init_mode).lower())
    if canonical_init_mode is None:
        raise ValueError(f'Unsupported prototype init mode: {init_mode}')
    return canonical_init_mode in DATA_DRIVEN_INIT_MODES


class PrototypeBank(nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        prototype_dim: int,
        init_mode: str = 'normalized_random',
        init_path: Optional[str] = None,
        normalize_init: bool = True,
        init_scale: float = 0.02,
        init_hybrid_ratio: float = 0.5,
        init_max_iters: int = 50,
        init_tol: float = 1e-4,
        init_seed: Optional[int] = None,
        init_features: Optional[torch.Tensor] = None,
        defer_initialization: bool = False,
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
        self.init_hybrid_ratio = float(init_hybrid_ratio)
        self.init_max_iters = int(init_max_iters)
        self.init_tol = float(init_tol)
        self.init_seed = None if init_seed is None else int(init_seed)
        self.init_features = None if init_features is None else torch.as_tensor(init_features, dtype=torch.float32, device='cpu')
        self.defer_initialization = bool(defer_initialization)
        self.last_init_diagnostics: Dict[str, Any] = {}
        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, self.prototype_dim))
        self._initialized = False
        if self.defer_initialization:
            with torch.no_grad():
                self.prototypes.zero_()
            self.last_init_diagnostics = {
                'deferred_initialization': True,
                'prototype_shape': tuple(self.prototypes.shape),
            }
        else:
            self.reset_parameters()

    def _normalize_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, dim=-1, eps=_NORMALIZATION_EPS)

    def _make_generator(self) -> Optional[torch.Generator]:
        if self.init_seed is None:
            return None
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.init_seed)
        return generator

    def _randn(self, shape: Tuple[int, ...], generator: Optional[torch.Generator]) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.float32, device='cpu', generator=generator)

    def _randint(self, high: int, size: Tuple[int, ...], generator: Optional[torch.Generator]) -> torch.Tensor:
        return torch.randint(high, size, device='cpu', generator=generator)

    def _stabilize_qr(self, q_matrix: torch.Tensor, r_matrix: torch.Tensor) -> torch.Tensor:
        diagonal = torch.diagonal(r_matrix, 0)
        signs = torch.sign(diagonal)
        signs[signs == 0] = 1
        return q_matrix * signs.unsqueeze(0)

    def _generate_orthogonal_normalized_random(
        self,
        num_rows: int,
        feature_dim: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if num_rows <= feature_dim:
            square = self._randn((feature_dim, feature_dim), generator=generator)
            q_matrix, r_matrix = torch.linalg.qr(square, mode='reduced')
            q_matrix = self._stabilize_qr(q_matrix, r_matrix)
            prototypes = q_matrix[:num_rows]
        else:
            random_matrix = self._randn((num_rows, feature_dim), generator=generator)
            q_matrix, r_matrix = torch.linalg.qr(random_matrix, mode='reduced')
            q_matrix = self._stabilize_qr(q_matrix, r_matrix)
            prototypes = q_matrix
        return self._normalize_rows(prototypes)

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

    def _collect_tensor_candidates(self, payload: Any, prefix: str = '') -> List[Tuple[str, torch.Tensor]]:
        candidates: List[Tuple[str, torch.Tensor]] = []
        if torch.is_tensor(payload) or (np is not None and isinstance(payload, np.ndarray)):
            key = prefix or '<root>'
            candidates.append((key, torch.as_tensor(payload)))
            return candidates
        if isinstance(payload, dict):
            for key in sorted(payload.keys()):
                child_prefix = f'{prefix}.{key}' if prefix else str(key)
                candidates.extend(self._collect_tensor_candidates(payload[key], child_prefix))
        return candidates

    def _select_feature_tensor(self, payload: Dict[str, Any]) -> Tuple[str, torch.Tensor]:
        candidates = self._collect_tensor_candidates(payload)
        if not candidates:
            raise ValueError(
                f'Prototype init mode `{self.requested_init_mode}` expected a tensor or checkpoint with feature tensors '
                f'at `{self.init_path}`, but found no tensor-like values.'
            )
        if len(candidates) == 1:
            return candidates[0]

        preferred_tokens = ('feature', 'features', 'embedding', 'embeddings', 'vector', 'vectors')
        preferred = [
            (key, tensor)
            for key, tensor in candidates
            if any(token in key.lower() for token in preferred_tokens)
        ]
        if len(preferred) == 1:
            return preferred[0]

        candidate_summary = ', '.join(f'{key}: {tuple(tensor.shape)}' for key, tensor in candidates)
        raise ValueError(
            f'Ambiguous feature checkpoint for prototype init mode `{self.requested_init_mode}` at `{self.init_path}`. '
            f'Candidate tensor keys/shapes: {candidate_summary}'
        )

    def _prepare_feature_matrix(
        self,
        features: torch.Tensor,
        *,
        source_path: Optional[str],
        feature_key: str,
        source_kind: str,
        normalize: bool,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        features = torch.as_tensor(features, dtype=torch.float32)
        if features.ndim == 0:
            raise ValueError(
                f'Loaded feature tensor for prototype init mode `{self.requested_init_mode}` must have at least one dimension, '
                f'but got scalar shape {tuple(features.shape)}.'
            )
        if features.ndim == 1:
            features = features.unsqueeze(0)
        else:
            features = features.reshape(-1, features.shape[-1])
        if features.shape[-1] != self.prototype_dim:
            raise ValueError(
                f'Prototype init mode `{self.requested_init_mode}` expected image features in prototype space with '
                f'feature dim {self.prototype_dim}, but found feature dim {features.shape[-1]} '
                f'from source `{source_path or source_kind}`.'
            )
        if features.shape[0] == 0:
            raise ValueError(
                f'Prototype init mode `{self.requested_init_mode}` found zero usable feature rows in source '
                f'`{source_path or source_kind}` after flattening.'
            )
        if not torch.isfinite(features).all():
            raise ValueError(
                f'Prototype init mode `{self.requested_init_mode}` found non-finite values in feature source '
                f'`{source_path or source_kind}`.'
            )
        if normalize:
            features = self._normalize_rows(features)
        diagnostics = {
            'source_path': source_path,
            'source_kind': source_kind,
            'feature_key': feature_key,
            'feature_count': int(features.shape[0]),
            'feature_dim': int(features.shape[1]),
            'feature_normalized_for_init': bool(normalize),
            'auto_train_image_fallback_used': bool(self.init_features is not None and not self.init_path),
        }
        return features, diagnostics

    def _load_feature_matrix(self, *, normalize: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.init_path:
            loaded = torch.load(self.init_path, map_location='cpu')
            feature_key = '<root>'
            if torch.is_tensor(loaded) or (np is not None and isinstance(loaded, np.ndarray)):
                features = torch.as_tensor(loaded)
            elif isinstance(loaded, dict):
                feature_key, features = self._select_feature_tensor(loaded)
            else:
                raise ValueError(
                    f'Prototype init mode `{self.requested_init_mode}` expected a tensor or dict checkpoint at '
                    f'`{self.init_path}`, but found `{type(loaded).__name__}`.'
                )
            return self._prepare_feature_matrix(
                features,
                source_path=self.init_path,
                feature_key=feature_key,
                source_kind='path_checkpoint',
                normalize=normalize,
            )
        if self.init_features is not None:
            return self._prepare_feature_matrix(
                self.init_features,
                source_path=None,
                feature_key='<in_memory>',
                source_kind='train_image_embeddings',
                normalize=normalize,
            )
        raise ValueError(
            f'Prototype init mode `{self.requested_init_mode}` requires a data source. '
            'Provide `prototype_init_path` or supply in-memory train image embeddings in prototype space.'
        )

    def _initialize_spherical_centers(
        self,
        features: torch.Tensor,
        num_centers: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        num_samples = features.size(0)
        first_index = int(self._randint(num_samples, (1,), generator=generator).item())
        selected_indices = [first_index]
        best_similarity = features @ features[first_index]
        for _ in range(1, num_centers):
            next_index = int(torch.argmin(best_similarity).item())
            selected_indices.append(next_index)
            best_similarity = torch.maximum(best_similarity, features @ features[next_index])
        return features[selected_indices].clone()

    def _select_worst_fit_reseed_index(
        self,
        assigned_metric: torch.Tensor,
        used_mask: torch.Tensor,
        *,
        largest: bool,
    ) -> int:
        candidate_order = torch.argsort(assigned_metric, descending=largest)
        for candidate in candidate_order.tolist():
            if not bool(used_mask[candidate].item()):
                used_mask[candidate] = True
                return int(candidate)
        candidate = int(candidate_order[0].item())
        used_mask[candidate] = True
        return candidate

    def _initialize_kmeans_centers(
        self,
        features: torch.Tensor,
        num_centers: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        num_samples = features.size(0)
        first_index = int(self._randint(num_samples, (1,), generator=generator).item())
        selected_indices = [first_index]
        best_sq_distance = ((features - features[first_index]) ** 2).sum(dim=1)
        for _ in range(1, num_centers):
            next_index = int(torch.argmax(best_sq_distance).item())
            selected_indices.append(next_index)
            candidate_sq_distance = ((features - features[next_index]) ** 2).sum(dim=1)
            best_sq_distance = torch.minimum(best_sq_distance, candidate_sq_distance)
        return features[selected_indices].clone()

    def _run_kmeans(
        self,
        features: torch.Tensor,
        num_centers: int,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.init_max_iters <= 0:
            raise ValueError('prototype_init_max_iters must be positive for k-means prototype initialization.')
        if self.init_tol < 0:
            raise ValueError('prototype_init_tol must be non-negative for k-means prototype initialization.')

        centers = self._initialize_kmeans_centers(features, num_centers, generator=generator)
        empty_reseeds = 0
        iterations = 0

        for iteration in range(1, self.init_max_iters + 1):
            distances = torch.cdist(features, centers, p=2)
            assignments = distances.argmin(dim=1)
            assigned_distance = distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
            previous_centers = centers.clone()
            updated_centers = torch.empty_like(centers)
            reseed_used_mask = torch.zeros(features.size(0), dtype=torch.bool, device=features.device)

            for cluster_index in range(num_centers):
                cluster_mask = assignments.eq(cluster_index)
                if cluster_mask.any():
                    updated_centers[cluster_index] = features[cluster_mask].mean(dim=0)
                    continue
                reseed_index = self._select_worst_fit_reseed_index(
                    assigned_distance,
                    reseed_used_mask,
                    largest=True,
                )
                updated_centers[cluster_index] = features[reseed_index]
                empty_reseeds += 1

            centers = updated_centers
            max_center_change = (centers - previous_centers).norm(dim=-1).max().item()
            iterations = iteration
            if max_center_change <= self.init_tol:
                break

        return centers, {
            'cluster_iterations': iterations,
            'empty_cluster_reseeds': empty_reseeds,
        }

    def _run_spherical_kmeans(
        self,
        features: torch.Tensor,
        num_centers: int,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.init_max_iters <= 0:
            raise ValueError('prototype_init_max_iters must be positive for spherical prototype initialization.')
        if self.init_tol < 0:
            raise ValueError('prototype_init_tol must be non-negative for spherical prototype initialization.')

        centers = self._initialize_spherical_centers(features, num_centers, generator=generator)
        empty_reseeds = 0
        iterations = 0

        for iteration in range(1, self.init_max_iters + 1):
            similarities = features @ centers.t()
            assignments = similarities.argmax(dim=1)
            assigned_similarity = similarities.gather(1, assignments.unsqueeze(1)).squeeze(1)
            previous_centers = centers.clone()
            updated_centers = torch.empty_like(centers)
            reseed_used_mask = torch.zeros(features.size(0), dtype=torch.bool, device=features.device)

            for cluster_index in range(num_centers):
                cluster_mask = assignments.eq(cluster_index)
                if cluster_mask.any():
                    candidate_center = features[cluster_mask].mean(dim=0)
                    if candidate_center.norm(p=2).item() > _NORMALIZATION_EPS:
                        updated_centers[cluster_index] = self._normalize_rows(candidate_center.unsqueeze(0)).squeeze(0)
                        continue
                reseed_index = self._select_worst_fit_reseed_index(
                    assigned_similarity,
                    reseed_used_mask,
                    largest=False,
                )
                updated_centers[cluster_index] = features[reseed_index]
                empty_reseeds += 1

            centers = self._normalize_rows(updated_centers)
            max_center_change = (centers - previous_centers).norm(dim=-1).max().item()
            iterations = iteration
            if max_center_change <= self.init_tol:
                break

        return centers, {
            'cluster_iterations': iterations,
            'empty_cluster_reseeds': empty_reseeds,
        }

    def _cleanup_hybrid_duplicates(
        self,
        prototypes: torch.Tensor,
        random_start_index: int,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, int]:
        replacements = 0
        if prototypes.size(0) <= 1 or random_start_index >= prototypes.size(0):
            return prototypes, replacements

        prototypes = prototypes.clone()
        for row_index in range(random_start_index, prototypes.size(0)):
            for _ in range(_HYBRID_DUPLICATE_MAX_ATTEMPTS):
                similarities = torch.matmul(prototypes[:row_index], prototypes[row_index])
                if similarities.numel() == 0 or similarities.max().item() <= _HYBRID_DUPLICATE_COSINE_THRESHOLD:
                    break
                prototypes[row_index] = self._generate_orthogonal_normalized_random(1, self.prototype_dim, generator).squeeze(0)
                replacements += 1
            prototypes[row_index] = self._normalize_rows(prototypes[row_index].unsqueeze(0)).squeeze(0)
        return prototypes, replacements

    def _build_orthogonal_init(self, generator: Optional[torch.Generator]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        value = self._generate_orthogonal_normalized_random(
            self.num_prototypes,
            self.prototype_dim,
            generator=generator,
        )
        return value, {
            'source_path': None,
            'source_kind': 'random',
            'feature_count': None,
            'feature_dim': self.prototype_dim,
            'feature_normalized_for_init': False,
            'auto_train_image_fallback_used': False,
            'cluster_iterations': 0,
            'empty_cluster_reseeds': 0,
            'clustering_strategy': 'none',
        }

    def _build_sampled_embedding_init(
        self,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        features, diagnostics = self._load_feature_matrix(normalize=False)
        if features.size(0) >= self.num_prototypes:
            sample_indices = torch.randperm(features.size(0), generator=generator)[:self.num_prototypes]
            sampled_with_replacement = False
        else:
            sample_indices = self._randint(features.size(0), (self.num_prototypes,), generator=generator)
            sampled_with_replacement = True
        diagnostics.update({
            'cluster_iterations': 0,
            'empty_cluster_reseeds': 0,
            'clustering_strategy': 'sampled_image_embeddings',
            'sampled_with_replacement': sampled_with_replacement,
        })
        return features.index_select(0, sample_indices), diagnostics

    def _build_kmeans_init(
        self,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        features, diagnostics = self._load_feature_matrix(normalize=False)
        centers, clustering_diagnostics = self._run_kmeans(
            features,
            self.num_prototypes,
            generator=generator,
        )
        diagnostics.update(clustering_diagnostics)
        diagnostics['clustering_strategy'] = 'kmeans'
        return centers, diagnostics

    def _build_spherical_kmeans_init(
        self,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        features, diagnostics = self._load_feature_matrix(normalize=True)
        centers, clustering_diagnostics = self._run_spherical_kmeans(
            features,
            self.num_prototypes,
            generator=generator,
        )
        diagnostics.update(clustering_diagnostics)
        diagnostics['clustering_strategy'] = 'spherical_kmeans'
        return centers, diagnostics

    def _build_hybrid_spherical_random_init(
        self,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not 0.0 <= self.init_hybrid_ratio <= 1.0:
            raise ValueError('prototype_init_hybrid_ratio must be in the range [0, 1].')

        features, diagnostics = self._load_feature_matrix(normalize=True)
        data_count = int(round(self.num_prototypes * self.init_hybrid_ratio))
        data_count = max(0, min(self.num_prototypes, data_count))
        random_count = self.num_prototypes - data_count

        parts: List[torch.Tensor] = []
        clustering_diagnostics: Dict[str, Any] = {
            'cluster_iterations': 0,
            'empty_cluster_reseeds': 0,
        }
        if data_count > 0:
            data_centers, clustering_diagnostics = self._run_spherical_kmeans(
                features,
                data_count,
                generator=generator,
            )
            parts.append(data_centers)
        if random_count > 0:
            random_rows = self._generate_orthogonal_normalized_random(
                random_count,
                self.prototype_dim,
                generator=generator,
            )
            parts.append(random_rows)

        combined = torch.cat(parts, dim=0)
        combined = self._normalize_rows(combined)
        combined, cleanup_replacements = self._cleanup_hybrid_duplicates(
            combined,
            random_start_index=data_count,
            generator=generator,
        )
        combined = self._normalize_rows(combined)

        diagnostics.update(clustering_diagnostics)
        diagnostics.update({
            'hybrid_data_count': data_count,
            'hybrid_random_count': random_count,
            'hybrid_cleanup_replacements': cleanup_replacements,
            'clustering_strategy': 'hybrid_spherical_kmeans_random',
        })
        return combined, diagnostics

    def _summarize_initialized_tensor(self, tensor: torch.Tensor) -> Dict[str, float]:
        tensor = tensor.detach().to(dtype=torch.float32, device='cpu')
        row_norms = tensor.norm(dim=-1)
        summary: Dict[str, float] = {
            'row_norm_min': float(row_norms.min().item()),
            'row_norm_max': float(row_norms.max().item()),
            'row_norm_mean': float(row_norms.mean().item()),
        }
        if tensor.size(0) <= 1:
            summary['max_offdiag_cosine'] = 0.0
            return summary
        pairwise = tensor @ tensor.t()
        off_diagonal_mask = ~torch.eye(pairwise.size(0), dtype=torch.bool)
        summary['max_offdiag_cosine'] = float(pairwise.masked_fill(~off_diagonal_mask, -1.0).max().item())
        return summary

    def _log_init_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        LOGGER.info(
            'Initialized prototype bank mode=%s path_provided=%s source_kind=%s source_path=%s '
            'auto_train_image_fallback=%s feature_count=%s feature_dim=%s feature_normalized=%s '
            'clustering=%s cluster_iterations=%s empty_cluster_reseeds=%s prototype_shape=%s '
            'row_norm_min=%.6f row_norm_mean=%.6f row_norm_max=%.6f max_offdiag_cosine=%.6f',
            self.requested_init_mode,
            bool(self.init_path),
            diagnostics.get('source_kind'),
            diagnostics.get('source_path'),
            diagnostics.get('auto_train_image_fallback_used', False),
            diagnostics.get('feature_count'),
            diagnostics.get('feature_dim'),
            diagnostics.get('feature_normalized_for_init', False),
            diagnostics.get('clustering_strategy', 'none'),
            diagnostics.get('cluster_iterations', 0),
            diagnostics.get('empty_cluster_reseeds', 0),
            diagnostics.get('prototype_shape'),
            diagnostics['row_norm_min'],
            diagnostics['row_norm_mean'],
            diagnostics['row_norm_max'],
            diagnostics['max_offdiag_cosine'],
        )

    def reset_parameters(self) -> None:
        generator = self._make_generator()
        diagnostics: Dict[str, Any] = {
            'source_path': self.init_path,
            'source_kind': 'path_checkpoint' if self.init_path else ('train_image_embeddings' if self.init_features is not None else 'random'),
            'feature_count': None,
            'feature_dim': self.prototype_dim,
            'feature_normalized_for_init': False,
            'auto_train_image_fallback_used': bool(self.init_features is not None and not self.init_path),
            'cluster_iterations': 0,
            'empty_cluster_reseeds': 0,
            'clustering_strategy': 'none',
        }
        with torch.no_grad():
            if self.init_mode == 'sampled_image_embeddings':
                if self.init_path:
                    diagnostics['source_kind'] = 'path_prototypes'
                    diagnostics['clustering_strategy'] = 'precomputed_prototypes'
                    value = self._load_external_prototypes()
                else:
                    value, diagnostics = self._build_sampled_embedding_init(generator=generator)
            elif self.init_mode == 'kmeans_centroids':
                if self.init_path:
                    diagnostics['source_kind'] = 'path_prototypes'
                    diagnostics['clustering_strategy'] = 'precomputed_prototypes'
                    value = self._load_external_prototypes()
                else:
                    value, diagnostics = self._build_kmeans_init(generator=generator)
            elif self.init_mode == 'normalized_random':
                diagnostics['source_kind'] = 'random'
                value = torch.randn_like(self.prototypes) * self.init_scale
            elif self.init_mode == 'orthogonal_normalized_random':
                value, diagnostics = self._build_orthogonal_init(generator=generator)
            elif self.init_mode == 'spherical_kmeans_centroids':
                value, diagnostics = self._build_spherical_kmeans_init(generator=generator)
            elif self.init_mode == 'hybrid_spherical_kmeans_random':
                value, diagnostics = self._build_hybrid_spherical_random_init(generator=generator)
            else:
                raise ValueError(f'Unsupported canonical prototype init mode: {self.init_mode}')

            force_normalize = self.init_mode in {
                'orthogonal_normalized_random',
                'spherical_kmeans_centroids',
                'hybrid_spherical_kmeans_random',
            }
            if force_normalize:
                value = self._normalize_rows(value.to(dtype=torch.float32))
            elif self.normalize_init:
                value = self._normalize_rows(value.to(dtype=torch.float32))
            if not torch.isfinite(value).all():
                raise ValueError(f'Prototype init mode `{self.requested_init_mode}` produced non-finite values.')

            summary = self._summarize_initialized_tensor(value)
            diagnostics.update(summary)
            diagnostics['prototype_shape'] = tuple(value.shape)
            diagnostics['deferred_initialization'] = False
            self.last_init_diagnostics = diagnostics
            self._log_init_diagnostics(diagnostics)

            self.prototypes.copy_(value.to(dtype=self.prototypes.dtype))
            self._initialized = True

    def is_initialized(self) -> bool:
        return bool(self._initialized)

    def initialize_if_needed(self) -> bool:
        if self._initialized:
            return False
        self.reset_parameters()
        return True

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        prototype_key = f'{prefix}prototypes'
        if prototype_key in state_dict:
            self._initialized = True

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
            'prototype_init_diagnostics': dict(self.last_init_diagnostics),
        }
        return prototypes, debug
