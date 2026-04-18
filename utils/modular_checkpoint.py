from __future__ import annotations

import copy
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch

from utils.module_group_registry import CHECKPOINT_GROUPS, get_group_state_dict, load_group_state_dict


_ITSELF_LEGACY_KEY_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ('visul_emb_layer.linear.', 'host_head.visual_embedding_layer.fc.'),
    ('visual_emb_layer.linear.', 'host_head.visual_embedding_layer.fc.'),
    ('classifier_bge.', 'host_head.classifier_global.'),
    ('classifier_id_bge.', 'host_head.classifier_id_global.'),
    ('mlp_bge.', 'host_head.mlp_global.'),
    ('classifier_tse.', 'host_head.classifier_grab.'),
    ('classifier_id_tse.', 'host_head.classifier_id_grab.'),
    ('mlp_tse.', 'host_head.mlp_grab.'),
    ('visul_emb_layer.', 'host_head.visual_embedding_layer.'),
    ('visual_emb_layer.', 'host_head.visual_embedding_layer.'),
    ('texual_emb_layer.', 'host_head.textual_embedding_layer.'),
    ('textual_emb_layer.', 'host_head.textual_embedding_layer.'),
)


DEFAULT_CHECKPOINTING_CONFIG = {
    'metric': {
        'name': 'R1',
        'mode': 'max',
    },
    'groups': {
        'host': {'enabled': True},
        'prototype_bank': {'enabled': True},
        'prototype_projector': {'enabled': True},
    },
    'save': {
        'dir': None,
        'save_latest': True,
        'save_best': True,
        'keep_last_n': 1,
        'artifacts': {
            'host': {
                'enabled': True,
                'filename_latest': 'checkpoint_host_latest.pth',
                'filename_best': 'checkpoint_host_best.pth',
            },
            'prototype_bank': {
                'enabled': True,
                'filename_latest': 'checkpoint_prototype_bank_latest.pth',
                'filename_best': 'checkpoint_prototype_bank_best.pth',
            },
            'prototype_projector': {
                'enabled': True,
                'filename_latest': 'checkpoint_prototype_projector_latest.pth',
                'filename_best': 'checkpoint_prototype_projector_best.pth',
            },
        },
    },
    'load': {
        'enabled': False,
        'strict': True,
        'sources': {
            'host': {'enabled': False, 'path': None},
            'prototype_bank': {'enabled': False, 'path': None},
            'prototype_projector': {'enabled': False, 'path': None},
        },
    },
    'authority_validation': {
        'enabled': True,
        'strict': True,
        'warn_only': False,
        'allow_fallback_row_name_classification': True,
    },
}


def _deep_merge(base: dict, override: dict):
    merged = copy.deepcopy(base)
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_checkpointing_config(config_data: Optional[dict]):
    checkpointing_from_config = {}
    if isinstance(config_data, dict):
        checkpointing_from_config = config_data.get('checkpointing', {}) or {}
    config = _deep_merge(DEFAULT_CHECKPOINTING_CONFIG, checkpointing_from_config)
    return config


def _extract_state_dict_from_checkpoint_payload(payload):
    if isinstance(payload, dict):
        modular_state = payload.get('state_dict')
        if isinstance(modular_state, dict) and 'group_name' in payload:
            return modular_state, 'modular_group_payload'

        model_state = payload.get('model')
        if isinstance(model_state, dict):
            return model_state, 'legacy_full_checkpoint_model_field'

        generic_state = payload.get('state_dict')
        if isinstance(generic_state, dict):
            return generic_state, 'generic_state_dict_field'

        if payload and all(isinstance(key, str) for key in payload.keys()):
            return payload, 'plain_state_dict'

    raise ValueError('Checkpoint payload does not contain a valid state_dict mapping.')


def _normalize_checkpoint_key(key: str):
    normalized = str(key)
    if normalized.startswith('module.'):
        normalized = normalized[len('module.'):]
    if normalized.startswith('model.'):
        normalized = normalized[len('model.'):]
    return normalized


def _remap_legacy_itself_host_key(key: str):
    for legacy_prefix, current_prefix in _ITSELF_LEGACY_KEY_PREFIXES:
        if key.startswith(legacy_prefix):
            return current_prefix + key[len(legacy_prefix):], True
    return key, False


def _prepare_state_dict_for_group_compatibility(group_name: str, state_dict: dict, model, host_type: str):
    if group_name != 'host':
        return state_dict, {'legacy_renamed': 0, 'host_head_prefixed': 0}

    model_keys = set(model.state_dict().keys())
    prepared = {}
    stats = {'legacy_renamed': 0, 'host_head_prefixed': 0}

    for raw_key, value in state_dict.items():
        key = _normalize_checkpoint_key(str(raw_key))

        if host_type == 'itself':
            key, renamed = _remap_legacy_itself_host_key(key)
            if renamed:
                stats['legacy_renamed'] += 1

            if (
                not key.startswith(('host_head.', 'base_model.', 'prototype_head.', 'host_core.', 'prototype_plugin.'))
                and ('host_head.' + key) in model_keys
            ):
                key = 'host_head.' + key
                stats['host_head_prefixed'] += 1

        prepared[key] = value

    return prepared, stats


GROUP_AUTHORITY_BUCKET = {
    'host': 'host',
    'prototype_bank': 'host',
    'prototype_projector': 'host',
}

GROUP_COMPONENT_NAME = {
    'host': 'HostCore',
    'prototype_bank': 'PrototypePlugin',
    'prototype_projector': 'PrototypePlugin',
}


@dataclass(frozen=True)
class AuthorityValidationResult:
    valid: bool
    expected_bucket: Optional[str]
    actual_bucket: Optional[str]
    reason: str
    group_name: str
    row_name: Optional[str]


class MetricAuthorityPolicy:
    def __init__(
        self,
        *,
        enabled: bool = True,
        strict: bool = True,
        warn_only: bool = False,
        allow_fallback_row_name_classification: bool = True,
    ):
        self.enabled = bool(enabled)
        self.strict = bool(strict)
        self.warn_only = bool(warn_only)
        self.allow_fallback_row_name_classification = bool(allow_fallback_row_name_classification)

    @staticmethod
    def expected_bucket_for_group(group_name: str) -> Optional[str]:
        return GROUP_AUTHORITY_BUCKET.get(str(group_name))

    @staticmethod
    def _normalize_row_name(row_name: Optional[str]) -> Optional[str]:
        if row_name is None:
            return None
        normalized = str(row_name).strip()
        return normalized or None

    @staticmethod
    def _classify_row_name_fallback(row_name: Optional[str]) -> Optional[str]:
        if row_name is None:
            return None
        label = str(row_name).strip().lower()
        if not label:
            return None
        if re.match(r'^host(?:[-_\s]*only)?(?:[-_\s]*t2i)?(?:\b|[-_\s].*)?$', label):
            return 'host'
        return None

    def classify_row(self, row_name: Optional[str], row_roles: Optional[Dict[str, str]] = None) -> Optional[str]:
        normalized = self._normalize_row_name(row_name)
        if normalized is None:
            return None
        if isinstance(row_roles, dict):
            role = str(row_roles.get(normalized, '')).strip().lower()
            if role in {'host'}:
                return role
        if self.allow_fallback_row_name_classification:
            fallback_role = self._classify_row_name_fallback(normalized)
            if fallback_role is not None:
                return fallback_role
        return None

    def validate_row_for_group(
        self,
        *,
        group_name: str,
        row_name: Optional[str],
        row_roles: Optional[Dict[str, str]] = None,
    ) -> AuthorityValidationResult:
        expected_bucket = self.expected_bucket_for_group(group_name)
        normalized_row = self._normalize_row_name(row_name)
        if not self.enabled:
            return AuthorityValidationResult(
                valid=True,
                expected_bucket=expected_bucket,
                actual_bucket=None,
                reason='authority_validation_disabled',
                group_name=str(group_name),
                row_name=normalized_row,
            )

        if expected_bucket is None:
            return AuthorityValidationResult(
                valid=False,
                expected_bucket=None,
                actual_bucket=None,
                reason='unsupported_checkpoint_group',
                group_name=str(group_name),
                row_name=normalized_row,
            )

        actual_bucket = self.classify_row(normalized_row, row_roles=row_roles)
        if actual_bucket is None:
            return AuthorityValidationResult(
                valid=False,
                expected_bucket=expected_bucket,
                actual_bucket=None,
                reason='row_bucket_unresolved',
                group_name=str(group_name),
                row_name=normalized_row,
            )

        if actual_bucket != expected_bucket:
            return AuthorityValidationResult(
                valid=False,
                expected_bucket=expected_bucket,
                actual_bucket=actual_bucket,
                reason='row_bucket_mismatch',
                group_name=str(group_name),
                row_name=normalized_row,
            )

        return AuthorityValidationResult(
            valid=True,
            expected_bucket=expected_bucket,
            actual_bucket=actual_bucket,
            reason='ok',
            group_name=str(group_name),
            row_name=normalized_row,
        )


class ModularCheckpointManager:
    def __init__(self, args, save_dir: str, logger):
        self.args = args
        self.logger = logger
        self.save_dir = save_dir
        self.config = resolve_checkpointing_config(getattr(args, 'config_data', None))
        self.metric_name = str(self.config.get('metric', {}).get('name', 'R1')).strip() or 'R1'
        if self.metric_name.upper() != 'R1':
            self.logger.warning(
                'checkpointing.metric.name=%s is currently unsupported for selection. Falling back to R1.',
                self.metric_name,
            )
            self.metric_name = 'R1'
        self.metric_mode = str(self.config.get('metric', {}).get('mode', 'max')).strip().lower() or 'max'
        if self.metric_mode not in {'max', 'min'}:
            raise ValueError(f'checkpointing.metric.mode must be one of ["max", "min"], got {self.metric_mode!r}.')
        self.best_metric_value_by_group: Dict[str, float] = {}

        authority_cfg = self.config.get('authority_validation', {}) if isinstance(self.config.get('authority_validation', {}), dict) else {}
        self.authority_policy = MetricAuthorityPolicy(
            enabled=bool(authority_cfg.get('enabled', True)),
            strict=bool(authority_cfg.get('strict', True)),
            warn_only=bool(authority_cfg.get('warn_only', False)),
            allow_fallback_row_name_classification=bool(
                authority_cfg.get('allow_fallback_row_name_classification', True)
            ),
        )

    @staticmethod
    def _strict_or_warn(*, strict: bool, logger, message: str) -> None:
        if strict:
            raise RuntimeError(message)
        logger.warning(message)

    @staticmethod
    def _normalize_compatibility_payload(metadata: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        if not isinstance(metadata, dict):
            return normalized
        compatibility_block = metadata.get('compatibility')
        if isinstance(compatibility_block, dict):
            normalized.update(compatibility_block)
        for key in (
            'runtime_mode',
            'component_name',
            'component_schema_version',
            'host_export_interface_version',
            'accepted_host_interface_versions',
            'expected_host_score_schema_version',
            'compatible_runtime_modes',
        ):
            if key in metadata and key not in normalized:
                normalized[key] = metadata.get(key)
        return normalized

    @staticmethod
    def _model_group_compatibility(model, group_name: str) -> Dict[str, Any]:
        if hasattr(model, 'get_group_checkpoint_compatibility'):
            compatibility = model.get_group_checkpoint_compatibility(group_name)
            if isinstance(compatibility, dict):
                return dict(compatibility)
        return {}

    def _validate_checkpoint_compatibility(
        self,
        *,
        group_name: str,
        payload_type: str,
        payload_metadata: Dict[str, Any],
        model,
        strict: bool,
    ) -> None:
        if payload_type != 'modular_group_payload':
            return
        metadata = payload_metadata if isinstance(payload_metadata, dict) else {}
        checkpoint_compat = self._normalize_compatibility_payload(metadata)
        model_compat = self._model_group_compatibility(model, group_name)
        expected_component = GROUP_COMPONENT_NAME.get(group_name)

        payload_component = checkpoint_compat.get('component_name')
        if expected_component and payload_component and str(payload_component) != str(expected_component):
            self._strict_or_warn(
                strict=strict,
                logger=self.logger,
                message=(
                    f'Checkpoint component mismatch for group={group_name}: '
                    f'expected {expected_component!r}, got {payload_component!r}.'
                ),
            )
        model_component = model_compat.get('component_name')
        if model_component and payload_component and str(payload_component) != str(model_component):
            self._strict_or_warn(
                strict=strict,
                logger=self.logger,
                message=(
                    f'Checkpoint/model component mismatch for group={group_name}: '
                    f'checkpoint={payload_component!r}, model={model_component!r}.'
                ),
            )

        payload_schema_version = checkpoint_compat.get('component_schema_version')
        model_schema_version = model_compat.get('component_schema_version')
        if payload_schema_version and model_schema_version and str(payload_schema_version) != str(model_schema_version):
            self._strict_or_warn(
                strict=strict,
                logger=self.logger,
                message=(
                    f'Checkpoint component schema mismatch for group={group_name}: '
                    f'checkpoint={payload_schema_version!r}, model={model_schema_version!r}.'
                ),
            )

        payload_runtime_mode = checkpoint_compat.get('runtime_mode')
        model_runtime_mode = model_compat.get('runtime_mode')
        model_allowed_modes = model_compat.get('compatible_runtime_modes')
        if (
            payload_runtime_mode is not None
            and isinstance(model_allowed_modes, (list, tuple, set))
            and len(model_allowed_modes) > 0
            and str(payload_runtime_mode) not in {str(item) for item in model_allowed_modes}
        ):
            self._strict_or_warn(
                strict=strict,
                logger=self.logger,
                message=(
                    f'Checkpoint runtime_mode incompatible for group={group_name}: '
                    f'checkpoint={payload_runtime_mode!r}, allowed={sorted({str(item) for item in model_allowed_modes})!r}.'
                ),
            )
        elif payload_runtime_mode and model_runtime_mode and str(payload_runtime_mode) != str(model_runtime_mode):
            # When no explicit allow-list exists, enforce exact runtime mode match.
            self._strict_or_warn(
                strict=strict,
                logger=self.logger,
                message=(
                    f'Checkpoint runtime_mode mismatch for group={group_name}: '
                    f'checkpoint={payload_runtime_mode!r}, model={model_runtime_mode!r}.'
                ),
            )

        payload_host_export_version = checkpoint_compat.get('host_export_interface_version')
        if group_name in {'prototype_bank', 'prototype_projector'}:
            model_accepted_interface_versions = model_compat.get('accepted_host_interface_versions', [])
            if (
                payload_host_export_version is not None
                and isinstance(model_accepted_interface_versions, (list, tuple, set))
                and len(model_accepted_interface_versions) > 0
                and str(payload_host_export_version) not in {str(item) for item in model_accepted_interface_versions}
            ):
                self._strict_or_warn(
                    strict=strict,
                    logger=self.logger,
                    message=(
                        f'Checkpoint interface version incompatible for group={group_name}: '
                        f'checkpoint={payload_host_export_version!r}, '
                        f'model accepts={sorted({str(item) for item in model_accepted_interface_versions})!r}.'
                    ),
                )


    def _checkpoint_output_dir(self):
        configured = self.config.get('save', {}).get('dir')
        output_dir = self.save_dir if configured in (None, '', 'null') else str(configured)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _group_is_enabled(self, group_name: str):
        group_cfg = self.config.get('groups', {}).get(group_name, {})
        artifact_cfg = self.config.get('save', {}).get('artifacts', {}).get(group_name, {})
        return bool(group_cfg.get('enabled', False)) and bool(artifact_cfg.get('enabled', False))

    def _selected_groups(self):
        return [group_name for group_name in CHECKPOINT_GROUPS if self._group_is_enabled(group_name)]

    @staticmethod
    def _normalize_metric_row(row_name: Optional[str]) -> Optional[str]:
        if row_name is None:
            return None
        normalized = str(row_name).strip()
        return normalized or None

    def _resolve_group_metric_context(
        self,
        *,
        group_name: str,
        metric_value: Optional[float],
        metric_row: Optional[str],
        metric_display_row: Optional[str],
        metric_source_row: Optional[str],
        authority_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context = authority_context if isinstance(authority_context, dict) else {}
        candidates = context.get('candidates', {}) if isinstance(context.get('candidates', {}), dict) else {}
        row_roles = context.get('row_roles', {}) if isinstance(context.get('row_roles', {}), dict) else {}
        row_metrics = context.get('row_metrics', {}) if isinstance(context.get('row_metrics', {}), dict) else {}
        source_row_from_context = self._normalize_metric_row(context.get('source_row'))
        display_row_from_context = self._normalize_metric_row(context.get('display_row'))
        expected_bucket = self.authority_policy.expected_bucket_for_group(group_name)

        candidate_row = None
        selection_reason = 'input_metric_row'
        if expected_bucket is not None:
            candidate_row = self._normalize_metric_row(candidates.get(expected_bucket))
            if candidate_row is not None:
                selection_reason = f'authority_candidate:{expected_bucket}'

        resolved_source_row = self._normalize_metric_row(metric_source_row) or source_row_from_context
        resolved_display_row = self._normalize_metric_row(metric_display_row) or display_row_from_context

        selected_row = (
            candidate_row
            or resolved_source_row
            or self._normalize_metric_row(metric_row)
            or resolved_display_row
        )
        if selected_row == resolved_source_row and resolved_source_row is not None and candidate_row is None:
            selection_reason = 'source_row'
        elif selected_row == resolved_display_row and resolved_display_row is not None and candidate_row is None:
            selection_reason = 'display_row'

        selected_metric_value = None if metric_value is None else float(metric_value)
        row_metric_payload = row_metrics.get(selected_row) if selected_row is not None else None
        if isinstance(row_metric_payload, dict) and self.metric_name in row_metric_payload:
            selected_metric_value = float(row_metric_payload[self.metric_name])

        validation = self.authority_policy.validate_row_for_group(
            group_name=group_name,
            row_name=selected_row,
            row_roles=row_roles,
        )
        return {
            'group_name': group_name,
            'selected_row': selected_row,
            'selected_metric_value': selected_metric_value,
            'display_row': resolved_display_row,
            'source_row': resolved_source_row,
            'selection_reason': selection_reason,
            'validation': validation,
            'row_roles': row_roles,
        }

    def _enforce_authority_validation(self, resolved_metric_context: Dict[str, Any]) -> bool:
        validation: AuthorityValidationResult = resolved_metric_context['validation']
        if validation.valid:
            return True
        message = (
            'Authority validation failed for checkpoint group=%s row=%s expected_bucket=%s actual_bucket=%s reason=%s'
            % (
                validation.group_name,
                validation.row_name,
                validation.expected_bucket,
                validation.actual_bucket,
                validation.reason,
            )
        )
        if self.authority_policy.warn_only:
            self.logger.warning(message + ' (warn_only=true; save rejected for this group)')
            return False
        if self.authority_policy.strict:
            raise ValueError(message)
        self.logger.error(message + ' (strict=false; save rejected for this group)')
        return False

    def _log_parameterless_group_skip(self, group_name: str, action: str, source_path: Optional[str] = None):
        base_message = (
            'Skipping modular checkpoint %s for group=%s because current model exposes no parameters for this group.'
            % (action, group_name)
        )
        if source_path:
            base_message += f' path={source_path}'
        self.logger.info(base_message)

    def _build_payload(
        self,
        model,
        group_name: str,
        epoch: int,
        global_step: int,
        metric_value: Optional[float],
        metric_row: Optional[str] = None,
        metric_display_row: Optional[str] = None,
        metric_source_row: Optional[str] = None,
        metric_authority_bucket: Optional[str] = None,
        metric_authority_valid: Optional[bool] = None,
        metric_selection_reason: Optional[str] = None,
    ):
        group_state_dict = get_group_state_dict(model, group_name)
        if not group_state_dict:
            return None
        compatibility = self._model_group_compatibility(model, group_name)
        runtime_mode = str(compatibility.get('runtime_mode', getattr(self.args, 'runtime_mode', 'auto')))
        component_name = str(compatibility.get('component_name', GROUP_COMPONENT_NAME.get(group_name, group_name)))
        component_schema_version = compatibility.get('component_schema_version')
        host_export_interface_version = compatibility.get('host_export_interface_version')
        accepted_host_interface_versions = compatibility.get('accepted_host_interface_versions')
        expected_host_score_schema_version = compatibility.get('expected_host_score_schema_version')
        compatible_runtime_modes = compatibility.get('compatible_runtime_modes')
        return {
            'group_name': group_name,
            'state_dict': group_state_dict,
            'epoch': int(epoch),
            'global_step': int(global_step),
            'metric': {
                'name': self.metric_name,
                'value': None if metric_value is None else float(metric_value),
                'mode': self.metric_mode,
                'row': None if metric_row is None else str(metric_row),
                'display_row': None if metric_display_row is None else str(metric_display_row),
                'source_row': None if metric_source_row is None else str(metric_source_row),
                'authority_bucket': None if metric_authority_bucket is None else str(metric_authority_bucket),
                'authority_valid': None if metric_authority_valid is None else bool(metric_authority_valid),
                'selection_reason': None if metric_selection_reason is None else str(metric_selection_reason),
            },
            'metadata': {
                'host_type': str(getattr(self.args, 'host_type', 'clip')),
                'run_name': str(getattr(self.args, 'run_name', '')),
                'training_stage': str(getattr(self.args, 'training_stage', 'joint')),
                'runtime_mode': runtime_mode,
                'component_name': component_name,
                'component_schema_version': None if component_schema_version is None else str(component_schema_version),
                'host_export_interface_version': None if host_export_interface_version is None else str(host_export_interface_version),
                'accepted_host_interface_versions': None
                if not isinstance(accepted_host_interface_versions, (list, tuple, set))
                else [str(item) for item in accepted_host_interface_versions],
                'expected_host_score_schema_version': None
                if expected_host_score_schema_version is None
                else str(expected_host_score_schema_version),
                'compatible_runtime_modes': None
                if not isinstance(compatible_runtime_modes, (list, tuple, set))
                else [str(item) for item in compatible_runtime_modes],
                'authority_bucket_expected': str(GROUP_AUTHORITY_BUCKET.get(group_name, 'unknown')),
                'compatibility': compatibility,
            },
        }

    def _save_group_payload(
        self,
        model,
        group_name: str,
        file_name: str,
        epoch: int,
        global_step: int,
        metric_value: Optional[float],
        metric_row: Optional[str] = None,
        metric_display_row: Optional[str] = None,
        metric_source_row: Optional[str] = None,
        metric_authority_bucket: Optional[str] = None,
        metric_authority_valid: Optional[bool] = None,
        metric_selection_reason: Optional[str] = None,
    ):
        output_path = os.path.join(self._checkpoint_output_dir(), file_name)
        payload = self._build_payload(
            model=model,
            group_name=group_name,
            epoch=epoch,
            global_step=global_step,
            metric_value=metric_value,
            metric_row=metric_row,
            metric_display_row=metric_display_row,
            metric_source_row=metric_source_row,
            metric_authority_bucket=metric_authority_bucket,
            metric_authority_valid=metric_authority_valid,
            metric_selection_reason=metric_selection_reason,
        )
        if payload is None:
            self._log_parameterless_group_skip(group_name=group_name, action='save')
            return
        torch.save(payload, output_path)
        self.logger.info(
            'Saved modular checkpoint group=%s path=%s metric=%s value=%s row=%s source_row=%s display_row=%s authority_bucket=%s authority_valid=%s epoch=%d step=%d',
            group_name,
            output_path,
            self.metric_name,
            metric_value,
            metric_row,
            metric_source_row,
            metric_display_row,
            metric_authority_bucket,
            metric_authority_valid,
            epoch,
            global_step,
        )

    def _rotate_history_if_needed(self, file_name_latest: str, epoch: int):
        keep_last_n = int(self.config.get('save', {}).get('keep_last_n', 1) or 0)
        if keep_last_n <= 0:
            return
        latest_path = os.path.join(self._checkpoint_output_dir(), file_name_latest)
        base_name = os.path.basename(file_name_latest)
        stem, ext = os.path.splitext(base_name)
        history_path = os.path.join(self._checkpoint_output_dir(), f'{stem}_epoch_{int(epoch):04d}{ext}')
        try:
            if os.path.exists(latest_path):
                torch.save(torch.load(latest_path, map_location='cpu'), history_path)
        except Exception:
            return
        pattern = os.path.join(self._checkpoint_output_dir(), f'{stem}_epoch_*{ext}')
        history_files = sorted(glob.glob(pattern))
        if len(history_files) <= keep_last_n:
            return
        stale_files = history_files[: max(0, len(history_files) - keep_last_n)]
        for stale in stale_files:
            try:
                os.remove(stale)
            except OSError:
                pass

    def save_latest(
        self,
        model,
        epoch: int,
        global_step: int,
        metric_value: Optional[float],
        metric_row: Optional[str] = None,
        metric_display_row: Optional[str] = None,
        metric_source_row: Optional[str] = None,
        authority_context: Optional[Dict[str, Any]] = None,
    ):
        if not bool(self.config.get('save', {}).get('save_latest', True)):
            return
        for group_name in self._selected_groups():
            artifact_cfg = self.config.get('save', {}).get('artifacts', {}).get(group_name, {})
            file_name = artifact_cfg.get('filename_latest')
            if not file_name:
                continue
            if not get_group_state_dict(model, group_name):
                self._log_parameterless_group_skip(group_name=group_name, action='save_latest')
                continue
            resolved_metric = self._resolve_group_metric_context(
                group_name=group_name,
                metric_value=metric_value,
                metric_row=metric_row,
                metric_display_row=metric_display_row,
                metric_source_row=metric_source_row,
                authority_context=authority_context,
            )
            if not self._enforce_authority_validation(resolved_metric):
                continue
            validation: AuthorityValidationResult = resolved_metric['validation']
            self.logger.info(
                'Authority validation passed for latest-save group=%s selected_row=%s source_row=%s display_row=%s expected_bucket=%s selection_reason=%s',
                group_name,
                resolved_metric['selected_row'],
                resolved_metric['source_row'],
                resolved_metric['display_row'],
                validation.expected_bucket,
                resolved_metric['selection_reason'],
            )
            self._save_group_payload(
                model=model,
                group_name=group_name,
                file_name=str(file_name),
                epoch=epoch,
                global_step=global_step,
                metric_value=resolved_metric['selected_metric_value'],
                metric_row=resolved_metric['selected_row'],
                metric_display_row=resolved_metric['display_row'],
                metric_source_row=resolved_metric['source_row'],
                metric_authority_bucket=validation.expected_bucket,
                metric_authority_valid=validation.valid,
                metric_selection_reason=resolved_metric['selection_reason'],
            )
            self._rotate_history_if_needed(str(file_name), epoch)

    def _is_metric_improved(self, group_name: str, metric_value: float):
        current_best = self.best_metric_value_by_group.get(str(group_name))
        if current_best is None:
            return True
        if self.metric_mode == 'max':
            return float(metric_value) > float(current_best)
        return float(metric_value) < float(current_best)

    def save_best_if_improved(
        self,
        model,
        epoch: int,
        global_step: int,
        metric_value: Optional[float],
        metric_row: Optional[str] = None,
        metric_display_row: Optional[str] = None,
        metric_source_row: Optional[str] = None,
        authority_context: Optional[Dict[str, Any]] = None,
    ):
        if not bool(self.config.get('save', {}).get('save_best', True)):
            return
        for group_name in self._selected_groups():
            artifact_cfg = self.config.get('save', {}).get('artifacts', {}).get(group_name, {})
            file_name = artifact_cfg.get('filename_best')
            if not file_name:
                continue
            if not get_group_state_dict(model, group_name):
                self._log_parameterless_group_skip(group_name=group_name, action='save_best')
                continue
            resolved_metric = self._resolve_group_metric_context(
                group_name=group_name,
                metric_value=metric_value,
                metric_row=metric_row,
                metric_display_row=metric_display_row,
                metric_source_row=metric_source_row,
                authority_context=authority_context,
            )
            if not self._enforce_authority_validation(resolved_metric):
                continue
            selected_metric_value = resolved_metric['selected_metric_value']
            if selected_metric_value is None:
                continue
            if not self._is_metric_improved(group_name, float(selected_metric_value)):
                continue
            self.best_metric_value_by_group[str(group_name)] = float(selected_metric_value)
            validation: AuthorityValidationResult = resolved_metric['validation']
            self.logger.info(
                'Authority validation passed for best-save group=%s selected_row=%s source_row=%s display_row=%s expected_bucket=%s selection_reason=%s metric_value=%.4f',
                group_name,
                resolved_metric['selected_row'],
                resolved_metric['source_row'],
                resolved_metric['display_row'],
                validation.expected_bucket,
                resolved_metric['selection_reason'],
                float(selected_metric_value),
            )
            self._save_group_payload(
                model=model,
                group_name=group_name,
                file_name=str(file_name),
                epoch=epoch,
                global_step=global_step,
                metric_value=selected_metric_value,
                metric_row=resolved_metric['selected_row'],
                metric_display_row=resolved_metric['display_row'],
                metric_source_row=resolved_metric['source_row'],
                metric_authority_bucket=validation.expected_bucket,
                metric_authority_valid=validation.valid,
                metric_selection_reason=resolved_metric['selection_reason'],
            )

    def has_enabled_group_loading(self):
        if not bool(self.config.get('load', {}).get('enabled', False)):
            return False
        for group_name in CHECKPOINT_GROUPS:
            source_cfg = self.config.get('load', {}).get('sources', {}).get(group_name, {})
            if bool(source_cfg.get('enabled', False)) and source_cfg.get('path'):
                return True
        return False

    def get_group_load_source(self, group_name: str):
        if group_name not in CHECKPOINT_GROUPS:
            return {}
        return self.config.get('load', {}).get('sources', {}).get(group_name, {}) or {}

    def has_group_loading_enabled(self, group_name: str):
        if group_name not in CHECKPOINT_GROUPS:
            return False
        if not bool(self.config.get('load', {}).get('enabled', False)):
            return False
        source_cfg = self.get_group_load_source(group_name)
        return bool(source_cfg.get('enabled', False)) and bool(source_cfg.get('path'))

    def load_configured_groups(self, model):
        if not bool(self.config.get('load', {}).get('enabled', False)):
            self.logger.info('Modular checkpoint loading disabled (checkpointing.load.enabled=false).')
            return

        strict = bool(self.config.get('load', {}).get('strict', True))
        checked_groups = 0
        groups_with_issues = 0
        skipped_parameterless_groups = 0
        for group_name in CHECKPOINT_GROUPS:
            source_cfg = self.config.get('load', {}).get('sources', {}).get(group_name, {})
            if not bool(source_cfg.get('enabled', False)):
                continue
            source_path = source_cfg.get('path')
            if not source_path:
                self.logger.info('Skipping checkpoint group load for %s (path is empty).', group_name)
                continue

            # Explicitly skip loading groups that have no parameters in the current model.
            if not get_group_state_dict(model, group_name):
                skipped_parameterless_groups += 1
                self._log_parameterless_group_skip(
                    group_name=group_name,
                    action='load',
                    source_path=str(source_path),
                )
                continue

            checked_groups += 1
            self.logger.info(
                'Loading checkpoint group=%s path=%s strict=%s',
                group_name,
                source_path,
                strict,
            )
            payload = torch.load(str(source_path), map_location='cpu')
            state_dict, payload_type = _extract_state_dict_from_checkpoint_payload(payload)
            metadata = payload.get('metadata', {}) if isinstance(payload, dict) else {}
            if isinstance(metadata, dict):
                expected_component = GROUP_COMPONENT_NAME.get(group_name)
                payload_component = metadata.get('component_name')
                if expected_component and payload_component and str(payload_component) != str(expected_component):
                    message = (
                        f'Checkpoint component mismatch for group={group_name}: '
                        f'expected {expected_component!r}, got {payload_component!r}.'
                    )
                    if strict:
                        raise RuntimeError(message)
                    self.logger.warning(message)
            self._validate_checkpoint_compatibility(
                group_name=group_name,
                payload_type=payload_type,
                payload_metadata=metadata if isinstance(metadata, dict) else {},
                model=model,
                strict=strict,
            )
            host_type = str(getattr(self.args, 'host_type', 'clip')).lower()
            compatible_state_dict, compat_stats = _prepare_state_dict_for_group_compatibility(
                group_name=group_name,
                state_dict=state_dict,
                model=model,
                host_type=host_type,
            )
            if group_name == 'host' and payload_type != 'modular_group_payload':
                self.logger.info(
                    'Host load compatibility mode active: payload_type=%s host_type=%s legacy_renamed=%d host_head_prefixed=%d',
                    payload_type,
                    host_type,
                    compat_stats['legacy_renamed'],
                    compat_stats['host_head_prefixed'],
                )

            load_summary = load_group_state_dict(
                model=model,
                group_name=group_name,
                state_dict=compatible_state_dict,
                strict=strict,
            )
            self.logger.info(
                'Loaded checkpoint group=%s loaded_keys=%d missing_keys=%d unexpected_keys=%d shape_mismatches=%d',
                group_name,
                len(load_summary['loaded_keys']),
                len(load_summary['missing_keys']),
                len(load_summary['unexpected_keys']),
                len(load_summary['shape_mismatches']),
            )

            missing_keys = list(load_summary.get('missing_keys', []))
            unexpected_keys = list(load_summary.get('unexpected_keys', []))
            shape_mismatches = list(load_summary.get('shape_mismatches', []))
            ignored_keys = sorted(set(unexpected_keys + [entry.split(':', 1)[0] for entry in shape_mismatches]))
            has_missing = bool(missing_keys)
            has_ignored = bool(ignored_keys)
            has_unexpected = bool(unexpected_keys)

            if has_missing or has_ignored or has_unexpected:
                groups_with_issues += 1
                self.logger.warning(
                    'Group %s load diagnostics: has_missing=%s has_ignored=%s has_unexpected=%s',
                    group_name,
                    has_missing,
                    has_ignored,
                    has_unexpected,
                )
                self.logger.warning('Group %s loaded keys (all): %s', group_name, load_summary.get('loaded_keys', []))
                if has_missing:
                    self.logger.warning('Group %s missing model keys (all): %s', group_name, missing_keys)
                if has_ignored:
                    self.logger.warning('Group %s ignored checkpoint keys (all): %s', group_name, ignored_keys)
                if has_unexpected:
                    self.logger.warning('Group %s unexpected checkpoint keys (all): %s', group_name, unexpected_keys)

        if checked_groups == 0:
            self.logger.warning(
                'Modular checkpoint load enabled but no group had both enabled=true and a non-empty source path.'
            )
        else:
            self.logger.info(
                'Modular checkpoint load summary: checked_groups=%d groups_with_issues=%d groups_clean=%d skipped_parameterless_groups=%d',
                checked_groups,
                groups_with_issues,
                checked_groups - groups_with_issues,
                skipped_parameterless_groups,
            )
