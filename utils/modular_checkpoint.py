from __future__ import annotations

import copy
import glob
import os
from typing import Dict, Optional, Tuple

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
        'fusion': {'enabled': True},
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
            'fusion': {
                'enabled': True,
                'filename_latest': 'checkpoint_fusion_latest.pth',
                'filename_best': 'checkpoint_fusion_best.pth',
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
            'fusion': {'enabled': False, 'path': None},
        },
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
                not key.startswith(('host_head.', 'base_model.', 'prototype_head.'))
                and ('host_head.' + key) in model_keys
            ):
                key = 'host_head.' + key
                stats['host_head_prefixed'] += 1

        prepared[key] = value

    return prepared, stats


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
        self.best_metric_value = None

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

    def _log_parameterless_group_skip(self, group_name: str, action: str, source_path: Optional[str] = None):
        base_message = (
            'Skipping modular checkpoint %s for group=%s because current model exposes no parameters for this group.'
            % (action, group_name)
        )
        if source_path:
            base_message += f' path={source_path}'
        self.logger.info(base_message)

    def _build_payload(self, model, group_name: str, epoch: int, global_step: int, metric_value: Optional[float]):
        group_state_dict = get_group_state_dict(model, group_name)
        if not group_state_dict:
            return None
        return {
            'group_name': group_name,
            'state_dict': group_state_dict,
            'epoch': int(epoch),
            'global_step': int(global_step),
            'metric': {
                'name': self.metric_name,
                'value': None if metric_value is None else float(metric_value),
                'mode': self.metric_mode,
            },
            'metadata': {
                'host_type': str(getattr(self.args, 'host_type', 'clip')),
                'run_name': str(getattr(self.args, 'run_name', '')),
                'training_stage': str(getattr(self.args, 'training_stage', 'joint')),
            },
        }

    def _save_group_payload(self, model, group_name: str, file_name: str, epoch: int, global_step: int, metric_value: Optional[float]):
        output_path = os.path.join(self._checkpoint_output_dir(), file_name)
        payload = self._build_payload(
            model=model,
            group_name=group_name,
            epoch=epoch,
            global_step=global_step,
            metric_value=metric_value,
        )
        if payload is None:
            self._log_parameterless_group_skip(group_name=group_name, action='save')
            return
        torch.save(payload, output_path)
        self.logger.info(
            'Saved modular checkpoint group=%s path=%s metric=%s value=%s epoch=%d step=%d',
            group_name,
            output_path,
            self.metric_name,
            metric_value,
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

    def save_latest(self, model, epoch: int, global_step: int, metric_value: Optional[float]):
        if not bool(self.config.get('save', {}).get('save_latest', True)):
            return
        for group_name in self._selected_groups():
            artifact_cfg = self.config.get('save', {}).get('artifacts', {}).get(group_name, {})
            file_name = artifact_cfg.get('filename_latest')
            if not file_name:
                continue
            self._save_group_payload(
                model=model,
                group_name=group_name,
                file_name=str(file_name),
                epoch=epoch,
                global_step=global_step,
                metric_value=metric_value,
            )
            self._rotate_history_if_needed(str(file_name), epoch)

    def _is_metric_improved(self, metric_value: float):
        if self.best_metric_value is None:
            return True
        if self.metric_mode == 'max':
            return float(metric_value) > float(self.best_metric_value)
        return float(metric_value) < float(self.best_metric_value)

    def save_best_if_improved(self, model, epoch: int, global_step: int, metric_value: Optional[float]):
        if not bool(self.config.get('save', {}).get('save_best', True)):
            return
        if metric_value is None:
            return
        if not self._is_metric_improved(metric_value):
            return
        self.best_metric_value = float(metric_value)
        for group_name in self._selected_groups():
            artifact_cfg = self.config.get('save', {}).get('artifacts', {}).get(group_name, {})
            file_name = artifact_cfg.get('filename_best')
            if not file_name:
                continue
            self._save_group_payload(
                model=model,
                group_name=group_name,
                file_name=str(file_name),
                epoch=epoch,
                global_step=global_step,
                metric_value=metric_value,
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
            self.logger.info('Group %s loaded keys: %s', group_name, load_summary['loaded_keys'])
            self.logger.info('Group %s missing keys: %s', group_name, load_summary['missing_keys'])
            self.logger.info('Group %s unexpected keys: %s', group_name, load_summary['unexpected_keys'])
            self.logger.info('Group %s shape mismatches: %s', group_name, load_summary['shape_mismatches'])

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
                if has_missing:
                    self.logger.warning('Group %s missing model keys (all): %s', group_name, missing_keys)
                if has_ignored:
                    self.logger.warning('Group %s ignored checkpoint keys (all): %s', group_name, ignored_keys)
                if has_unexpected:
                    self.logger.warning('Group %s unexpected checkpoint keys (all): %s', group_name, unexpected_keys)
            else:
                self.logger.info(
                    'Group %s load diagnostics: has_missing=false has_ignored=false has_unexpected=false',
                    group_name,
                )

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
