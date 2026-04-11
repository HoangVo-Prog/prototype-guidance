from __future__ import annotations

import copy
import glob
import os
from typing import Dict, Optional

import torch

from utils.module_group_registry import CHECKPOINT_GROUPS, get_group_state_dict, load_group_state_dict


DEFAULT_CHECKPOINTING_CONFIG = {
    'metric': {
        'name': 'R1',
        'mode': 'max',
    },
    'groups': {
        'host': {'enabled': True},
        'prototype_bank': {'enabled': True},
        'prototype_projector': {'enabled': True},
        'routing': {'enabled': True},
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
            'routing': {
                'enabled': True,
                'filename_latest': 'checkpoint_routing_latest.pth',
                'filename_best': 'checkpoint_routing_best.pth',
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
            'routing': {'enabled': False, 'path': None},
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
            self.logger.info('Skipping modular checkpoint save for group=%s (no parameters matched).', group_name)
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

    def load_configured_groups(self, model):
        if not bool(self.config.get('load', {}).get('enabled', False)):
            self.logger.info('Modular checkpoint loading disabled (checkpointing.load.enabled=false).')
            return

        strict = bool(self.config.get('load', {}).get('strict', True))
        for group_name in CHECKPOINT_GROUPS:
            source_cfg = self.config.get('load', {}).get('sources', {}).get(group_name, {})
            if not bool(source_cfg.get('enabled', False)):
                continue
            source_path = source_cfg.get('path')
            if not source_path:
                self.logger.info('Skipping checkpoint group load for %s (path is empty).', group_name)
                continue

            self.logger.info(
                'Loading checkpoint group=%s path=%s strict=%s',
                group_name,
                source_path,
                strict,
            )
            payload = torch.load(str(source_path), map_location='cpu')
            state_dict = payload.get('state_dict') if isinstance(payload, dict) and isinstance(payload.get('state_dict'), dict) else payload
            if not isinstance(state_dict, dict):
                raise ValueError(f'Checkpoint payload at {source_path} does not contain a valid state_dict mapping.')

            load_summary = load_group_state_dict(
                model=model,
                group_name=group_name,
                state_dict=state_dict,
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
