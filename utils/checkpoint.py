# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import copy
import datetime as _dt
import logging
import os
import random
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch


class Checkpointer:
    FORMAT_VERSION = 2

    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    @staticmethod
    def _is_plain_state_dict(payload: Any) -> bool:
        if not isinstance(payload, dict) or not payload:
            return False
        if any(not isinstance(key, str) for key in payload.keys()):
            return False
        tensor_like_count = 0
        for value in payload.values():
            if isinstance(value, torch.Tensor):
                tensor_like_count += 1
                continue
            if isinstance(value, (int, float, str, bool, type(None))):
                continue
            if isinstance(value, (list, tuple, dict)):
                return False
            return False
        return tensor_like_count > 0

    @classmethod
    def _resolve_model_state_dict(cls, checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            model_state = checkpoint.get("model")
            if isinstance(model_state, dict):
                return model_state
            state_dict = checkpoint.get("state_dict")
            if isinstance(state_dict, dict):
                return state_dict
            if cls._is_plain_state_dict(checkpoint):
                return checkpoint
        raise ValueError(
            "Checkpoint payload does not contain a model state dict under `model` or `state_dict`, "
            "and is not a plain state_dict mapping."
        )

    @staticmethod
    def _capture_rng_state() -> Dict[str, Any]:
        rng_state: Dict[str, Any] = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                rng_state["torch_cuda"] = None
        else:
            rng_state["torch_cuda"] = None
        return rng_state

    @staticmethod
    def _restore_rng_state(rng_state: Dict[str, Any], logger) -> bool:
        if not isinstance(rng_state, dict):
            return False
        restored_any = False
        try:
            if "python_random" in rng_state and rng_state["python_random"] is not None:
                random.setstate(rng_state["python_random"])
                restored_any = True
        except Exception as exc:
            logger.warning("Failed to restore python RNG state: %s", exc)
        try:
            if "numpy_random" in rng_state and rng_state["numpy_random"] is not None:
                np.random.set_state(rng_state["numpy_random"])
                restored_any = True
        except Exception as exc:
            logger.warning("Failed to restore numpy RNG state: %s", exc)
        try:
            if "torch_cpu" in rng_state and rng_state["torch_cpu"] is not None:
                torch.set_rng_state(rng_state["torch_cpu"])
                restored_any = True
        except Exception as exc:
            logger.warning("Failed to restore torch CPU RNG state: %s", exc)
        cuda_state = rng_state.get("torch_cuda")
        if cuda_state is not None:
            if not torch.cuda.is_available():
                logger.warning("Checkpoint contains CUDA RNG state but CUDA is not available in this process.")
            else:
                try:
                    if isinstance(cuda_state, (list, tuple)):
                        torch.cuda.set_rng_state_all(list(cuda_state))
                        restored_any = True
                    elif isinstance(cuda_state, torch.Tensor):
                        torch.cuda.set_rng_state(cuda_state)
                        restored_any = True
                except Exception as exc:
                    logger.warning("Failed to restore torch CUDA RNG state: %s", exc)
        return restored_any

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def save_training_checkpoint(
        self,
        name: str,
        *,
        epoch: int,
        global_step: int,
        iteration_in_epoch: int = 0,
        checkpoint_kind: str = "training_resume",
        metric_name: str = "R1",
        metric_mode: str = "max",
        best_metric_state: Optional[Dict[str, Any]] = None,
        latest_metric_state: Optional[Dict[str, Any]] = None,
        early_stopping_state: Optional[Dict[str, Any]] = None,
        modular_best_metric_value_by_group: Optional[Dict[str, Any]] = None,
        scaler=None,
        config_snapshot: Optional[Dict[str, Any]] = None,
        include_rng_state: bool = True,
        additional_training_state: Optional[Dict[str, Any]] = None,
        prototype_runtime_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.save_dir:
            return ""
        if not self.save_to_disk:
            return ""

        epoch_completed = int(epoch)
        next_epoch = epoch_completed + 1
        training_state: Dict[str, Any] = {
            "epoch_completed": epoch_completed,
            "epoch": epoch_completed,
            "next_epoch": next_epoch,
            "global_step": int(global_step),
            "iteration_in_epoch": int(iteration_in_epoch),
            "metric_name": str(metric_name),
            "metric_mode": str(metric_mode),
        }
        if isinstance(best_metric_state, dict):
            training_state["best_metric_state"] = copy.deepcopy(best_metric_state)
        if isinstance(latest_metric_state, dict):
            training_state["latest_metric_state"] = copy.deepcopy(latest_metric_state)
        if isinstance(early_stopping_state, dict):
            training_state["early_stopping_state"] = copy.deepcopy(early_stopping_state)
        if isinstance(modular_best_metric_value_by_group, dict):
            training_state["modular_best_metric_value_by_group"] = {
                str(key): float(value) for key, value in modular_best_metric_value_by_group.items()
            }
        if isinstance(additional_training_state, dict):
            training_state.update(copy.deepcopy(additional_training_state))

        payload: Dict[str, Any] = {
            "format_version": int(self.FORMAT_VERSION),
            "checkpoint_kind": str(checkpoint_kind),
            "saved_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "model": self.model.state_dict(),
            # Keep top-level epoch/global_step for compatibility with legacy resume callers.
            "epoch": int(next_epoch),
            "global_step": int(global_step),
            "training_state": training_state,
        }
        if self.optimizer is not None:
            payload["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()
        scaler_state_dict = None
        if scaler is not None and hasattr(scaler, "state_dict"):
            try:
                scaler_state_dict = scaler.state_dict()
            except Exception:
                scaler_state_dict = None
        if scaler_state_dict:
            payload["scaler"] = scaler_state_dict
        if include_rng_state:
            payload["rng_state"] = self._capture_rng_state()
        if isinstance(prototype_runtime_state, dict) and prototype_runtime_state:
            payload["prototype_runtime_state"] = copy.deepcopy(prototype_runtime_state)
        if isinstance(config_snapshot, dict) and config_snapshot:
            payload["config_snapshot"] = copy.deepcopy(config_snapshot)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        torch.save(payload, save_file)
        self.logger.info(
            (
                "Saved resumable checkpoint path=%s kind=%s epoch_completed=%d next_epoch=%d global_step=%d "
                "metric_name=%s best_metric=%s selected_row=%s optimizer_saved=%s scheduler_saved=%s scaler_saved=%s rng_saved=%s"
            ),
            save_file,
            checkpoint_kind,
            epoch_completed,
            next_epoch,
            int(global_step),
            str(metric_name),
            None if not isinstance(best_metric_state, dict) else best_metric_state.get("value"),
            None if not isinstance(best_metric_state, dict) else best_metric_state.get("selected_row"),
            self.optimizer is not None,
            self.scheduler is not None,
            bool(scaler_state_dict),
            bool(include_rng_state),
        )
        return save_file

    def load(self, f=None):
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if isinstance(checkpoint.get("prototype_runtime_state"), dict) and hasattr(self.model, "load_prototype_runtime_state"):
            try:
                self.model.load_prototype_runtime_state(checkpoint.get("prototype_runtime_state"))
                self.logger.info("Restored prototype runtime state from checkpoint.")
            except Exception as exc:
                warnings.append(f"prototype_runtime_state_restore_failed:{exc}")
        return checkpoint

    def resume_training(
        self,
        f=None,
        *,
        strict: bool = False,
        restore_rng: bool = True,
        scaler=None,
    ):
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            raise IOError(f"No Checkpoint file found on {f}")
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        optimizer_restored = False
        scheduler_restored = False
        scaler_restored = False
        rng_restored = False
        warnings = []

        optimizer_state = checkpoint.get("optimizer")
        if self.optimizer is not None:
            if isinstance(optimizer_state, dict):
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(optimizer_state)
                optimizer_restored = True
            else:
                warnings.append("optimizer_state_missing")

        scheduler_state = checkpoint.get("scheduler")
        if self.scheduler is not None:
            if isinstance(scheduler_state, dict):
                self.logger.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(scheduler_state)
                scheduler_restored = True
            else:
                warnings.append("scheduler_state_missing")

        scaler_state = checkpoint.get("scaler")
        if scaler is not None and isinstance(scaler_state, dict) and hasattr(scaler, "load_state_dict"):
            try:
                scaler.load_state_dict(scaler_state)
                scaler_restored = True
            except Exception as exc:
                warnings.append(f"scaler_state_restore_failed:{exc}")
        elif scaler is not None and scaler_state is None:
            warnings.append("scaler_state_missing")

        if restore_rng:
            if isinstance(checkpoint.get("rng_state"), dict):
                rng_restored = self._restore_rng_state(checkpoint.get("rng_state"), self.logger)
                if not rng_restored:
                    warnings.append("rng_state_restore_failed")
            else:
                warnings.append("rng_state_missing")

        training_state = checkpoint.get("training_state")
        has_training_state = isinstance(training_state, dict)
        if not has_training_state:
            training_state = {}
            if "epoch" in checkpoint:
                legacy_epoch = int(checkpoint.get("epoch", 1))
                training_state["epoch"] = max(legacy_epoch, 0)
                training_state["next_epoch"] = max(legacy_epoch + 1, 1)
            if "global_step" in checkpoint:
                training_state["global_step"] = int(checkpoint.get("global_step", 0))
            warnings.append("training_state_missing")

        if strict:
            missing = []
            if not optimizer_restored:
                missing.append("optimizer")
            if not scheduler_restored:
                missing.append("scheduler")
            if not has_training_state:
                missing.append("training_state")
            if missing:
                raise RuntimeError(
                    "Strict resume requested but checkpoint is missing required resume fields: {}".format(
                        ", ".join(missing)
                    )
                )

        if warnings:
            self.logger.warning(
                "Resume checkpoint compatibility fallback activated for %s: %s",
                f,
                warnings,
            )

        next_epoch = training_state.get("next_epoch")
        if next_epoch is None:
            # Legacy fallback: old checkpoints stored the completed/current epoch at top-level.
            legacy_epoch = int(checkpoint.get("epoch", training_state.get("epoch", 0)) or 0)
            next_epoch = max(legacy_epoch + 1, 1)
        next_epoch = max(int(next_epoch), 1)
        global_step = int(training_state.get("global_step", checkpoint.get("global_step", 0) or 0))

        return {
            "checkpoint": checkpoint,
            "training_state": training_state,
            "start_epoch": next_epoch,
            "global_step": global_step,
            "scaler_state_dict": scaler_state if isinstance(scaler_state, dict) else None,
            "optimizer_restored": optimizer_restored,
            "scheduler_restored": scheduler_restored,
            "scaler_restored": scaler_restored,
            "rng_restored": rng_restored,
            "warnings": warnings,
        }

    def resume(self, f=None):
        resume_bundle = self.resume_training(f, strict=False, restore_rng=False, scaler=None)
        checkpoint = dict(resume_bundle.get("checkpoint", {}) or {})
        checkpoint.pop("model", None)
        checkpoint.pop("optimizer", None)
        checkpoint.pop("scheduler", None)
        return checkpoint

    def _load_file(self, f):
        load_kwargs = {"map_location": torch.device("cpu")}
        # PyTorch 2.6 changed torch.load default weights_only=True. Our internal
        # training-resume checkpoints intentionally store non-tensor Python state
        # (optimizer/scheduler metadata, RNG snapshots), so we must opt out here.
        try:
            return torch.load(f, weights_only=False, **load_kwargs)
        except TypeError:
            # Older torch versions do not support the weights_only argument.
            return torch.load(f, **load_kwargs)

    def _load_model(self, checkpoint, except_keys=None):
        model_state_dict = self._resolve_model_state_dict(checkpoint)
        load_state_dict(self.model, model_state_dict, except_keys)


def check_key(key, except_keys):
    if except_keys is None:
        return False
    else:
        for except_key in except_keys:
            if except_key in key:
                return True
        return False


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys=None):
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger("PersonSearch.checkpoint")
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if check_key(key, except_keys):
            continue
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, except_keys=None):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys)

    # use strict loading
    model.load_state_dict(model_state_dict)
