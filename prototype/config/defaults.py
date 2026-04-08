"""Default integration config aligned with stage-control contract."""

from __future__ import annotations


DEFAULT_CONFIG = {
    "train_mode": "itself",
    "prototype": {
        "enabled": False,
        "num_prototypes": 64,
        "dim": 512,
        "contextualization": {
            "enabled": False,
            "type": "self_attention",
        },
        "temperatures": {
            "routing": 0.07,
            "basis": 0.07,
            "teacher": 0.07,
            "retrieval": 0.07,
        },
        "regularization": {
            "diversity": {
                "enabled": False,
                "weight": 0.0,
            },
            "balance": {
                "enabled": False,
                "weight": 0.0,
            },
        },
    },
    "fusion": {
        "enabled": True,
        "lambda_f": 0.0,
    },
    "host": {
        "lambda_s": 0.5,
    },
    "training": {
        "stage": "stage0",
        "calibration_only": False,
        "freeze": {
            "host": False,
            "prototype": True,
            "host_allowlist": [],
            "prototype_policy": "freeze_all",
        },
        "initialization": {
            "clip_backbone_source": None,
            "stage1_prototype_checkpoint": None,
            "host_checkpoint": None,
            "host_checkpoint_compatible": False,
        },
    },
    "loss": {
        "host_enabled": True,
        "prototype_ret_enabled": False,
        "prototype_diag_enabled": False,
        "prototype_div_enabled": False,
        "prototype_bal_enabled": False,
    },
}
