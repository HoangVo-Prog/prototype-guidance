"""Default config placeholders for integration runtime."""

from __future__ import annotations


DEFAULT_CONFIG = {
    "train_mode": "itself",
    "prototype": {
        "enabled": False,
        "num_prototypes": 64,
        "dim": 512,
        "contextualization_enabled": False,
        "routing_temperature": 0.07,
        "basis_temperature": 0.07,
        "teacher_temperature": 0.07,
        "retrieval_temperature": 0.07,
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
        "freeze_host": False,
        "freeze_prototype": True,
    },
    "loss": {
        "host_enabled": True,
        "prototype_ret_enabled": False,
        "prototype_diag_enabled": False,
        "prototype_div_enabled": False,
        "prototype_bal_enabled": False,
    },
}
