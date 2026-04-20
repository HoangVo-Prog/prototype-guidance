import pytest

from utils.config import validate_config_data
from utils.options import get_args


def test_removed_loss_keys_are_rejected_in_loss_section():
    with pytest.raises(ValueError, match="loss\\.use_loss_ret was removed"):
        validate_config_data(
            {
                "loss": {
                    "use_loss_ret": True,
                }
            }
        )


def test_removed_loss_keys_are_rejected_in_objectives_section():
    with pytest.raises(ValueError, match="objectives\\.objectives\\.use_loss_proxy_image was removed"):
        validate_config_data(
            {
                "objectives": {
                    "objectives": {
                        "use_loss_proxy_image": True,
                    }
                }
            }
        )


def test_removed_align_weight_ret_keys_are_rejected_in_loss_section():
    with pytest.raises(ValueError, match="loss\\.lambda_align was removed"):
        validate_config_data(
            {
                "loss": {
                    "lambda_align": 1.0,
                }
            }
        )

    with pytest.raises(ValueError, match="loss\\.use_loss_weight_ret was removed"):
        validate_config_data(
            {
                "loss": {
                    "use_loss_weight_ret": True,
                }
            }
        )


def test_removed_loss_cli_flags_fail_fast():
    with pytest.raises(ValueError, match="loss_ret was removed"):
        get_args(["--use_loss_ret=true"])

    with pytest.raises(ValueError, match="loss_align was removed"):
        get_args(["--lambda_align=1.0"])
