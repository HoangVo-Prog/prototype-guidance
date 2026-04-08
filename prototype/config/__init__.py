"""Integration config schema and defaults."""

from .defaults import DEFAULT_CONFIG
from .schema import IntegrationConfig, validate_integration_config

__all__ = ["DEFAULT_CONFIG", "IntegrationConfig", "validate_integration_config"]
