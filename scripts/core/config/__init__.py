"""OPC configuration package.

Public API:
    from scripts.core.config import get_config, reset_config

    cfg = get_config()
    cfg.dedup.threshold
"""

from scripts.core.config.handlers import get_config, reset_config
from scripts.core.config.models import OPCConfig

__all__ = ["get_config", "reset_config", "OPCConfig"]
