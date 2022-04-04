"""This module defines how to configure Wicker from the user environment
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Optional


@dataclasses.dataclass(frozen=True)
class WickerWandBConfig:
    wandb_base_url: str
    wandb_api_key: str

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerWandBConfig:
        # only load them if they exist, otherwise leave out
        return cls(
            wandb_api_key=data.get("wandb_api_key", None),
            wandb_base_url=data.get("wandb_base_url", None),
        )


@dataclasses.dataclass(frozen=True)
class WickerAwsS3Config:
    s3_datasets_path: str
    region: str

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerAwsS3Config:
        return cls(
            s3_datasets_path=data["s3_datasets_path"],
            region=data["region"],
        )


@dataclasses.dataclass(frozen=True)
class WickerConfig:
    raw: Dict[str, Any]
    aws_s3_config: WickerAwsS3Config
    wandb_config: WickerWandBConfig

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerConfig:
        return cls(
            raw=data,
            aws_s3_config=WickerAwsS3Config.from_json(data["aws_s3_config"]),
            wandb_config=WickerWandBConfig.from_json(data.get("wandb_config", {})),
        )


_CONFIG: Optional[WickerConfig] = None
_PREV_WICKER_CONFIG_PATH: Optional[str] = None


def get_config() -> WickerConfig:
    """Retrieves the Wicker config for the current process"""
    global _CONFIG
    global _PREV_WICKER_CONFIG_PATH

    loc_wicker_config_path = os.getenv("WICKER_CONFIG_PATH", os.path.expanduser("~/wickerconfig.json"))
    # Hack to get around having tests need to specify config hermetically
    # ToDo: Abstract the config additionally to clean this up
    if _CONFIG is None or loc_wicker_config_path != _PREV_WICKER_CONFIG_PATH:
        _PREV_WICKER_CONFIG_PATH = loc_wicker_config_path
        with open(loc_wicker_config_path, "r") as f:
            _CONFIG = WickerConfig.from_json(json.load(f))
    return _CONFIG
