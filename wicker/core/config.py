"""This module defines how to configure Wicker from the user environment
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict


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


def get_config() -> WickerConfig:
    """Retrieves the Wicker config for the current process"""

    wicker_config_path = os.getenv("WICKER_CONFIG_PATH", os.path.expanduser("~/wickerconfig.json"))
    with open(wicker_config_path, "r") as f:
        config = WickerConfig.from_json(json.load(f))
    return config
