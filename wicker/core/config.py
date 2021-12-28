"""This module defines how to configure Wicker from the user environment
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Optional

WICKER_CONFIG_PATH_ENVVAR = "WICKER_CONFIG_PATH"
WICKER_CONFIG_PATH = os.getenv("WICKER_CONFIG_PATH", os.path.expanduser("~/.wickerconfig"))


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

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerConfig:
        return cls(
            raw=data,
            aws_s3_config=WickerAwsS3Config.from_json(data["aws_s3_config"]),
        )


_CONFIG: Optional[WickerConfig] = None


def get_config() -> WickerConfig:
    """Retrieves the Wicker config for the current process"""
    global _CONFIG
    if _CONFIG is None:
        with open(WICKER_CONFIG_PATH, "r") as f:
            _CONFIG = WickerConfig.from_json(json.load(f))
    return _CONFIG
