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
class BotoS3Config:
    max_pool_connections: int
    read_timeout: int
    connect_timeout: int

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> BotoS3Config:
        return cls(
            max_pool_connections=data["max_pool_connections"],
            read_timeout=data["read_timeout"],
            connect_timeout=data["connect_timeout"],
        )


@dataclasses.dataclass(frozen=True)
class WickerAwsS3Config:
    s3_datasets_path: str
    region: str
    boto_config: BotoS3Config
    store_concatenated_bytes_files_in_dataset: bool = False

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerAwsS3Config:
        return cls(
            s3_datasets_path=data["s3_datasets_path"],
            region=data["region"],
            boto_config=BotoS3Config.from_json(data["boto_config"]),
            store_concatenated_bytes_files_in_dataset=data.get("store_concatenated_bytes_files_in_dataset", False),
        )


@dataclasses.dataclass(frozen=True)
class S3StorageConfig:
    retries: int
    timeout: int

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> S3StorageConfig:
        return cls(
            retries=data["retries"],
            timeout=data["timeout"],
        )


@dataclasses.dataclass(frozen=True)
class WickerConfig:
    raw: Dict[str, Any]
    aws_s3_config: WickerAwsS3Config
    wandb_config: WickerWandBConfig
    s3_storage_config: S3StorageConfig

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerConfig:
        return cls(
            raw=data,
            aws_s3_config=WickerAwsS3Config.from_json(data["aws_s3_config"]),
            wandb_config=WickerWandBConfig.from_json(data.get("wandb_config", {})),
            s3_storage_config=S3StorageConfig.from_json(data["s3_storage_config"]),
        )


def get_config() -> WickerConfig:
    """Retrieves the Wicker config for the current process"""

    wicker_config_path = os.getenv("WICKER_CONFIG_PATH", os.path.expanduser("~/wickerconfig.json"))
    with open(wicker_config_path, "r") as f:
        config = WickerConfig.from_json(json.load(f))
    return config
