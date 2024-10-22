"""This module defines how to configure Wicker from the user environment
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict

AWS_S3_CONFIG = "aws_s3_config"
FILESYSTEM_CONFIG = "filesystem_config"


@dataclasses.dataclass(frozen=True)
class WickerWandBConfig:
    wandb_base_url: str
    wandb_api_key: str

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerWandBConfig:
        return cls(
            wandb_api_key=data.get("wandb_api_key", None),
            wandb_base_url=data.get("wandb_base_url", None),
        )


@dataclasses.dataclass(frozen=True)
class BotoS3Config:
    max_pool_connections: int
    read_timeout_s: int
    connect_timeout_s: int

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> BotoS3Config:
        return cls(
            max_pool_connections=data.get("max_pool_connections", 0),
            read_timeout_s=data.get("read_timeout_s", 0),
            connect_timeout_s=data.get("connect_timeout_s", 0),
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
            s3_datasets_path=data.get("s3_datasets_path", ""),
            region=data.get("region", ""),
            boto_config=BotoS3Config.from_json(data.get("boto_config", {})),
            store_concatenated_bytes_files_in_dataset=data.get("store_concatenated_bytes_files_in_dataset", False),
        )


@dataclasses.dataclass(frozen=True)
class WickerFileSystemConfig:
    prefix_replace_path: str
    root_datasets_path: str

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerFileSystemConfig:
        return cls(
            prefix_replace_path=data.get("prefix_replace_path", ""),
            root_datasets_path=data.get("root_datasets_path", ""),
        )


@dataclasses.dataclass(frozen=True)
class StorageDownloadConfig:
    retries: int
    timeout: int
    retry_backoff: int
    retry_delay_s: int

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> StorageDownloadConfig:
        return cls(
            retries=data.get("retries", 0),
            timeout=data.get("timeout", 0),
            retry_backoff=data.get("retry_backoff", 0),
            retry_delay_s=data.get("retry_delay_s", 0),
        )


@dataclasses.dataclass()
class WickerConfig:
    raw: Dict[str, Any]
    aws_s3_config: WickerAwsS3Config
    filesystem_config: WickerFileSystemConfig
    storage_download_config: StorageDownloadConfig
    wandb_config: WickerWandBConfig

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerConfig:
        return cls(
            raw=data,
            aws_s3_config=WickerAwsS3Config.from_json(data.get(AWS_S3_CONFIG, {})),
            filesystem_config=WickerFileSystemConfig.from_json(data.get(FILESYSTEM_CONFIG, {})),
            storage_download_config=StorageDownloadConfig.from_json(data.get("storage_download_config", {})),
            wandb_config=WickerWandBConfig.from_json(data.get("wandb_config", {})),
        )


def get_config() -> WickerConfig:
    """Retrieves the Wicker config for the current process"""

    wicker_config_path = os.getenv("WICKER_CONFIG_PATH", os.path.expanduser("~/wickerconfig.json"))
    with open(wicker_config_path, "r") as f:
        config = WickerConfig.from_json(json.load(f))
    return config
