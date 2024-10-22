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
    read_timeout_s: int
    connect_timeout_s: int

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> BotoS3Config:
        return cls(
            max_pool_connections=data["max_pool_connections"],
            read_timeout_s=data["read_timeout_s"],
            connect_timeout_s=data["connect_timeout_s"],
        )


@dataclasses.dataclass(frozen=True)
class WickerAwsS3Config:
    loaded: bool = False
    s3_datasets_path: str
    region: str
    boto_config: BotoS3Config
    store_concatenated_bytes_files_in_dataset: bool = False

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerAwsS3Config:
        return cls(
            loaded=True,
            s3_datasets_path=data["s3_datasets_path"],
            region=data["region"],
            boto_config=BotoS3Config.from_json(data["boto_config"]),
            store_concatenated_bytes_files_in_dataset=data.get("store_concatenated_bytes_files_in_dataset", False),
        )


@dataclasses.dataclass(frozen=True)
class WickerFileSystemConfig:
    loaded: bool = False
    prefix_replace_path: str
    root_datasets_path: str

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerFileSystemConfig:
        return cls(
            loaded=True,
            prefix_replace_path=data.get("prefix_replace_path", ""),
            root_datasets_path=data["root_datasets_path"],
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
            retries=data["retries"],
            timeout=data["timeout"],
            retry_backoff=data["retry_backoff"],
            retry_delay_s=data["retry_delay_s"],
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
        config = cls(raw=data)
        if "aws_s3_config" in data and "filesystem_config" in data:
            raise ValueError("Cannot define both aws_s3_config and filesystem_config in wickerconfig file")
        if "aws_s3_config" in data:
            config.aws_s3_config = WickerAwsS3Config.from_json(data.get["aws_s3_config"])
        if "filesystem_config" in data:
            config.filesystem_config = WickerFileSystemConfig.from_json(data.get["filesystem_config"])
        if "storage_download_config" in data:
            config.storage_download_config = StorageDownloadConfig.from_json(data.get["storage_download_config"])
        if "wandb_config" in data:
            config.wandb_config = WickerWandBConfig.from_json(data.get["wandb_config"])
        return config


def get_config() -> WickerConfig:
    """Retrieves the Wicker config for the current process"""

    wicker_config_path = os.getenv("WICKER_CONFIG_PATH", os.path.expanduser("~/wickerconfig.json"))
    with open(wicker_config_path, "r") as f:
        config = WickerConfig.from_json(json.load(f))
    return config
