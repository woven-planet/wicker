"""This module defines how to configure Wicker from the user environment
"""

from __future__ import annotations

import dataclasses
import json
import os
from functools import lru_cache
from typing import Any, Dict


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
class GCloudStorageConfig:
    bucket: str
    aws_transfer_cut_prefix: str = ""
    bucket_wicker_data_head_path: str = ""
    local_gcloud_tmp_data_transfer_dir: str = ""

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> GCloudStorageConfig:
        return cls(
            aws_transfer_cut_prefix="" if "aws_transfer_cut_prefix" not in data else data["aws_transfer_cut_prefix"],
            bucket=data["bucket"],  # only hard requirement is bucket,
            bucket_wicker_data_head_path=""
            if "bucket_wicker_data_head_path" not in data
            else data["bucket_wicker_data_head_path"],
            local_gcloud_tmp_data_transfer_dir=os.getenv("TMPDIR", "tmp_datasets")
            if "local_gcloud_tmp_data_transfer_dir" not in data
            else data["local_gcloud_tmp_data_transfer_dir"],
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
class WickerConfig:
    raw: Dict[str, Any]
    aws_s3_config: WickerAwsS3Config
    gcloud_storage_config: GCloudStorageConfig
    storage_download_config: StorageDownloadConfig
    wandb_config: WickerWandBConfig

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> WickerConfig:
        return cls(
            raw=data,
            aws_s3_config=WickerAwsS3Config.from_json(data["aws_s3_config"]),
            gcloud_storage_config=GCloudStorageConfig.from_json(data["gcloud_storage_config"]),
            storage_download_config=StorageDownloadConfig.from_json(data["storage_download_config"]),
            wandb_config=WickerWandBConfig.from_json(data.get("wandb_config", {})),
        )


@lru_cache(maxsize=1)
def get_config() -> WickerConfig:
    """Retrieves the Wicker config for the current process

    Cached with lru size 1 so multiple threads can pull object without
    re-opening and reading file.

    Returns:
        WickerConfig: dataset config file
    """

    wicker_config_path = os.getenv("WICKER_CONFIG_PATH", os.path.expanduser("~/wickerconfig.json"))
    with open(wicker_config_path, "r") as f:
        config = WickerConfig.from_json(json.load(f))
    return config
