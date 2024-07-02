"""
This file is for S3 storage routines used by S3 datasets

Boto3 S3 documentation links:
- https://boto3.amazonaws.com/v1/documentation/api/latest/guide/index.html
- https://boto3.amazonaws.com/v1/documentation/api/latest/guide/resources.html
- https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
"""
import io
import logging
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import boto3
import boto3.session
import botocore  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
from retry import retry

from wicker.core.config import get_config
from wicker.core.definitions import DatasetID, DatasetPartition
from wicker.core.filelock import SimpleUnixFileLock
from wicker.core.utils import time_limit

logger = logging.getLogger(__name__)


class AbstractDataStorage(ABC):
    """Abstract data storage class that implements required methods for Column Bytes File Cache"""

    def __init__(self):
        self.read_timeout = get_config().storage_download_config.timeout

    @abstractmethod
    def _download(self, input_path: str, local_dst_path: str) -> str:
        """Download file from data source.

        Args:
            input_path (str): input path to file to download.
            local_dst_path (str): path to download location.

        Returns:
            str: path file was downloaded.
        """
        pass

    @abstractmethod
    def fetch_file(self, input_path: str, local_prefix: str, timeout_seconds: int) -> str:
        """Fetch file from chosen data storage method.

        :param input_path: input file path
        :param local_prefix: local path that specifies where to download the file
        :param timeout_seconds: number of seconds till timing out on waiting for the file to be downloaded
        :return: local path to the downloaded file

        """
        pass

    @abstractmethod
    def put_file(self, input_path: str, target_path: str) -> None:
        """Put file on data storage in target location.

        :param input_path: input path of file
        :type input_path: str
        :param target_path: target path to place file
        :type target_path: str
        """
        pass

    @abstractmethod
    def put_object(self, object_bytes: bytes, target_path: str) -> None:
        """Put object on data storage in target location.

        :param object_bytes: Bytes of object to store.
        :type object_bytes: bytes
        :param target_path: Path to storage destination.
        :type target_path: str
        """
        pass

    def _safe_file_download(self, input_path: str, local_dst_path: str) -> str:
        """Download file safely from data source with locking.

        Locks the file path and downloads the file to the local system. Does not release the lock
        until the file is downloaded fully.

        Args:
            input_path (str): path to file to download
            local_dst_path (str): path to save file at on local system

        Returns:
            str: local path of the file when downloaded
        """
        os.makedirs(os.path.dirname(local_dst_path), exist_ok=True)
        lock_path = local_dst_path + ".lock"
        success_marker = local_dst_path + ".success"
        while not os.path.isfile(success_marker):
            with SimpleUnixFileLock(lock_path, timeout_seconds=self.read_timeout):
                if not os.path.isfile(success_marker):
                    filedir = os.path.split(local_dst_path)[0]
                    os.makedirs(filedir, exist_ok=True)
                    try:
                        shutil.copyfile(input_path, local_dst_path)
                        with open(success_marker, "w"):
                            pass
                    except Exception:
                        logging.error(f"Failed to download/move object for file path: {input_path}")
                        raise
        return local_dst_path


class FileSystemDataStorage(AbstractDataStorage):
    """Storage routines for reading and writing objects in file system"""

    def __init__(self) -> None:
        """Constructor for a file system storage instance."""
        super().__init__()

    def _download(self, input_path: str, local_dst_path: str) -> str:
        return self._safe_file_download(input_path=input_path, local_dst_path=local_dst_path)

    def fetch_file(self, input_path: str, local_prefix: str, timeout_seconds: int = 120) -> str:
        """Fetches a file system, ie: gets the path to the file.

        This function assumes the input_path is a valid file in the file system based on wicker assumed pathing.

        The reasoning here is that if you use this for a mounted file system that doesn't have caching
        to local automatically you can grab and move files to local file system on instance.

        :param input_path: input file path on system
        :param local_prefix: local path that specifies where to download the file
        :param timeout_seconds: number of seconds till timing out on waiting for the file to be downloaded.
            Kept so api's are identical between S3 storage and local storage, deprecated and unused.
        :return: local path to the file on the local file system
        """
        local_full_path = os.path.join(local_prefix, os.path.basename(input_path))
        return self._download(input_path=input_path, local_dst_path=local_full_path)

    def put_file(self, input_path: str, target_path: str) -> None:
        """Put file on local or mounted data storage.

        :param input_path: file path on the local system.
        :param target_path: file path on local system or mounted drive.
        """
        os.makedirs(Path(target_path).parent, exist_ok=True)
        shutil.copy2(input_path, target_path)

    def put_object(self, object_bytes: bytes, target_path: str) -> None:
        """Put object on data storage

        :param object_bytes: bytes to write to path
        :param target_path: path to write object
        """
        os.makedirs(Path(target_path).parent, exist_ok=True)
        with open(target_path, "wb") as binary_file:
            binary_file.write(object_bytes)


class S3DataStorage(AbstractDataStorage):
    """Storage routines for reading and writing objects in S3"""

    def __init__(self, session: Optional[boto3.session.Session] = None) -> None:
        """Constructor for an S3Storage instance

        Part of the reason to structure S3 routines as a class rather utility functions is to enable
        unit tests to easily mock / patch the S3 client or to utilize the S3 Stubber. Unit tests
        might also find it convenient to mock or patch member functions on instances of this class.
        """
        super().__init__()
        boto_config = get_config().aws_s3_config.boto_config
        boto_client_config = botocore.config.Config(
            max_pool_connections=boto_config.max_pool_connections,
            read_timeout=boto_config.read_timeout_s,
            connect_timeout=boto_config.connect_timeout_s,
        )
        self.session = boto3.session.Session() if session is None else session
        self.client = self.session.client("s3", config=boto_client_config)

    def __getstate__(self) -> Dict[Any, Any]:
        return {}

    def __setstate__(self, state: Dict[Any, Any]) -> None:
        self.session = boto3.session.Session()
        self.client = self.session.client("s3")
        self.read_timeout = get_config().storage_download_config.timeout
        return

    @staticmethod
    def bucket_key_from_s3_path(s3_path: str) -> Tuple[str, str]:
        """Parses a fully-qualified S3 path such as "s3://hello/world" into the bucket and key

        :param s3_path: fully-qualified S3 path
        :type s3_path: str
        :return: tuple consisting of the S3 bucket and key
        :rtype: Tuple[str, str]
        """
        url_parsed = urlparse(s3_path)
        return url_parsed.netloc, url_parsed.path[1:]

    def check_exists_s3(self, input_path: str) -> bool:
        """Checks if a file exists on S3 under given path

        :param input_path: input file path in S3
        :type input_path: str
        :return: whether or not the file exists in S3
        :rtype: bool
        """
        bucket, key = self.bucket_key_from_s3_path(input_path)
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    def _download(self, input_path: str, local_dst_path: str) -> str:
        """Download file from input path to local path.

        Head on top of download_with_retries for backwards compatibility.

        Args:
            input_path (str): file input path to download.
            local_dst_path (str): path to download file.

        Returns:
            str: local path to downloaded file
        """
        bucket, key = self.bucket_key_from_s3_path(input_path)
        lock_path = local_dst_path + ".lock"
        success_marker = local_dst_path + ".success"
        os.makedirs(os.path.dirname(local_dst_path), exist_ok=True)

        while not os.path.isfile(success_marker):
            with SimpleUnixFileLock(lock_path, timeout_seconds=self.read_timeout):
                if not os.path.isfile(success_marker):

                    # For now, we will only download the file if it has not already been downloaded already.
                    # Long term, we would also add a size check or md5sum comparison against the object in S3.
                    filedir = os.path.split(local_dst_path)[0]
                    os.makedirs(filedir, exist_ok=True)
                    self.download_with_retries(bucket=bucket, key=key, local_path=local_dst_path)
                    with open(success_marker, "w"):
                        pass

        return local_dst_path

    @retry(
        Exception,
        tries=get_config().storage_download_config.retries,
        backoff=get_config().storage_download_config.retry_backoff,
        delay=get_config().storage_download_config.retry_delay_s,
        logger=logger,
    )
    def download_with_retries(self, bucket: str, key: str, local_path: str):
        try:
            with time_limit(
                self.read_timeout, f"Timing out in trying to download object for bucket: {bucket}, key: {key}"
            ):
                self.client.download_file(bucket, key, local_path)
        except Exception as e:
            logging.error(f"Failed to download s3 object in bucket: {bucket}, key: {key}")
            raise e

    def fetch_file(self, input_path: str, local_prefix: str, timeout_seconds: int = 120) -> str:
        """
        Fetches a file from S3 to the local machine and skips it if it already exists. This function
        is safe to call concurrently from multiple processes and utilizes a local filelock to block
        parallel downloads such that only one process will perform the download.

        This function assumes the input_path is a valid file in S3.

        :param input_path: input file path in S3
        :param local_prefix: local path that specifies where to download the file
        :param timeout_seconds: number of seconds till timing out on waiting for the file to be downloaded.
            Deprecated in favor of the internal variable already assigned.
        :return: local path to the downloaded file
        """
        bucket, key = self.bucket_key_from_s3_path(input_path)
        local_path = os.path.join(local_prefix, key)
        return self._download(input_path=input_path, local_dst_path=local_path)

    def fetch_file_s3(self, input_path: str, local_prefix: str, timeout_seconds: int = 120) -> str:
        """Deprecated fetch file access, function signature kept to preserve backwards compatibility."""
        return self.fetch_file(input_path=input_path, local_prefix=local_prefix, timeout_seconds=timeout_seconds)

    def fetch_obj_s3(self, input_path: str) -> bytes:
        """Fetches an object from S3 as bytes in memory

        :param input_path: path to object in s3
        :return: bytes of data in file
        """
        bucket, key = self.bucket_key_from_s3_path(input_path)
        bio = io.BytesIO()
        self.client.download_fileobj(bucket, key, bio)
        return bio.getvalue()

    def put_object(self, object_bytes: bytes, target_path: str) -> None:
        """Upload an object to S3

        :param object_bytes: the object to upload to S3
        :type object_bytes: bytes
        :param target_path: path to the file in S3
        :type target_path: str
        """
        # Long term, we would add an md5sum check and short-circuit the upload if they are the same
        bucket, key = self.bucket_key_from_s3_path(target_path)
        self.client.put_object(Body=object_bytes, Bucket=bucket, Key=key)

    def put_object_s3(self, object_bytes: bytes, s3_path: str) -> None:
        """Deprecated api access to the put object functionality."""
        self.put_object(object_bytes=object_bytes, target_path=s3_path)

    def put_file(self, local_path: str, target_path: str) -> None:
        """Upload a file to S3

        :param local_path: local path to the file
        :param target_path: s3 path to dump file to
        """
        bucket, key = self.bucket_key_from_s3_path(target_path)
        self.client.upload_file(local_path, bucket, key)

    def put_file_s3(self, local_path: str, s3_path: str) -> None:
        """Deprecated api access to the put file functionality."""
        self.put_file(local_path=local_path, target_path=s3_path)

    def __eq__(self, other: Any) -> bool:
        # We don't want to use isinstance here to make sure we have the same implementation.
        return super().__eq__(other) and type(self) == type(other)


class WickerPathFactory:
    """Factory class for Wicker dataset paths

    Our bucket should look like::

        <dataset-root>
        /__COLUMN_CONCATENATED_FILES__
            - file1
            - file2
        /dataset_name_1
        /dataset_name_2
            /0.0.1
            /0.0.2
            - avro_schema.json
            / assets
                - files added by users
            / partition_1.parquet
            / partition_2.parquet
                - _l5ml_dataset_partition_metadata.json
                - _SUCCESS
                - part-0-attempt-1234.parquet
                - part-1-attempt-2345.parquet

    If store_concatenated_bytes_files_in_dataset is True, then the bucket structure will
    be under the dataset directory:

            <root_path>
            /dataset_name_1
            /dataset_name_2
                /__COLUMN_CONCATENATED_FILES__
                    - file1
                    - file2
                /0.0.1
                /0.0.2
                - avro_schema.json
                / assets
                    - files added by users
                / partition_1.parquet
                / partition_2.parquet
                    - _l5ml_dataset_partition_metadata.json
                    - _SUCCESS
                    - part-0-attempt-1234.parquet
                    - part-1-attempt-2345.parquet
    """

    def __init__(self, root_path: str, store_concatenated_bytes_files_in_dataset: bool = False) -> None:
        """Init the path factory.

        Object to form the expected paths and return them to the user based of root path and storage bool.

        Args:.
            root_path (str): File system location of the root of the wicker file structure.
            store_concatenated_bytes_files_in_dataset (bool, optional): Whether to assume concatenated bytes files
                are stored in the dataset folder. Defaults to False
        """
        self.root_path: str = root_path
        self.store_concatenated_bytes_files_in_dataset = store_concatenated_bytes_files_in_dataset

    def __eq__(self, other: Any) -> bool:
        return super().__eq__(other) and type(self) == type(other) and self.root_path == other.root_path

    def _get_dataset_assets_path(self, dataset_id: DatasetID, prefix_to_trim: Optional[str] = None) -> str:
        """Get the asset path in known file structure.

        Args:
            dataset_id (DatasetID): ID of the dataset
            prefix_to_trim(Optional[str], optional): Optional prefix to remove from file paths. Defaults to None.

        Returns:
            str: path to assets folder
        """
        full_path = os.path.join(
            self.root_path,
            dataset_id.name,
            dataset_id.version,
            "assets",
        )
        if prefix_to_trim:
            return full_path.replace(prefix_to_trim, "")
        return full_path

    def _get_dataset_partition_metadata_path(
        self, data_partition: DatasetPartition, prefix_to_trim: Optional[str] = None
    ) -> str:
        """Get partition metadata path in wicker known file structure.

        Args:
            data_partition (DatasetPartition): data partition to use for path
            prefix_to_trim(Optional[str], optional): Optional prefix to remove from file paths. Defaults to None.

        Returns:
            str: Path to partition metadata json file.
        """
        full_path = os.path.join(
            self._get_dataset_partition_path(data_partition),
            "_l5ml_dataset_partition_metadata.json",
        )
        if prefix_to_trim:
            return full_path.replace(prefix_to_trim, "")
        return full_path

    def _get_dataset_partition_path(
        self, data_partition: DatasetPartition, prefix_to_trim: Optional[str] = None
    ) -> str:
        """Get the dataset partition parquet path.

        Private getter handling logic of pathing centrally.

        Args:
            data_partition (DatasetPartition): DatasetPartition to use for pathing
            prefix_to_trim(Optional[str], optional): Optional prefix to remove from file paths. Defaults to None.

        Returns:
            str: Path to partition parquet file
        """
        full_path = os.path.join(
            self.root_path,
            data_partition.dataset_id.name,
            data_partition.dataset_id.version,
            f"{data_partition.partition}.parquet",
        )

        if prefix_to_trim:
            return full_path.replace(prefix_to_trim, "")
        return full_path

    def _get_dataset_schema_path(self, dataset_id: DatasetID, prefix_to_trim: Optional[str] = None) -> str:
        """Get the dataset schema path.

        Private getter handling pathing logic to avro_schema json file.

        Args:
            dataset_id (DatasetID): ID of the dataset to use for pathing.
            prefix_to_trim(Optional[str], optional): Optional prefix to remove from file path. Defaults to None.

        Returns:
            str: Path to dataset avro schema file.
        """
        full_path = os.path.join(
            self.root_path,
            dataset_id.name,
            dataset_id.version,
            "avro_schema.json",
        )
        if prefix_to_trim:
            return full_path.replace(prefix_to_trim, "")
        return full_path

    def _get_column_concatenated_bytes_files_path(
        self, dataset_name: Optional[str] = None, prefix_to_trim: Optional[str] = None
    ) -> str:
        """Gets the path to the root of all column_concatenated_bytes files

        :param dataset_name: if self.store_concatenated_bytes_files_in_dataset is True,
            it requires dataset name, defaults to None
        :param prefix_to_trim: prefix to trim off path, if none skip
        :return: path to the column_concatenated_bytes file with the file_id
        """
        if self.store_concatenated_bytes_files_in_dataset:
            if dataset_name is None:
                raise ValueError("Must provide dataset_id if store_concatenated_bytes_files_in_dataset is True")
            full_path = os.path.join(self.root_path, dataset_name, "__COLUMN_CONCATENATED_FILES__")
        else:
            full_path = os.path.join(self.root_path, "__COLUMN_CONCATENATED_FILES__")
        if prefix_to_trim:
            return full_path.replace(prefix_to_trim, "")
        return full_path

    def get_temporary_row_files_path(self, dataset_id: DatasetID) -> str:
        """Get path to temporary rows file path.

        Args:
            dataset_id (DatasetID): ID of dataset to use for pathing.

        Returns:
            str: Path to temporary rows file.
        """
        full_path = os.path.join(
            self.root_path,
            "__TEMP_FILES__",
            dataset_id.name,
            dataset_id.version,
        )
        return full_path


class S3PathFactory(WickerPathFactory):
    """Factory class for S3 Wicker dataset paths

    Our bucket should look like::

        s3://<dataset-root>
        /__COLUMN_CONCATENATED_FILES__
            - file1
            - file2
        /dataset_name_1
        /dataset_name_2
            /0.0.1
            /0.0.2
            - avro_schema.json
            / assets
                - files added by users
            / partition_1.parquet
            / partition_2.parquet
                - _l5ml_dataset_partition_metadata.json
                - _SUCCESS
                - part-0-attempt-1234.parquet
                - part-1-attempt-2345.parquet

    If store_concatenated_bytes_files_in_dataset is True, then the bucket structure will
    be under the dataset directory:

            s3://<root_path>
            /dataset_name_1
            /dataset_name_2
                /__COLUMN_CONCATENATED_FILES__
                    - file1
                    - file2
                /0.0.1
                /0.0.2
                - avro_schema.json
                / assets
                    - files added by users
                / partition_1.parquet
                / partition_2.parquet
                    - _l5ml_dataset_partition_metadata.json
                    - _SUCCESS
                    - part-0-attempt-1234.parquet
                    - part-1-attempt-2345.parquet
    """

    def __init__(self, s3_root_path: Optional[str] = None) -> None:
        """Init S3PathFactory.

        Args:
            s3_root_path (Optional[str], optional): path to s3 root path. Defaults to None.
        """
        s3_config = get_config().aws_s3_config
        store_concatenated_bytes_files_in_dataset = s3_config.store_concatenated_bytes_files_in_dataset
        s3_root_path = s3_root_path if s3_root_path is not None else s3_config.s3_datasets_path
        # ignore type as we already handled none case above
        super().__init__(s3_root_path, store_concatenated_bytes_files_in_dataset)  # type: ignore

    def get_dataset_assets_path(self, dataset_id: DatasetID, s3_prefix: bool = True) -> str:
        """Get path to data assets folder.

        Public getter for data asset folder path logic.

        Args:
            dataset_id (DatasetID): ID to gather file path.
            s3_prefix (bool, optional): Whether to keep the s3 prefix or not. Defaults to True.

        Returns:
            str: Path to data assets folder.
        """
        prefix_to_trim = "s3://" if not s3_prefix else None
        return self._get_dataset_assets_path(dataset_id=dataset_id, prefix_to_trim=prefix_to_trim)

    def get_dataset_partition_metadata_path(self, data_partition: DatasetPartition, s3_prefix: bool = True) -> str:
        """Get metadata file path for partition.

        Args:
            data_partition (DatasetPartition): Partition to gather file path.
            s3_prefix (bool, optional): Whether to keep the s3 prefix or not. Defaults to True.

        Returns:
            str: Path to dataset partition metadata file.
        """
        prefix_to_trim = "s3://" if not s3_prefix else None
        return self._get_dataset_partition_metadata_path(data_partition, prefix_to_trim)

    def get_dataset_partition_path(self, data_partition: DatasetPartition, s3_prefix: bool = True) -> str:
        """Get path to dataset partition data file.

        Args:
            data_partition (DatasetPartition): Partition to gather file path.
            s3_prefix (bool, optional): Whether to keep the s3 prefix or not. Defaults to True.

        Returns:
            str: Path to dataset partition data file.
        """
        prefix_to_trim = "s3://" if not s3_prefix else None
        return self._get_dataset_partition_path(data_partition=data_partition, prefix_to_trim=prefix_to_trim)

    def get_dataset_schema_path(self, dataset_id: DatasetID, s3_prefix: bool = True) -> str:
        """Get path to the dataset schema.

        Args:
            dataset_id (DatasetID): ID of the dataset.
            s3_prefix (bool, optional): Whether to keep the s3 prefix or not. Defaults to True.

        Returns:
            str: Path to dataset schema.
        """
        prefix_to_trim = "s3://" if not s3_prefix else None
        return self._get_dataset_schema_path(dataset_id=dataset_id, prefix_to_trim=prefix_to_trim)

    def get_column_concatenated_bytes_files_path(self, s3_prefix: bool = True, dataset_name: str = None) -> str:
        """Gets the path to the root of all column_concatenated_bytes files

        :param s3_prefix: whether to return the s3:// prefix, defaults to True
        :param dataset_name: if self.store_concatenated_bytes_files_in_dataset is True,
            it requires dataset name, defaults to None
        :return: path to the column_concatenated_bytes file with the file_id
        """
        prefix_to_trim = "s3://" if not s3_prefix else None
        return self._get_column_concatenated_bytes_files_path(dataset_name=dataset_name, prefix_to_trim=prefix_to_trim)

    def get_column_concatenated_bytes_s3path_from_uuid(self, file_uuid: bytes, dataset_name: str = None) -> str:
        """Public gettr for column concat bytes with uuid

        Args:
            file_uuid (bytes): uuid of the file
            dataset_name (str, optional): Name of the dataset to gather. Defaults to None.

        Returns:
            str: _description_
        """
        columns_root_path = self.get_column_concatenated_bytes_files_path(dataset_name=dataset_name)
        return os.path.join(columns_root_path, str(uuid.UUID(bytes=file_uuid)))
