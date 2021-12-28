import abc
import os
from typing import Any, Dict, List, Optional

import pyarrow  # type: ignore
import pyarrow.fs as pafs  # type: ignore
import pyarrow.parquet as papq  # type: ignore

from wicker.core.column_files import ColumnBytesFileCache
from wicker.core.config import get_config  # type: ignore
from wicker.core.definitions import DatasetDefinition, DatasetID, DatasetPartition
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.schema import dataloading, serialization
from wicker.schema.schema import DatasetSchema

# How long to wait before timing out on filelocks in seconds
FILE_LOCK_TIMEOUT_SECONDS = 240


class AbstractDataset(abc.ABC):
    """Interface for a Map-style (non-streaming) Dataset"""

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Gets the ith Example from the given dataset/version/partition"""
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset/version/partition"""
        pass

    @abc.abstractmethod
    def schema(self) -> DatasetSchema:
        """Return the schema of the dataset."""
        pass

    @abc.abstractmethod
    def arrow_table(self) -> pyarrow.Table:
        """Return the pyarrow table with all the metadata fields and pointers of the dataset."""
        pass


class S3Dataset(AbstractDataset):
    """Implementation for a Map-based dataset"""

    def __init__(
        self,
        dataset_name: str,
        dataset_version: str,
        dataset_partition_name: str,
        local_cache_path_prefix: str = os.getenv("TMPDIR", "/tmp"),
        columns_to_load: Optional[List[str]] = None,
        storage: Optional[S3DataStorage] = None,
        s3_path_factory: Optional[S3PathFactory] = None,
        pa_filesystem: Optional[pafs.FileSystem] = None,
        filelock_timeout_seconds: int = FILE_LOCK_TIMEOUT_SECONDS,
        treat_objects_as_bytes: bool = False,
    ):
        """Initializes an S3Dataset

        :param dataset_name: name of the dataset
        :param dataset_version: version of the dataset
        :param dataset_partition_name: partition name
        :param columns_to_load: list of columns to load, defaults to None which loads all columns
        :param data_service: S3DataService instance to use, defaults to None which uses default initializations
        :param filelock_timeout_seconds: number of seconds after which to timeout on waiting for downloads,
            defaults to FILE_LOCK_TIMEOUT_SECONDS
        :param treat_objects_as_bytes: If set, don't try to decode ObjectFields and keep them as binary data.
        """
        super().__init__()
        self._columns_to_load: Optional[List[str]] = columns_to_load
        self._treat_objects_as_bytes = treat_objects_as_bytes
        self._schema: Optional[DatasetSchema] = None
        self._arrow_table: Optional[pyarrow.Table] = None

        self._local_cache_path_prefix = local_cache_path_prefix
        self._filelock_timeout_seconds = filelock_timeout_seconds
        self._storage = storage if storage is not None else S3DataStorage()
        self._s3_path_factory = s3_path_factory if s3_path_factory is not None else S3PathFactory()
        self._column_bytes_file_cache = ColumnBytesFileCache(
            local_cache_path_prefix=local_cache_path_prefix,
            filelock_timeout_seconds=filelock_timeout_seconds,
            path_factory=self._s3_path_factory,
            storage=self._storage,
        )
        self._pa_filesystem = (
            pafs.S3FileSystem(region=get_config().aws_s3_config.region) if pa_filesystem is None else pa_filesystem
        )

        self._dataset_id = DatasetID(name=dataset_name, version=dataset_version)
        self._partition = DatasetPartition(dataset_id=self._dataset_id, partition=dataset_partition_name)
        self._dataset_definition = DatasetDefinition(
            self._dataset_id,
            schema=self.schema(),
        )

    def schema(self) -> DatasetSchema:
        if self._schema is None:
            schema_path = self._s3_path_factory.get_dataset_schema_path(self._dataset_id)
            local_path = self._storage.fetch_file_s3(
                schema_path, self._local_cache_path_prefix, timeout_seconds=self._filelock_timeout_seconds
            )
            with open(local_path, "rb") as f:
                self._schema = serialization.loads(
                    f.read().decode("utf-8"), treat_objects_as_bytes=self._treat_objects_as_bytes
                )
        return self._schema

    def arrow_table(self) -> pyarrow.Table:
        path = self._s3_path_factory.get_dataset_partition_path(self._partition, s3_prefix=False)
        if not self._arrow_table:
            self._arrow_table = papq.read_table(path, filesystem=self._pa_filesystem)
        return self._arrow_table

    def __len__(self) -> int:
        return len(self.arrow_table())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tbl = self.arrow_table()
        columns = self._columns_to_load if self._columns_to_load is not None else tbl.column_names
        row = {col: tbl[col][idx].as_py() for col in columns}
        return dataloading.load_example(
            self._column_bytes_file_cache.resolve_pointers(row, self.schema()),
            self.schema(),
        )
