import abc
import logging
import os
from functools import cached_property
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pyarrow  # type: ignore
import pyarrow.fs as pafs  # type: ignore
import pyarrow.parquet as papq  # type: ignore
import tqdm  # type: ignore

from wicker.core.column_files import (
    ColumnBytesFileCache,
    ColumnBytesFileLocationV1,
    ColumnBytesFileReader,
)
from wicker.core.config import get_config  # type: ignore
from wicker.core.definitions import DatasetDefinition, DatasetID, DatasetPartition
from wicker.core.shuffle import ShuffleWorker
from wicker.core.storage import (
    AbstractDataStorage,
    FileSystemDataStorage,
    S3DataStorage,
    S3PathFactory,
    WickerPathFactory,
)
from wicker.schema import dataloading, serialization
from wicker.schema.schema import DatasetSchema

# How long to wait before timing out on filelocks in seconds
FILE_LOCK_TIMEOUT_SECONDS = 300

logger = logging.getLogger(__name__)


class AbstractDataset(abc.ABC):
    """Interface for a Map-style (non-streaming) Dataset"""

    def __init__(
        self,
        dataset_name: str,
        dataset_partition_name: str,
        dataset_version: str,
        path_factory: WickerPathFactory,
        pa_filesystem: pafs.FileSystem,
        storage: AbstractDataStorage,
        columns_to_load: Optional[List[str]] = None,
        filelock_timeout_seconds: int = FILE_LOCK_TIMEOUT_SECONDS,
        local_cache_path_prefix: Optional[str] = os.getenv("TMPDIR", "/tmp"),
        treat_objects_as_bytes: bool = False,
    ) -> None:
        """Init an AbstractDataset object.

        :param dataset_name: name of the dataset
        :param dataset_partition_name: partition name
        :param dataset_version: version of the dataset
        :param path_factory: WickerPathFactory for pulling consistent paths.
        :param pa_filesystem: Pyarrow filesystem for reading the parquet files and tables.
        :param storage: AbstractDataStorage for data access.
        :param columns_to_load: list of columns to load, defaults to None which loads all columns
        :param filelock_timeout_seconds: number of seconds after which to timeout on waiting for downloads,
            defaults to FILE_LOCK_TIMEOUT_SECONDS
        :param local_cache_path_prefix: Path to local cache path, if None don't create cache
        :param treat_objects_as_bytes: If set, don't try to decode ObjectFields and keep them as binary data.
        """
        super().__init__()
        self._arrow_table: Optional[pyarrow.Table] = None
        self._columns_to_load = columns_to_load
        self._filelock_timeout_seconds = filelock_timeout_seconds
        self._local_cache_path_prefix = local_cache_path_prefix
        self._path_factory = path_factory
        self._pa_filesystem = pa_filesystem
        self._storage = storage
        self._treat_objects_as_bytes = treat_objects_as_bytes

        self._column_bytes_file_reader: ColumnBytesFileReader = ColumnBytesFileReader(
            column_bytes_root_path=path_factory._get_column_concatenated_bytes_files_path(dataset_name=dataset_name)
        )
        # if we have a cache prefix create a cache
        if local_cache_path_prefix is not None:
            logging.info(f"Cache passed at path - {local_cache_path_prefix}, creating read through cache on top.")
            self._column_bytes_file_reader = ColumnBytesFileCache(
                column_root_path=path_factory._get_column_concatenated_bytes_files_path(dataset_name=dataset_name),
                local_cache_path_prefix=local_cache_path_prefix,
                filelock_timeout_seconds=filelock_timeout_seconds,
                storage=self._storage,
            )
        else:
            logging.info("No cache passed, reading without caching.")
        self._dataset_id = DatasetID(name=dataset_name, version=dataset_version)
        self._dataset_definition = DatasetDefinition(
            self._dataset_id,
            schema=self.schema,
        )
        self._partition = DatasetPartition(dataset_id=self._dataset_id, partition=dataset_partition_name)

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Gets the ith Example from the given dataset/version/partition"""
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset/version/partition"""
        pass

    @property
    def schema(self) -> DatasetSchema:
        """Return the schema of the dataset."""
        schema_data = None
        schema_path = self._path_factory._get_dataset_schema_path(self._dataset_id)
        local_path = schema_path
        # only fetch the file locally if we have a cache prefix
        if self._local_cache_path_prefix is not None:
            local_path = self._storage.fetch_file(
                schema_path, self._local_cache_path_prefix, timeout_seconds=self._filelock_timeout_seconds
            )
        with open(local_path, "rb") as f:
            schema_data = serialization.loads(
                f.read().decode("utf-8"), treat_objects_as_bytes=self._treat_objects_as_bytes
            )
        return schema_data

    @abc.abstractmethod
    def arrow_table(self) -> pyarrow.Table:
        """Return the pyarrow table with all the metadata fields and pointers of the dataset."""
        pass


class FileSystemDataset(AbstractDataset):
    """Implementation of a Map-based dataset on local file system or mounted drive"""

    def __init__(
        self,
        dataset_name: str,
        dataset_partition_name: str,
        dataset_version: str,
        columns_to_load: Optional[List[str]] = None,
        filelock_timeout_seconds: int = FILE_LOCK_TIMEOUT_SECONDS,
        filesystem_root_path: Optional[str] = None,
        local_cache_path_prefix: Optional[str] = os.getenv("TMPDIR", "/tmp"),
        pa_filesystem: Optional[pafs.LocalFileSystem] = None,
        path_factory: Optional[WickerPathFactory] = None,
        storage: Optional[FileSystemDataStorage] = None,
        treat_objects_as_bytes: bool = False,
    ):
        """Initializes a FileSystemDataset.

        :param dataset_name: name of the dataset
        :param dataset_partition_name: partition name
        :param dataset_version: version of the dataset
        :param columns_to_load: list of columns to load, defaults to None which loads all columns
        :param filelock_timeout_seconds: number of seconds after which to timeout on waiting for downloads,
            defaults to FILE_LOCK_TIMEOUT_SECONDS
        :param filesystem_root_path: path to the root of the wicker file system. If path factory is none,
            this must be set.
        :param local_cache_path_prefix: Path to local cache path, if None don't create cache
        :param pa_filesystem: Pyarrow filesystem for reading the parquet files and tables.
        :param path_factory: Optional WickerPathFactory for pulling consistent paths.
        :param storage: Optional FileSystemDataStorage object for pulling files from filesystem
        :param treat_objects_as_bytes: If set, don't try to decode ObjectFields and keep them as binary data.
        """
        if path_factory is None and filesystem_root_path is None:
            raise ValueError("Need to pass either path factory of wicker dataset or root of the tree.")
        # ignore type failure here as we handle the case where they're both none above
        path_factory = (
            path_factory
            if path_factory is not None
            else WickerPathFactory(root_path=filesystem_root_path)  # type: ignore
        )
        pa_filesystem = pafs.LocalFileSystem() if pa_filesystem is None else pa_filesystem
        storage = storage if storage is not None else FileSystemDataStorage()

        super().__init__(
            dataset_name=dataset_name,
            dataset_partition_name=dataset_partition_name,
            dataset_version=dataset_version,
            storage=storage,
            pa_filesystem=pa_filesystem,
            path_factory=path_factory,
            columns_to_load=columns_to_load,
            filelock_timeout_seconds=filelock_timeout_seconds,
            local_cache_path_prefix=local_cache_path_prefix,
            treat_objects_as_bytes=treat_objects_as_bytes,
        )

    def arrow_table(self) -> pyarrow.Table:
        """Grab and load arrow table from expected path.

        Returns:
            pyarrow.Table: Arrow table object for the loaded dataset.
        """
        path = self._path_factory._get_dataset_partition_path(self._partition)
        if not self._arrow_table:
            self._arrow_table = papq.read_table(path, columns=self._columns_to_load, filesystem=self._pa_filesystem)
        return self._arrow_table

    def __len__(self) -> int:
        """Get length of arrow table inferring it is the same as the dataset.

        Returns:
            int: length of arrow table.
        """
        return len(self.arrow_table())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get data item at index within arrow table.

        Pulls from either cache or data store the item in the dataset at index specified.

        Args:
            idx (int): idx in arrow table to grab data.

        Returns:
            Dict[str, Any]: Row of data defined through schema object.
        """
        tbl = self.arrow_table()
        columns = self._columns_to_load if self._columns_to_load is not None else tbl.column_names
        row = {col: tbl[col][idx].as_py() for col in columns}
        return dataloading.load_example(
            self._column_bytes_file_reader.resolve_pointers(row, self.schema),
            self.schema,
        )


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
        pa_filesystem = (
            pafs.S3FileSystem(region=get_config().aws_s3_config.region) if pa_filesystem is None else pa_filesystem
        )
        s3_path_factory = s3_path_factory if s3_path_factory is not None else S3PathFactory()
        storage = storage if storage is not None else S3DataStorage()

        super().__init__(
            dataset_name=dataset_name,
            dataset_partition_name=dataset_partition_name,
            dataset_version=dataset_version,
            storage=storage,
            pa_filesystem=pa_filesystem,
            path_factory=s3_path_factory,
            columns_to_load=columns_to_load,
            local_cache_path_prefix=local_cache_path_prefix,
            filelock_timeout_seconds=filelock_timeout_seconds,
            treat_objects_as_bytes=treat_objects_as_bytes,
        )

    def arrow_table(self) -> pyarrow.Table:
        path = self._path_factory._get_dataset_partition_path(self._partition, prefix_to_trim="s3://")
        if not self._arrow_table:
            self._arrow_table = papq.read_table(path, columns=self._columns_to_load, filesystem=self._pa_filesystem)
        return self._arrow_table

    def __len__(self) -> int:
        return len(self.arrow_table())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tbl = self.arrow_table()
        columns = self._columns_to_load if self._columns_to_load is not None else tbl.column_names
        row = {col: tbl[col][idx].as_py() for col in columns}
        return dataloading.load_example(
            self._column_bytes_file_reader.resolve_pointers(row, self.schema),
            self.schema,
        )

    def _get_parquet_dir_size(self) -> int:
        """Get the arrow path and find all the files within, count their bytes

        Returns:
            int: bytes in parquet directory
        """
        # bytes size of arrow table not bytes in arrow table
        # bytes in arrow table is a method of arrow table but it doesn't
        # reflect the size of the file sizes stored on s3 just the loaded data
        arrow_path = self._path_factory._get_dataset_partition_path(self._partition, prefix_to_trim="s3://")
        bucket, key = arrow_path.replace("s3://", "").split("/", 1)

        def get_folder_size(bucket, prefix):
            total_size = 0
            for obj in boto3.resource("s3").Bucket(bucket).objects.filter(Prefix=prefix):
                total_size += obj.size
            return total_size

        return get_folder_size(bucket, key)

    def _get_dataset_size(self):
        """Gets total size of the dataset in bits

        Returns:
            int: total dataset size in bits
        """
        # intialize with size of parquet dir
        par_dir_bytes = self._get_parquet_dir_size()

        # need to know which columns are heavy pntr columns we go to for
        # byte adding
        schema = self.schema
        heavy_pointer_cols = []
        for col_name in schema.get_all_column_names():
            if schema.get_column(col_name).is_heavy_pointer:
                heavy_pointer_cols.append(col_name)

        # Each individual row only knows which column file it goes to, so we have to
        # neccesarily parse all rows :( to get the column files. This should be cached
        # as metadata but that would require re-curating the datasets.
        print("Creating arrow table")
        arrow_table = self.arrow_table()

        buckets_keys = set()
        # ignore typing to avoid changing the typing of Shuffle Worker yet
        # ToDo: Change typing of ShuffleWorker and let it take local data storage
        worker = ShuffleWorker(storage=self._storage)  # type: ignore
        print("Processing through heavy pointers")
        for heavy_pntr_col in heavy_pointer_cols:
            print(f"Evaulating {heavy_pntr_col} for column file locations")
            for location_bytes in tqdm.tqdm(arrow_table[heavy_pntr_col].to_pylist()):
                location = ColumnBytesFileLocationV1.from_bytes(location_bytes)
                path = worker.s3_path_factory.get_column_concatenated_bytes_s3path_from_uuid(
                    location.file_id.bytes, dataset_name=self._dataset_id.name
                )
                bucket, key = path.replace("s3://", "").split("/", 1)
                buckets_keys.add((bucket, key))

        # we use existing shuffle worker here to grab the column file from
        # know location.
        def get_file_size_s3(bucket_key: Tuple[str, str]) -> int:
            s3 = boto3.resource("s3")
            bucket, key = bucket_key
            bucket, key = path.replace("s3://", "").split("/", 1)
            byte_length = s3.Object(bucket, key).content_length
            return byte_length

        print("Grabbing file information from s3 heads")
        with ThreadPool() as pool:
            results = list(tqdm.tqdm(pool.imap(get_file_size_s3, buckets_keys), total=len(buckets_keys)))
        return sum(results) + par_dir_bytes

    @cached_property
    def dataset_size(self) -> int:
        """Total dataset size in bits

        Returns:
            int: total dataset size in bits
        """
        return self._get_dataset_size()
