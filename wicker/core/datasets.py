import abc
import logging
import os
from functools import cached_property
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pyarrow  # type: ignore
import pyarrow.fs as pafs  # type: ignore
import pyarrow.parquet as papq  # type: ignore

from wicker.core.column_files import ColumnBytesFileCache, ColumnBytesFileLocationV1
from wicker.core.config import get_config  # type: ignore
from wicker.core.definitions import DatasetDefinition, DatasetID, DatasetPartition
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.schema import dataloading, serialization
from wicker.schema.schema import DatasetSchema

# How long to wait before timing out on filelocks in seconds
FILE_LOCK_TIMEOUT_SECONDS = 300


def get_file_size_s3_multiproc(buckets_keys: List[Tuple[str, str]]) -> int:
    """Get file size of s3 files, most often column files.

    This works on any list of buckets and keys but is generally only
    used for column files as those are the majority of what is stored on
    s3 for Wicker. Wicker also stores parquet files on s3 but those are limited
    to one file per dataset and one schema file.

    This splits your buckets_keys_list across multiple processes on your local host
    where each process is then multi threaded further. This reduces the i/o wait by
    parellelizing across all available procs and threads on a single machine.

    Args:
        buckets_keys: (List[Tuple[str, str]]): A list of buckets and keys for which
        to fetch size in bytes on s3. Tuple index 0 is bucket and index 1 is key of the file.

    Returns:
        int size of file list in bytes.
    """
    buckets_keys_chunks = chunk_data_for_split(chunkable_data=buckets_keys, chunk_number=200)

    logging.info("Grabbing file information from s3 heads")
    pool = Pool(cpu_count() - 1)
    return sum(list(pool.map(get_file_size_s3_threaded, buckets_keys_chunks)))


def get_file_size_s3_threaded(buckets_keys_chunks_local: List[Tuple[str, str]]) -> int:
    """Get file size of a list of s3 paths.

    Args:
        buckets_keys_chunks_local - The list of tuples denoting bucket and key of files on s3 to
        parse. Generally column files but will work with any data.

    Returns:
        int: size of the set of files in bytes
    """
    local_chunks = chunk_data_for_split(chunkable_data=buckets_keys_chunks_local, chunk_number=200)
    thread_pool = ThreadPool()

    return sum(list(thread_pool.map(iterate_bucket_key_chunk_for_size, local_chunks)))  # type: ignore


def chunk_data_for_split(chunkable_data: List[Any], chunk_number: int = 500) -> List[List[Any]]:
    """Chunk data into a user specified number of chunks.

    Args:
        chunkable_data (List[Any]): Data to be chunked into smaller pieces.
        chunk_number (int): Number of chunks to form.

    Returns:
        List[List[Any]]: List of subsets of input data.
    """
    local_chunks = []
    local_chunk_size = len(chunkable_data) // chunk_number
    for i in range(0, chunk_number - 1):
        chunk = chunkable_data[i * local_chunk_size : (i + 1) * local_chunk_size]
        local_chunks.append(chunk)

    last_chunk_size = len(chunkable_data) - (chunk_number * local_chunk_size)
    if last_chunk_size > 0:
        last_chunk = chunkable_data[-last_chunk_size:]
        local_chunks.append(last_chunk)

    return local_chunks


def iterate_bucket_key_chunk_for_size(bucket_key_locs: List[Tuple[str, str]]) -> int:  # type: ignore
    """Iterate on chunk of s3 files to get local length of bytes.

    Args:
        bucket_key_locs: List of Tuple[str, str] containing the s3 bucket and key to check size.

    Returns:
        int: total amount of bytes in the file chunk list

    """
    local_len = 0

    # create the s3 resource locally and don't pass in. Boto3 docs state to do this in each thread
    # and not pass around.
    s3_resource = boto3.resource("s3")
    for bucket_key_loc in bucket_key_locs:
        bucket_loc, key_loc = bucket_key_loc
        # get the byte length for the object
        byte_length = s3_resource.Object(bucket_loc, key_loc).content_length
        local_len += byte_length
    return local_len


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
        filters=None,
    ):
        """Initializes an S3Dataset

        :param dataset_name: name of the dataset
        :param dataset_version: version of the dataset
        :param dataset_partition_name: partition name
        :param columns_to_load: list of columns to load, defaults to None which loads all columns
        :param data_service: S3DataService instance to use, defaults to None which uses default initializations
        :param filelock_timeout_seconds: number of seconds after which to timeout on waiting for downloads,
            defaults to FILE_LOCK_TIMEOUT_SECONDS
        :param treat_objects_as_bytes: If set, don't try to decode ObjectFields and keep them as binary data
        :param filters: Only returns rows which match the filter. Defaults to None, i.e., returns all rows.
        :type filters: pyarrow.compute.Expression, List[Tuple], or List[List[Tuple]], optional
        .. seealso:: `filters in <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html>`__ # noqa
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
            dataset_name=dataset_name,
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
        self.filters = filters

    def schema(self) -> DatasetSchema:
        if self._schema is None:
            schema_path = self._s3_path_factory.get_dataset_schema_path(self._dataset_id)
            local_path = self._storage.fetch_file(
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
            self._arrow_table = papq.read_table(
                path, columns=self._columns_to_load, filesystem=self._pa_filesystem, filters=self.filters
            )
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

    def _get_parquet_dir_size(self) -> int:
        """Get the parquet path and find all the files within, count their bytes

        Returns:
            int: bytes in parquet directory
        """
        # bytes size of arrow table not bytes in arrow table
        # bytes in arrow table is a method of arrow table but it doesn't
        # reflect the size of the file sizes stored on s3 just the loaded data
        arrow_path = self._s3_path_factory.get_dataset_partition_path(self._partition, s3_prefix=False)

        bucket, key = arrow_path.replace("s3://", "").split("/", 1)

        def get_folder_size(bucket, prefix):
            total_size = 0
            for obj in boto3.resource("s3").Bucket(bucket).objects.filter(Prefix=prefix):
                total_size += obj.size
            return total_size

        return get_folder_size(bucket, key)

    def _get_dataset_partition_size(self) -> int:
        """Gets total size of the dataset partition in bytes.

        Returns:
            int: total dataset partition size in bytes
        """

        # intialize with size of parquet dir
        logging.info("Parsing parquet and arrow dir for size.")
        par_dir_bytes = self._get_parquet_dir_size()

        # need to know which columns are heavy pntr columns we go to for
        # byte adding, parse which are heavy pointers based off metadata
        # baked into schema
        schema = self.schema()
        heavy_pointer_cols = []
        for col_name in schema.get_all_column_names():
            column = schema.get_column(col_name)
            if column is not None and column.is_heavy_pointer:
                heavy_pointer_cols.append(col_name)

        # create arrow table for parsing
        # only know the single partition arrow table loc so can only get one partition size
        logging.info("Creating arrow table")
        arrow_table = self.arrow_table()

        buckets_keys = set()

        logging.info("Processing through heavy pointers")
        for heavy_pntr_col in heavy_pointer_cols:
            logging.info(f"Evaulating {heavy_pntr_col} for column file locations")
            # Each individual row only knows which column file it goes to, so we have to
            # neccesarily parse all rows :( to get the column files. This should be cached
            # as metadata but that would require re-curating the datasets.
            for location_bytes in arrow_table[heavy_pntr_col].to_pylist():
                location = ColumnBytesFileLocationV1.from_bytes(location_bytes)
                path = self._s3_path_factory.get_column_concatenated_bytes_s3path_from_uuid(
                    location.file_id.bytes, dataset_name=self._dataset_id.name
                )
                bucket, key = path.replace("s3://", "").split("/", 1)
                buckets_keys.add((bucket, key))

        # pass the data to the multi proc management function
        buckets_keys_list = list(buckets_keys)
        column_files_byte_size = get_file_size_s3_multiproc(buckets_keys_list)
        return column_files_byte_size + par_dir_bytes

    @cached_property
    def dataset_size(self) -> int:
        """Total dataset partition size in bytes

        Returns:
            int: total dataset size in bytes
        """
        return self._get_dataset_partition_size()
