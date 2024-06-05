import abc
import os
import subprocess
from functools import cached_property
from multiprocessing import Manager, Pool, cpu_count
from multiprocessing.managers import ValueProxy
from multiprocessing.pool import ThreadPool
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pyarrow  # type: ignore
import pyarrow.fs as pafs  # type: ignore
import pyarrow.parquet as papq  # type: ignore
import tqdm  # type: ignore

from wicker.core.column_files import ColumnBytesFileCache, ColumnBytesFileLocationV1
from wicker.core.config import get_config  # type: ignore
from wicker.core.definitions import DatasetDefinition, DatasetID, DatasetPartition
from wicker.core.shuffle import ShuffleWorker
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.schema import dataloading, serialization
from wicker.schema.schema import DatasetSchema

# How long to wait before timing out on filelocks in seconds
FILE_LOCK_TIMEOUT_SECONDS = 300


def get_file_size_s3_optional_copy_gcloud(
    input_tuple: Tuple[List[Tuple[str, str]], ValueProxy[int], Lock, bool, str, str]
):
    """Get file size of a list of s3 paths. Optionally copy to gcloud if specified

    Tuple structure - The tuple contains 5 parts denoted below.

        buckets_keys_chunks_local - The list of tuples denoting bucket and key of files on s3 to
        parse. Generally column files but will work with any data.

        sum_value - A python shared memory object (ctype) for storing the byte sum across all processes.

        lock - A shared memory lock for locking the sum_value during write procedure.

        local_output_loc - The local location to store files during transfer as intermediary.

        gcloud_output_loc - The output loc of where to put data on gcloud.

    Args:
        input_tuple: A tuple containing the relevant pieces of data used to grab, copy, and store byte size.
            Works across procs so needs a lock and shared memory space.

    Returns:
        Void function - does all modifications in place on the shared memory value.

    """
    buckets_keys_chunks_local, sum_value, lock, move_to_destination, local_output_loc, gcloud_output_loc = input_tuple
    # set up the s3 resource and client
    s3 = boto3.resource("s3")

    # chunk the process again between multiple threads on each proc
    # done to reduce i/o wait on each individual proc
    local_chunks = []
    local_chunk_nums = 500
    local_chunk_size = len(buckets_keys_chunks_local) // local_chunk_nums
    for i in range(0, local_chunk_nums):
        chunk = buckets_keys_chunks_local[i * local_chunk_size : (i + 1) * local_chunk_size]
        local_chunks.append((chunk, move_to_destination))

    # form and add in the last chunk
    last_chunk_size = len(buckets_keys_chunks_local) - (local_chunk_nums * local_chunk_size)
    last_chunk = buckets_keys_chunks_local[-last_chunk_size:]
    local_chunks.append((last_chunk, move_to_destination))

    def iterate_proc_chunk(input_tuple):
        local_len = 0
        chunk_list, move_to_destination_loc = input_tuple
        for bucket_key_loc in chunk_list:
            bucket_loc, key_loc = bucket_key_loc
            # get the byte length for the object
            byte_length = s3.Object(bucket_loc, key_loc).content_length
            if move_to_destination_loc:
                # establish session to copy data over to the host
                session = boto3.session.Session()
                client = session.client("s3")
                output_loc = os.path.join(local_output_loc, key_loc)
                if not os.path.exists(os.path.dirname(output_loc)):
                    os.makedirs(os.path.dirname(output_loc), exist_ok=True)
                if not os.path.exists(output_loc):
                    # download and then push to gcloud
                    client.download_file(bucket_loc, key_loc, output_loc)
                    gcloud_path = f"gs://{os.path.join(gcloud_output_loc, '__COLUMN_CONCATENATED_FILES__')}"
                    subprocess.run(f"gcloud storage cp -n {output_loc} {gcloud_path}", shell=True)
                    # delete file locally to not blow up size
                    subprocess.run(f"rm {output_loc}", shell=True)
            local_len += byte_length
        return local_len

    # set up thread pool with max threads
    thread_pool = ThreadPool()
    # run the threads and sum the results together
    # could use shared memory again but not worth it imo
    result = sum(list(tqdm.tqdm(thread_pool.map(iterate_proc_chunk, local_chunks))))
    with lock:
        sum_value.value += result


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
            self._arrow_table = papq.read_table(path, columns=self._columns_to_load, filesystem=self._pa_filesystem)
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

    def _copy_parquet_dir_gcloud(self, config) -> None:
        """Copy the parquet path to gcloud."""
        # copy the arrow table and the parquet files
        parquet_partition_path = self._s3_path_factory.get_dataset_partition_path(self._partition, s3_prefix=False)

        # have to cut off the end because we want entire dir size not just one partition
        # the entire dataset is all parquet files
        bucket_dataset, key_dataset = parquet_partition_path.replace("s3://", "").split("/", 1)
        key_dataset = key_dataset[::-1].split("/", 1)[1][::-1]
        s3_resource = boto3.resource("s3")
        bucket = s3_resource.Bucket(bucket_dataset)
        for obj in bucket.objects.filter(Prefix=key_dataset):
            if "lock" not in obj.key and "success" not in obj.key:
                dest_path = os.path.join(self._local_cache_path_prefix, obj.key)
                if not os.path.exists(os.path.dirname(dest_path)):
                    os.makedirs(os.path.dirname(dest_path))
                bucket.download_file(obj.key, dest_path)  # save to same path

        # upload dir to gcloud
        # hacky because we don't want "l5ml_datasets" present
        gcloud_key = key_dataset.split("/", 1)[1]
        gcloud_dest = f"gs://{config.gcloud_config.dataset_bucket}/{gcloud_key}"
        local_path = os.path.join(self._local_cache_path_prefix, key_dataset)
        subprocess.run(f"gcloud storage cp -r -n {local_path} {gcloud_dest}", shell=True)

    def _get_parquet_dir_size(self) -> int:
        """Get the parquet path and find all the files within, count their bytes

        Returns:
            int: bytes in parquet directory
        """
        # bytes size of arrow table not bytes in arrow table
        # bytes in arrow table is a method of arrow table but it doesn't
        # reflect the size of the file sizes stored on s3 just the loaded data
        arrow_path = self._s3_path_factory.get_dataset_partition_path(self._partition, s3_prefix=False)

        # have to cut off the end because we want entire dir size not just one partition
        # the entire dataset is all parquet files
        bucket, key = arrow_path.replace("s3://", "").split("/", 1)
        key = key[::-1].split("/", 1)[1][::-1]

        def get_folder_size(bucket, prefix):
            total_size = 0
            for obj in boto3.resource("s3").Bucket(bucket).objects.filter(Prefix=prefix):

                total_size += obj.size
            return total_size

        return get_folder_size(bucket, key)

    def _get_dataset_size(self, move_to_destination: bool = False):
        """Gets total size of the dataset in bits. Optinally copy files to gcloud,
        based on config location.

        Args:
            move_to_destination: Bool denoting if the data should be moved
                to gcloud during size parsing.

        Returns:
            int: total dataset size in bits
        """
        config = get_config()

        # intialize with size of parquet dir
        print("Parsing parquet and arrow dir for size.")
        par_dir_bytes = self._get_parquet_dir_size()

        # copy the parquet folder if move_to_destination is true
        if move_to_destination:
            print("Transferring parquet and arrow dir to gcloud.")
            self._copy_parquet_dir_gcloud(config)

        # need to know which columns are heavy pntr columns we go to for
        # byte adding, parse which are heavy pointers based off metadata
        # baked into schema
        schema = self.schema()
        heavy_pointer_cols = []
        for col_name in schema.get_all_column_names():
            column = schema.get_column(col_name)
            if column is not None:
                if column.is_heavy_pointer:
                    heavy_pointer_cols.append(col_name)

        # create arrow table for parsing
        print("Creating arrow table")
        arrow_table = self.arrow_table()

        buckets_keys = set()

        # ignore typing to avoid changing the typing of Shuffle Worker yet
        # ToDo: Change typing of ShuffleWorker and let it take local data storage
        worker = ShuffleWorker(storage=self._storage)  # type: ignore
        print("Processing through heavy pointers")
        for heavy_pntr_col in heavy_pointer_cols:
            print(f"Evaulating {heavy_pntr_col} for column file locations")
            # have to go through all data in arrow table col
            # this is because each row only knows where its col files are.
            # ToDo: Store set of col files in some metadata
            for location_bytes in tqdm.tqdm(arrow_table[heavy_pntr_col].to_pylist()):
                location = ColumnBytesFileLocationV1.from_bytes(location_bytes)
                path = worker.s3_path_factory.get_column_concatenated_bytes_s3path_from_uuid(
                    location.file_id.bytes, dataset_name=self._dataset_id.name
                )
                bucket, key = path.replace("s3://", "").split("/", 1)
                buckets_keys.add((bucket, key))

        # declare the variables neccessary to split
        # out between multiple procs
        buckets_keys_chunks = []
        manager = Manager()
        total_size = manager.Value("i", 0)
        lock = manager.Lock()
        buckets_keys_list = list(buckets_keys)
        chunk_size = 500
        total_len_chunks = len(buckets_keys_list) // chunk_size
        # iterate through the chunks and form each chunk
        for i in range(0, total_len_chunks):
            chunk = buckets_keys_list[i * chunk_size : (i + 1) * chunk_size]
            buckets_keys_chunks.append(
                (
                    chunk,
                    total_size,
                    lock,
                    move_to_destination,
                    self._local_cache_path_prefix,
                    config.gcloud_config.dataset_bucket,
                )
            )

        # form the last chunk based on what data is left
        last_chunk_size = len(buckets_keys_chunks) - (total_len_chunks * chunk_size)
        last_chunk = buckets_keys_list[-last_chunk_size:]
        buckets_keys_chunks.append(
            (
                last_chunk,
                total_size,
                lock,
                move_to_destination,
                self._local_cache_path_prefix,
                config.gcloud_config.dataset_bucket,
            )
        )

        print("Grabbing file information from s3 heads")
        # set the proc pool up and distribute the data between
        # along with shared values
        pool = Pool(cpu_count() - 1)
        pool.map(get_file_size_s3_optional_copy_gcloud, buckets_keys_chunks)
        pool.close()
        pool.join()
        # add the value of the shared memory and the parquet dir bytes
        # return that value to the user
        return total_size.value + par_dir_bytes

    @cached_property
    def dataset_size(self) -> int:
        """Total dataset size in bits

        Returns:
            int: total dataset size in bits
        """
        return self._get_dataset_size()
