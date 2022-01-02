"""Classes handling the shuffling of data in S3 when committing a dataset.

When committing a dataset, we sort the data by primary_key before materializing in S3
as Parquet files.

1. The ShuffleJob is the unit of work, and it is just an ordered set of Examples that should
be bundled together in one Parquet file

2. The ShuffleJobFactory produces ShuffleJobs, using a DatasetWriter object to retrieve the
written examples for divvying up as ShuffleJobs.

3. ShuffleWorkers receive the ShuffleJobs and perform the act of retrieving the data and
persisting the data into S3 as Parquet files (one for each ShuffleJob)
"""
from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import os
import pickle
import tempfile
from typing import Any, Dict, Generator, List, Optional, Tuple

import boto3
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as papq

from wicker.core.column_files import ColumnBytesFileWriter
from wicker.core.definitions import DatasetID, DatasetPartition
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.core.writer import DatasetWriterBackend
from wicker.schema import serialization

# Maximum working set for each worker
DEFAULT_WORKER_MAX_WORKING_SET_SIZE = 16384


@dataclasses.dataclass
class ShuffleJob:
    """Represents all the shuffling operations that will happen for a given partition (train/eval/test) on a given
    compute shard."""

    dataset_partition: DatasetPartition
    files: List[Tuple[str, int]]


class ShuffleJobFactory:
    def __init__(
        self,
        writer_backend: DatasetWriterBackend,
        worker_max_working_set_size: int = DEFAULT_WORKER_MAX_WORKING_SET_SIZE,
    ):
        self.writer_backend = writer_backend

        # Factory configurations
        self.worker_max_working_set_size = worker_max_working_set_size

    def build_shuffle_jobs(self, dataset_id: DatasetID) -> Generator[ShuffleJob, None, None]:
        # Initialize with first item
        example_keys = self.writer_backend._metadata_db.scan_sorted(dataset_id)
        try:
            initial_key = next(example_keys)
        except StopIteration:
            return
        job = ShuffleJob(
            dataset_partition=DatasetPartition(dataset_id=dataset_id, partition=initial_key.partition),
            files=[(initial_key.row_data_path, initial_key.row_size)],
        )

        # Yield ShuffleJobs as we accumulate ExampleKeys, where each ShuffleJob is upper-bounded in size by
        # self.worker_max_working_set_size and has all ExampleKeys from the same partition
        for example_key in example_keys:
            if (
                example_key.partition == job.dataset_partition.partition
                and len(job.files) < self.worker_max_working_set_size
            ):
                job.files.append((example_key.row_data_path, example_key.row_size))
                continue

            # Yield job and construct new job to keep iterating
            yield job
            job = ShuffleJob(
                dataset_partition=DatasetPartition(dataset_id=dataset_id, partition=example_key.partition),
                files=[(example_key.row_data_path, example_key.row_size)],
            )
        else:
            yield job


_download_thread_session: Optional[boto3.session.Session] = None
_download_thread_client: Optional[S3DataStorage] = None


def _initialize_download_thread():
    global _download_thread_session
    global _download_thread_client
    if _download_thread_client is None:
        _download_thread_session = boto3.session.Session()
        _download_thread_client = S3DataStorage(session=_download_thread_session)


class ShuffleWorker:
    def __init__(
        self,
        target_rowgroup_bytes_size: int = int(256e6),
        max_worker_threads: int = 16,
        max_memory_usage_bytes: int = int(2e9),
        storage: S3DataStorage = S3DataStorage(),
        s3_path_factory: S3PathFactory = S3PathFactory(),
    ):
        self.target_rowgroup_bytes_size = target_rowgroup_bytes_size
        self.max_worker_threads = max_worker_threads
        self.max_memory_usage_bytes = max_memory_usage_bytes
        self.storage = storage
        self.s3_path_factory = s3_path_factory

    def _download_files(self, job: ShuffleJob) -> Generator[Dict[str, Any], None, None]:
        """Downloads the files in a ShuffleJob, yielding a generator of Dict[str, Any]

        Internally, this function maintains a buffer of lookahead downloads to execute downloads
        in parallel over a ThreadPoolExecutor, up to a maximum of `max_memory_usage_bytes` bytes.

        Args:
            job (ShuffleJob): job to download files for

        Yields:
            Generator[Dict[str, Any], None, None]: stream of Dict[str, Any] from downloading the files
                in order
        """
        # TODO(jchia): Add retries here
        def _download_file(filepath: str) -> Dict[str, Any]:
            assert _download_thread_client is not None
            return pickle.loads(_download_thread_client.fetch_obj_s3(filepath))

        with concurrent.futures.ThreadPoolExecutor(
            self.max_worker_threads, initializer=_initialize_download_thread
        ) as executor:
            buffer: Dict[int, Tuple[concurrent.futures.Future[Dict[str, Any]], int]] = {}
            buffer_size = 0
            buffer_index = 0
            for current_index in range(len(job.files)):
                while buffer_index < len(job.files) and buffer_size < self.max_memory_usage_bytes:
                    filepath, file_size = job.files[buffer_index]
                    buffer[buffer_index] = (executor.submit(_download_file, filepath), file_size)
                    buffer_size += file_size
                    buffer_index += 1
                current_future, current_file_size = buffer[current_index]
                yield current_future.result()
                buffer_size -= current_file_size
                del buffer[current_index]

    def _estimate_target_file_rowgroup_size(
        self,
        job: ShuffleJob,
        target_rowgroup_size_bytes: int = int(256e6),
        min_target_rowgroup_size: int = 16,
    ) -> int:
        """Estimates the number of rows to include in each rowgroup using a target size for the rowgroup

        :param job: job to estimate
        :param target_rowgroup_size_bytes: target size in bytes of a rowgroup, defaults to 256MB
        :param min_target_rowgroup_size: minimum number of rows in a rowgroup
        :return: target number of rows in a rowgroup
        """
        average_filesize = sum([size for _, size in job.files]) / len(job.files)
        return max(min_target_rowgroup_size, int(target_rowgroup_size_bytes / average_filesize))

    def process_job(self, job: ShuffleJob) -> pa.Table:
        # Load dataset schema
        dataset_schema = serialization.loads(
            self.storage.fetch_obj_s3(
                self.s3_path_factory.get_dataset_schema_path(job.dataset_partition.dataset_id)
            ).decode("utf-8")
        )

        # Estimate how many rows to add to each ColumnBytesFile
        target_file_rowgroup_size = self._estimate_target_file_rowgroup_size(job)

        # Initialize data containers to dump into parquet
        heavy_pointer_columns = dataset_schema.get_pointer_columns()
        metadata_columns = dataset_schema.get_non_pointer_columns()
        parquet_metadata: Dict[str, List[Any]] = collections.defaultdict(list)

        # Parse each row, uploading heavy_pointer bytes to S3 and storing only pointers
        # in parquet_metadata
        with ColumnBytesFileWriter(
            self.storage,
            self.s3_path_factory,
            target_file_rowgroup_size=target_file_rowgroup_size,
        ) as writer:
            for data in self._download_files(job):
                for col in metadata_columns:
                    parquet_metadata[col].append(data[col])
                for col in heavy_pointer_columns:
                    loc = writer.add(col, data[col])
                    parquet_metadata[col].append(loc.to_bytes())

        # Save parquet_metadata as a PyArrow Table
        assert len({len(parquet_metadata[col]) for col in parquet_metadata}) == 1, "All columns must have same length"
        return pa.Table.from_pydict(parquet_metadata)


def save_index(
    dataset_name: str,
    dataset_version: str,
    final_indices: Dict[str, pa.Table],
    s3_path_factory: Optional[S3PathFactory] = None,
    s3_storage: Optional[S3DataStorage] = None,
) -> None:
    """Saves a generated final_index into persistent storage

    :param dataset_name: Name of the dataset
    :param dataset_version: Version of the dataset
    :param final_index: Dictionary of pandas dataframes which is the finalized index
    :param pyarrow_filesystem: PyArrow filesystem to use, defaults to None
    :param s3_path_factory: S3PathFactory to use
    :param s3_storage: S3DataStorage to use
    """
    s3_storage = s3_storage if s3_storage is not None else S3DataStorage()
    s3_path_factory = s3_path_factory if s3_path_factory is not None else S3PathFactory()
    for partition_name in final_indices:
        dataset_partition = DatasetPartition(
            dataset_id=DatasetID(
                name=dataset_name,
                version=dataset_version,
            ),
            partition=partition_name,
        )

        parquet_folder = s3_path_factory.get_dataset_partition_path(dataset_partition, s3_prefix=True)
        parquet_path = os.path.join(parquet_folder, "part-0.parquet")

        # Write the Parquet file as one file locally, then upload to S3
        with tempfile.NamedTemporaryFile() as tmpfile:
            pa_table = final_indices[partition_name]
            papq.write_table(
                pa_table,
                tmpfile.name,
                compression="zstd",
                row_group_size=None,
                filesystem=pafs.LocalFileSystem(),
                # We skip writing statistics since it bloats the file, and we don't actually run any queries
                # on the Parquet files that could make use of predicate push-down
                write_statistics=False,
            )
            s3_storage.put_file_s3(
                tmpfile.name,
                parquet_path,
            )

    return None
