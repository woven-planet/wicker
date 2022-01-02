from __future__ import annotations

import abc
import contextlib
import dataclasses
import hashlib
import os
import pickle
import threading
import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Type, Union

from wicker.core.definitions import DatasetDefinition, DatasetID
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.schema import dataparsing, serialization


@dataclasses.dataclass
class ExampleKey:
    """Unique identifier for one example."""

    # Name of the partition (usually train/eval/test)
    partition: str
    # Values of the primary keys for the example (in order of key precedence)
    primary_key_values: List[Union[str, int]]

    def hash(self) -> str:
        return hashlib.sha256(
            "/".join([self.partition, *[str(obj) for obj in self.primary_key_values]]).encode("utf-8")
        ).hexdigest()


@dataclasses.dataclass
class MetadataDatabaseScanRow:
    """Container for data obtained by scanning the MetadataDatabase"""

    partition: str
    row_data_path: str
    row_size: int


class AbstractDatasetWriterMetadataDatabase:
    """Database for storing metadata, used from inside the DatasetWriterBackend.

    NOTE: Implementors - this is the main implementation integration point for creating a new kind of DatasetWriter.
    """

    @abc.abstractmethod
    def save_row_metadata(self, dataset_id: DatasetID, key: ExampleKey, location: str, row_size: int) -> None:
        """Saves a row in the metadata database, marking it as having been uploaded to S3 and
        ready for shuffling.

        :param dataset_id: The ID of the dataset to save to
        :param key: The key of the example
        :param location: The location of the example in S3
        :param row_size: The size of the file in S3
        """
        pass

    @abc.abstractmethod
    def scan_sorted(self, dataset_id: DatasetID) -> Generator[MetadataDatabaseScanRow, None, None]:
        """Scans the MetadataDatabase for a **SORTED** stream of MetadataDatabaseScanRows for a given dataset.
        The stream is sorted by partition first, and then primary_key_values second.

        Should be fast O(minutes) to perform as this will be called from a single machine to assign chunks to jobs
        to run.

        :param dataset_id: The dataset to scan the metadata database for
        :return: a Generator of ExampleDBRows in **SORTED** partition + primary_key order
        """
        pass


class DatasetWriterBackend:
    """The backend for a DatasetWriter.

    Responsible for saving and retrieving data used during the dataset writing and committing workflow.
    """

    def __init__(
        self,
        s3_path_factory: S3PathFactory,
        s3_storage: S3DataStorage,
        metadata_database: AbstractDatasetWriterMetadataDatabase,
    ):
        self._s3_path_factory = s3_path_factory
        self._s3_storage = s3_storage
        self._metadata_db = metadata_database

    def save_row(self, dataset_id: DatasetID, key: ExampleKey, raw_data: Dict[str, Any]) -> None:
        """Adds an example to the backend

        :param dataset_id: ID of the dataset to save the row to
        :param key: Key of the example to write
        :param raw_data: raw data for the example that conforms to the schema provided at initialization
        """
        hashed_row_key = key.hash()
        pickled_row = pickle.dumps(raw_data)  # TODO(jchia): Do we want a more sophisticated storage format here?
        row_s3_path = os.path.join(
            self._s3_path_factory.get_temporary_row_files_path(dataset_id),
            hashed_row_key,
        )

        # Persist data in S3 and in MetadataDatabase
        self._s3_storage.put_object_s3(pickled_row, row_s3_path)
        self._metadata_db.save_row_metadata(dataset_id, key, row_s3_path, len(pickled_row))

    def commit_schema(
        self,
        dataset_definition: DatasetDefinition,
    ) -> None:
        """Write the schema to the backend as part of the commit step."""
        schema_path = self._s3_path_factory.get_dataset_schema_path(dataset_definition.identifier)
        serialized_schema = serialization.dumps(dataset_definition.schema)
        self._s3_storage.put_object_s3(serialized_schema.encode(), schema_path)


DEFAULT_BUFFER_SIZE_LIMIT = 1000


class DatasetWriter:
    """DatasetWriter providing async writing functionality. Implementors should override
    the ._save_row_impl method to define functionality for saving each individual row from inside the
    async thread executors.
    """

    def __init__(
        self,
        dataset_definition: DatasetDefinition,
        metadata_database: AbstractDatasetWriterMetadataDatabase,
        s3_path_factory: Optional[S3PathFactory] = None,
        s3_storage: Optional[S3DataStorage] = None,
        buffer_size_limit: int = DEFAULT_BUFFER_SIZE_LIMIT,
        executor: Optional[Executor] = None,
        wait_flush_timeout_seconds: int = 300,
    ):
        """Create a new DatasetWriter

        :param dataset_definition: definition of the dataset
        :param s3_path_factory: factory for s3 paths
        :param s3_storage: S3-compatible storage for storing data
        :param buffer_size_limit: size limit to the number of examples buffered in-memory, defaults to 1000
        :param executor: concurrent.futures.Executor to use for async writing, defaults to None
        :param wait_flush_timeout_seconds: number of seconds to wait before timing out on flushing
            all examples, defaults to 10
        """
        self.dataset_definition = dataset_definition
        self.backend = DatasetWriterBackend(
            s3_path_factory if s3_path_factory is not None else S3PathFactory(),
            s3_storage if s3_storage is not None else S3DataStorage(),
            metadata_database,
        )

        self.buffer: List[Tuple[ExampleKey, Dict[str, Any]]] = []
        self.buffer_size_limit = buffer_size_limit

        self.wait_flush_timeout_seconds = wait_flush_timeout_seconds
        self.executor = executor if executor is not None else ThreadPoolExecutor(max_workers=16)
        self.writes_in_flight: Dict[str, Dict[str, Any]] = {}
        self.flush_condition_variable = threading.Condition()

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.flush(block=True)

    def __del__(self) -> None:
        self.flush(block=True)

    def add_example(self, partition_name: str, raw_data: Dict[str, Any]) -> None:
        """Adds an example to the writer

        :param partition_name: partition name where the example belongs
        :param raw_data: raw data for the example that conforms to the schema provided at initialization
        """
        # Run sanity checks on the data, fill empty fields.
        ex = dataparsing.parse_example(raw_data, self.dataset_definition.schema)
        example_key = ExampleKey(
            partition=partition_name, primary_key_values=[ex[k] for k in self.dataset_definition.schema.primary_keys]
        )
        self.buffer.append((example_key, ex))

        # Flush buffer to persistent storage if at size limit
        if len(self.buffer) > self.buffer_size_limit:
            self.flush(block=False)

    def flush(self, block: bool = True) -> None:
        """Flushes the writer

        :param block: whether to block on flushing all currently buffered examples, defaults to True
        :raises TimeoutError: timing out on flushing all examples
        """
        batch_data = self.buffer
        self.buffer = []
        self._save_batch_data(batch_data)

        if block:
            with self._block_on_writes_in_flight(max_in_flight=0, timeout_seconds=self.wait_flush_timeout_seconds):
                pass

    @contextlib.contextmanager
    def _block_on_writes_in_flight(self, max_in_flight: int = 0, timeout_seconds: int = 60) -> Iterator[None]:
        """Blocks until number of writes in flight <= max_in_flight and yields control to the with block

        Usage:
        >>> with self._block_on_writes_in_flight(max_in_flight=10):
        >>>     # Code here holds the exclusive lock on self.flush_condition_variable
        >>>     ...
        >>> # Code here has released the lock, and has no guarantees about the number of writes in flight

        :param max_in_flight: maximum number of writes in flight, defaults to 0
        :param timeout_seconds: maximum number of seconds to wait, defaults to 10
        :raises TimeoutError: if waiting for more than timeout_seconds
        """
        start_time = time.time()
        with self.flush_condition_variable:
            while len(self.writes_in_flight) > max_in_flight:
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(
                        f"Timed out while flushing dataset writes with {len(self.writes_in_flight)} writes in flight"
                    )
                self.flush_condition_variable.wait()
            yield

    def _save_row(self, key: ExampleKey, data: Dict[str, Any]) -> str:
        self.backend.save_row(self.dataset_definition.identifier, key, data)
        return key.hash()

    def _save_batch_data(self, batch_data: List[Tuple[ExampleKey, Dict[str, Any]]]) -> None:
        """Save a batch of data to persistent storage

        :param row_keys: Unique identifiers for each row Uses only 0-9A-F, can be used as a file name.
        :param batch_data: Batch of data
        """

        def done_callback(future: Future[str]) -> None:
            # TODO(jchia): We can add retries on failure by reappending to the buffer
            # this currently may raise a CancelledError or some Exception thrown by the .save_row function
            with self.flush_condition_variable:
                del self.writes_in_flight[future.result()]
                self.flush_condition_variable.notify()

        for key, data in batch_data:
            # Keep the number of writes in flight always smaller than 2 * self.buffer_size_limit
            with self._block_on_writes_in_flight(
                max_in_flight=2 * self.buffer_size_limit,
                timeout_seconds=self.wait_flush_timeout_seconds,
            ):
                self.writes_in_flight[key.hash()] = data
            future = self.executor.submit(self._save_row, key, data)
            future.add_done_callback(done_callback)
