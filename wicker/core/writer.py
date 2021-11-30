from __future__ import annotations

import abc
import contextlib
import dataclasses
import hashlib
import json
import threading
import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

from wicker.core.definitions import DatasetDefinition
from wicker.schema import dataparsing


@dataclasses.dataclass
class ExampleDBKey:
    """Unique identifier for one example."""

    # Name of the partition (usually train/eval/test)
    partition: str
    # Values of the primary keys for the example (in order of key precedence)
    primary_key_values: List[Any]

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(self, data: str) -> ExampleDBKey:
        obj = json.loads(data)
        return ExampleDBKey(partition=obj["partition"], primary_key_values=obj["primary_key_values"])


class DatasetWriter:
    """Abstract class for writing a dataset"""

    @abc.abstractmethod
    def add_example(self, partition_name: str, raw_data: Dict[str, Any]) -> None:
        """Adds an example to the writer

        :param partition_name: partition name where the example belongs
        :param raw_data: raw data for the example that conforms to the schema provided at initialization
        """
        pass

    @abc.abstractmethod
    def commit(self) -> None:
        """Commits the added examples as a materialized dataset"""
        pass


DEFAULT_BUFFER_SIZE_LIMIT = 1000


class AsyncDatasetWriter(DatasetWriter):
    """Superclass providing async writing functionality to DatasetWriters. Implementors should override
    the ._save_row_impl method to define functionality for saving each individual row from inside the
    async thread executors.
    """

    def __init__(
        self,
        dataset_definition: DatasetDefinition,
        buffer_size_limit: int = DEFAULT_BUFFER_SIZE_LIMIT,
        executor: Optional[Executor] = None,
        wait_flush_timeout_seconds: int = 10,
    ):
        """Create a new AsyncDatasetWriter

        :param dataset_definition: definition of the dataset
        :param buffer_size_limit: size limit to the number of examples buffered in-memory, defaults to 1000
        :param executor: concurrent.futures.Executor to use for async writing, defaults to None
        :param wait_flush_timeout_seconds: number of seconds to wait before timing out on flushing
            all examples, defaults to 10
        """
        self.dataset_definition = dataset_definition
        self.buffer: List[Tuple[ExampleDBKey, Dict[str, Any]]] = []
        self.buffer_size_limit = buffer_size_limit

        self.wait_flush_timeout_seconds = wait_flush_timeout_seconds
        self.executor = executor if executor is not None else ThreadPoolExecutor(max_workers=16)
        self.writes_in_flight: Dict[str, Dict[str, Any]] = {}
        self.flush_condition_variable = threading.Condition()

    def __enter__(self) -> AsyncDatasetWriter:
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
        example_key = ExampleDBKey(
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

    def _save_row(self, key: ExampleDBKey, hashed_row_key: str, data: Dict[str, Any]) -> str:
        self._save_row_impl(key, hashed_row_key, data)
        return hashed_row_key

    def _save_batch_data(self, batch_data: List[Tuple[ExampleDBKey, Dict[str, Any]]]) -> None:
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
                hashed_row_key = hashlib.sha256(key.to_json().encode()).hexdigest()
                self.writes_in_flight[hashed_row_key] = data
            future = self.executor.submit(self._save_row, key, hashed_row_key, data)
            future.add_done_callback(done_callback)

    @abc.abstractmethod
    def _save_row_impl(self, key: ExampleDBKey, hashed_row_key: str, data: Dict[str, Any]) -> None:
        """Subclasses should implement this method to save each individual row

        :param key: key of row
        :param hashed_row_key: hashed key for row
        :param data: validated data for row
        """
        pass
