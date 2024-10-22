import os
import tempfile
import unittest
from contextlib import contextmanager
from typing import Any, Iterator, NamedTuple, Tuple
from unittest.mock import patch

import numpy as np
import pyarrow as pa  # type: ignore
import pyarrow.fs as pafs  # type: ignore
import pyarrow.parquet as papq  # type: ignore

from wicker.core.column_files import ColumnBytesFileWriter
from wicker.core.config import (
    FILESYSTEM_CONFIG,
    StorageDownloadConfig,
    WickerAwsS3Config,
    WickerConfig,
    WickerFileSystemConfig,
    WickerWandBConfig,
)
from wicker.core.datasets import FileSystemDataset, S3Dataset, build_dataset
from wicker.core.definitions import DatasetID, DatasetPartition
from wicker.core.storage import FileSystemDataStorage, S3PathFactory, WickerPathFactory
from wicker.schema import schema, serialization
from wicker.testing.storage import FakeS3DataStorage

FAKE_NAME = "dataset_name"
FAKE_VERSION = "0.0.1"
FAKE_PARTITION = "train"
FAKE_DATASET_ID = DatasetID(name=FAKE_NAME, version=FAKE_VERSION)
FAKE_DATASET_PARTITION = DatasetPartition(FAKE_DATASET_ID, partition=FAKE_PARTITION)

FAKE_SHAPE = (4, 4)
FAKE_DTYPE = "float64"
FAKE_NUMPY_CODEC = schema.WickerNumpyCodec(FAKE_SHAPE, FAKE_DTYPE)
FAKE_SCHEMA = schema.DatasetSchema(
    primary_keys=["foo"],
    fields=[
        schema.StringField("foo"),
        schema.NumpyField("np_arr", shape=FAKE_SHAPE, dtype=FAKE_DTYPE),
    ],
)
FAKE_DATA = [{"foo": f"bar{i}", "np_arr": np.eye(4)} for i in range(1000)]


def build_mock_wicker_config(tmpdir: str) -> WickerConfig:
    """Helper function to build WickerConfig objects to use as unit test mocks."""
    return WickerConfig(
        raw={},
        aws_s3_config=WickerAwsS3Config.from_json({}),
        filesystem_configs=[
            WickerFileSystemConfig.from_json(
                {
                    "config_name": "filesystem_1",
                    "prefix_replace_path": "",
                    "root_datasets_path": os.path.join(tmpdir, "fake_data"),
                }
            ),
        ],
        storage_download_config=StorageDownloadConfig.from_json({}),
        wandb_config=WickerWandBConfig.from_json({}),
    )


@contextmanager
def cwd(path):
    """Changes the current working directory, and returns to the previous directory afterwards"""
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


class TestFileSystemDataset(unittest.TestCase):
    @contextmanager
    def _setup_storage(self) -> Iterator[Tuple[FileSystemDataStorage, WickerPathFactory, str]]:
        with tempfile.TemporaryDirectory() as tmpdir, cwd(tmpdir):
            fake_local_fs_storage = FileSystemDataStorage()
            fake_local_path_factory = WickerPathFactory(root_path=os.path.join(tmpdir, "fake_data"))
            fake_s3_path_factory = S3PathFactory()
            fake_s3_storage = FakeS3DataStorage(tmpdir=tmpdir)
            with ColumnBytesFileWriter(
                storage=fake_s3_storage,
                s3_path_factory=fake_s3_path_factory,
                target_file_rowgroup_size=10,
            ) as writer:
                locs = [
                    writer.add("np_arr", FAKE_NUMPY_CODEC.validate_and_encode_object(data["np_arr"]))  # type: ignore
                    for data in FAKE_DATA
                ]

            arrow_metadata_table = pa.Table.from_pydict(
                {"foo": [data["foo"] for data in FAKE_DATA], "np_arr": [loc.to_bytes() for loc in locs]}
            )
            metadata_table_path = os.path.join(
                tmpdir, fake_local_path_factory._get_dataset_partition_path(FAKE_DATASET_PARTITION)
            )
            os.makedirs(os.path.dirname(metadata_table_path), exist_ok=True)
            papq.write_table(arrow_metadata_table, metadata_table_path)

            # The mock storage class here actually writes to local storage, so we can use it in the test.
            fake_s3_storage.put_object_s3(
                serialization.dumps(FAKE_SCHEMA).encode("utf-8"),
                fake_local_path_factory._get_dataset_schema_path(FAKE_DATASET_ID),
            )
            yield fake_local_fs_storage, fake_local_path_factory, tmpdir

    def test_filesystem_dataset(self):
        with self._setup_storage() as (fake_local_storage, fake_local_path_factory, tmpdir):
            ds = FileSystemDataset(
                FAKE_NAME,
                FAKE_VERSION,
                FAKE_PARTITION,
                fake_local_path_factory,
                fake_local_storage,
            )
            for i in range(len(FAKE_DATA)):
                retrieved = ds[i]
                reference = FAKE_DATA[i]
                self.assertEqual(retrieved["foo"], reference["foo"])
                np.testing.assert_array_equal(retrieved["np_arr"], reference["np_arr"])

            # Also double-check that the builder function is working correctly.
            with patch("wicker.core.datasets.get_config") as mock_get_config:
                mock_get_config.return_value = build_mock_wicker_config(tmpdir)
                ds2 = build_dataset(
                    FILESYSTEM_CONFIG,
                    FAKE_NAME,
                    FAKE_VERSION,
                    FAKE_PARTITION,
                    config_name="filesystem_1",
                )
                for i in range(len(FAKE_DATA)):
                    retrieved = ds2[i]
                    reference = FAKE_DATA[i]
                    self.assertEqual(retrieved["foo"], reference["foo"])
                    np.testing.assert_array_equal(retrieved["np_arr"], reference["np_arr"])


class TestS3Dataset(unittest.TestCase):
    @contextmanager
    def _setup_storage(self) -> Iterator[Tuple[FakeS3DataStorage, S3PathFactory, str]]:
        """Context manager that sets up a local directory to mimic S3 storage for a committed dataset,
        and returns a tuple of (S3DataStorage, S3PathFactory, tmpdir_path) for the caller to use as
        fixtures in their tests.
        """
        with tempfile.TemporaryDirectory() as tmpdir, cwd(tmpdir):
            fake_s3_storage = FakeS3DataStorage(tmpdir=tmpdir)
            fake_s3_path_factory = S3PathFactory()
            with ColumnBytesFileWriter(
                storage=fake_s3_storage,
                s3_path_factory=fake_s3_path_factory,
                target_file_rowgroup_size=10,
            ) as writer:
                locs = [
                    writer.add("np_arr", FAKE_NUMPY_CODEC.validate_and_encode_object(data["np_arr"]))  # type: ignore
                    for data in FAKE_DATA
                ]
            arrow_metadata_table = pa.Table.from_pydict(
                {
                    "foo": [data["foo"] for data in FAKE_DATA],
                    "np_arr": [loc.to_bytes() for loc in locs],
                }
            )
            metadata_table_path = os.path.join(
                tmpdir,
                fake_s3_path_factory.get_dataset_partition_path(FAKE_DATASET_PARTITION, s3_prefix=False),
                "part-1.parquet",
            )
            os.makedirs(os.path.dirname(metadata_table_path), exist_ok=True)
            papq.write_table(
                arrow_metadata_table,
                metadata_table_path,
            )
            fake_s3_storage.put_object_s3(
                serialization.dumps(FAKE_SCHEMA).encode("utf-8"),
                fake_s3_path_factory.get_dataset_schema_path(FAKE_DATASET_ID),
            )
            yield fake_s3_storage, fake_s3_path_factory, tmpdir

    def test_dataset(self) -> None:
        with self._setup_storage() as (fake_s3_storage, fake_s3_path_factory, tmpdir):
            ds = S3Dataset(
                FAKE_NAME,
                FAKE_VERSION,
                FAKE_PARTITION,
                local_cache_path_prefix=tmpdir,
                columns_to_load=None,
                storage=fake_s3_storage,
                s3_path_factory=fake_s3_path_factory,
                pa_filesystem=pafs.LocalFileSystem(),
            )

            for i in range(len(FAKE_DATA)):
                retrieved = ds[i]
                reference = FAKE_DATA[i]
                self.assertEqual(retrieved["foo"], reference["foo"])
                np.testing.assert_array_equal(retrieved["np_arr"], reference["np_arr"])

    def test_filters_dataset(self) -> None:
        filtered_value_list = [f"bar{i}" for i in range(100)]
        with self._setup_storage() as (fake_s3_storage, fake_s3_path_factory, tmpdir):
            ds = S3Dataset(
                FAKE_NAME,
                FAKE_VERSION,
                FAKE_PARTITION,
                local_cache_path_prefix=tmpdir,
                columns_to_load=None,
                storage=fake_s3_storage,
                s3_path_factory=fake_s3_path_factory,
                pa_filesystem=pafs.LocalFileSystem(),
                filters=[("foo", "in", filtered_value_list)],
            )
            self.assertEqual(len(ds), len(filtered_value_list))
            retrieved_values_list = [ds[i]["foo"] for i in range(len(ds))]
            retrieved_values_list.sort()
            filtered_value_list.sort()
            self.assertListEqual(retrieved_values_list, filtered_value_list)

    def test_dataset_size(self) -> None:
        with self._setup_storage() as (fake_s3_storage, fake_s3_path_factory, tmpdir):
            # overwrite the mocked resource function using the fake storage
            class FakeResponse(NamedTuple):
                content_length: int

            # we do this to mock out using boto3, we use boto3 on the dataset
            # size because we can get just file metadata but we only mock
            # out the file storage pull usually to mock out boto3 we sub in
            # a replacement function that uses the fake storage
            class MockedS3Resource:
                def __init__(self) -> None:
                    pass

                def Object(self, bucket: str, key: str) -> FakeResponse:
                    full_path = os.path.join("s3://", bucket, key)
                    data = fake_s3_storage.fetch_obj_s3(full_path)
                    return FakeResponse(content_length=len(data))

            def mock_resource_returner(_: Any):
                return MockedS3Resource()

            with patch("wicker.core.datasets.boto3.resource", mock_resource_returner):
                ds = S3Dataset(
                    FAKE_NAME,
                    FAKE_VERSION,
                    FAKE_PARTITION,
                    local_cache_path_prefix=tmpdir,
                    columns_to_load=None,
                    storage=fake_s3_storage,
                    s3_path_factory=fake_s3_path_factory,
                    pa_filesystem=pafs.LocalFileSystem(),
                )

                # sub this in to get the local size of the parquet dir
                def _get_parquet_dir_size_mocked():
                    def get_parquet_size(path="."):
                        total = 0
                        with os.scandir(path) as it:
                            for entry in it:
                                if entry.is_file() and ".parquet" in entry.name:
                                    total += entry.stat().st_size
                                elif entry.is_dir():
                                    total += get_parquet_size(entry.path)
                        return total

                    return get_parquet_size(fake_s3_storage._tmpdir)

                ds._get_parquet_dir_size = _get_parquet_dir_size_mocked  # type: ignore

                dataset_size = ds.dataset_size

                # get the expected size, all of the col files plus pyarrow table
                def get_dir_size(path="."):
                    total = 0
                    with os.scandir(path) as it:
                        for entry in it:
                            if entry.is_file() and ".json" not in entry.name:
                                total += entry.stat().st_size
                            elif entry.is_dir():
                                total += get_dir_size(entry.path)
                    return total

                expected_bytes = get_dir_size(fake_s3_storage._tmpdir)
                assert expected_bytes == dataset_size
