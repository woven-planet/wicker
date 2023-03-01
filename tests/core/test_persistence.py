import copy
import os
import random
import uuid
from typing import Any, Dict, Iterable, List, Tuple
from unittest.mock import patch

import pyarrow.parquet as papq
import pytest

from wicker import schema
from wicker.core.config import get_config
from wicker.core.persistance import (
    BasicPersistor,
    ColumnBytesFileWriter,
    ParsedExample,
    PointerParsedExample,
)
from wicker.core.storage import S3PathFactory
from wicker.schema.schema import DatasetSchema
from wicker.testing.storage import FakeS3DataStorage

DATASET_NAME = "dataset"
DATASET_VERSION = "0.0.1"
SCHEMA = schema.DatasetSchema(
    primary_keys=["global_index", "bar", "foo"],
    fields=[
        schema.IntField("global_index"),
        schema.IntField("foo"),
        schema.StringField("bar"),
        schema.BytesField("bytescol"),
    ],
)
EXAMPLES = [
    (
        "train" if i % 2 == 0 else "test",
        {
            "global_index": i,
            "foo": random.randint(0, 10000),
            "bar": str(uuid.uuid4()),
            "bytescol": b"0",
        },
    )
    for i in range(10000)
]
# Examples with a duplicated key
EXAMPLES_DUPES = copy.deepcopy(EXAMPLES)


@pytest.fixture
def mock_basic_persistor(request, tmpdir) -> Tuple[BasicPersistor, str]:
    storage = request.param.get("storage", FakeS3DataStorage(tmpdir=tmpdir))
    path_factory = request.param.get("path_factory", S3PathFactory())
    return BasicPersistor(storage, path_factory), tmpdir


def assert_written_correctness(tmpdir: str) -> None:
    """Asserts that all files are written as expected by the L5MLDatastore"""
    # Check that files are correctly written locally by Spark/Parquet with a _SUCCESS marker file
    prefix = get_config().aws_s3_config.s3_datasets_path.replace("s3://", "")
    assert DATASET_NAME in os.listdir(os.path.join(tmpdir, prefix))
    assert DATASET_VERSION in os.listdir(os.path.join(tmpdir, prefix, DATASET_NAME))
    for partition in ["train", "test"]:
        columns_path = os.path.join(tmpdir, prefix, "__COLUMN_CONCATENATED_FILES__")
        all_read_bytes = b""
        for filename in os.listdir(columns_path):
            concatenated_bytes_filepath = os.path.join(columns_path, filename)
            with open(concatenated_bytes_filepath, "rb") as bytescol_file:
                all_read_bytes += bytescol_file.read()
        assert all_read_bytes == b"0" * 10000

        # Load parquet file and assert ordering of primary_key
        assert f"{partition}.parquet" in os.listdir(os.path.join(tmpdir, prefix, DATASET_NAME, DATASET_VERSION))
        tbl = papq.read_table(os.path.join(tmpdir, prefix, DATASET_NAME, DATASET_VERSION, f"{partition}.parquet"))
        foobar = [
            (glo_idx.as_py(), barval.as_py(), fooval.as_py())
            for glo_idx, fooval, barval in zip(tbl["global_index"], tbl["foo"], tbl["bar"])
        ]
        assert foobar == sorted(foobar)


@pytest.mark.parametrize(
    "mock_basic_persistor, dataset_name, dataset_version, dataset_schema, dataset",
    [({}, DATASET_NAME, DATASET_VERSION, SCHEMA, copy.deepcopy(EXAMPLES_DUPES))],
    indirect=["mock_basic_persistor"],
)
def test_basic_persistor_no_shuffle(
    mock_basic_persistor: Tuple[BasicPersistor, str],
    dataset_name: str,
    dataset_version: str,
    dataset_schema: DatasetSchema,
    dataset: List[Tuple[str, Dict[str, Any]]],
):
    """
    Test if the basic persistor can persist data in the format we have established.

    Ensure we read the right file locations, the right amount of bytes,
    and the ordering is correct.
    """
    # in order to assert that we are not shuffling we are going to sub out the
    # persist partition function and get average distance on global index
    # if it is == 2 (ie: samples are adjacent in partitions) then shuffling has occured
    def mock_persist_wicker_partition(
        self,
        spark_partition_iter: Iterable[Tuple[str, ParsedExample]],
        schema: schema.DatasetSchema,
        s3_storage: FakeS3DataStorage,
        s3_path_factory: S3PathFactory,
        target_max_column_file_numrows: int = 50,
    ) -> Iterable[Tuple[str, PointerParsedExample]]:
        # set up the global sum and counter for calcing mean
        global_sum = 0
        global_counter = 0
        # we still have to do all of the regular logic to test writing
        column_bytes_file_writers: Dict[str, ColumnBytesFileWriter] = {}
        heavy_pointer_columns = schema.get_pointer_columns()
        metadata_columns = schema.get_non_pointer_columns()
        previous_value, previous_parition = None, None

        for partition, example in spark_partition_iter:
            # if the previous value is unset or the parition has changed
            if not previous_value or previous_parition != partition:
                previous_value = example["global_index"]
                previous_parition = partition
            # if we can calculate the distance because we are on same parition
            # and the previous value is not None
            else:
                current_diff = abs(example["global_index"] - previous_value)
                previous_value = example["global_index"]
                previous_parition = partition
                global_sum += current_diff
                global_counter += 1
            # Create ColumnBytesFileWriter lazily as required, for each partition
            if partition not in column_bytes_file_writers:
                column_bytes_file_writers[partition] = ColumnBytesFileWriter(
                    s3_storage,
                    s3_path_factory,
                    target_file_rowgroup_size=target_max_column_file_numrows,
                )

            # Write to ColumnBytesFileWriter and return only metadata + heavy-pointers
            parquet_metadata: Dict[str, Any] = {col: example[col] for col in metadata_columns}
            for col in heavy_pointer_columns:
                loc = column_bytes_file_writers[partition].add(col, example[col])
                parquet_metadata[col] = loc.to_bytes()
            yield partition, parquet_metadata

        # Flush all writers when finished
        for partition in column_bytes_file_writers:
            column_bytes_file_writers[partition].close()
        # assert that we are at mean 2 and that we have not shuffled
        mean = global_sum / global_counter
        assert mean == 2.0

    with patch("wicker.core.persistance.AbstractDataPersistor.persist_wicker_partition", mock_persist_wicker_partition):
        # create the mock basic persistor
        mock_basic_persistor_obj, tempdir = mock_basic_persistor
        # persist the dataset
        mock_basic_persistor_obj.persist_wicker_dataset(dataset_name, dataset_version, dataset_schema, dataset)
        # assert the dataset is correctly written
        assert_written_correctness(tempdir)


@pytest.mark.parametrize(
    "mock_basic_persistor, dataset_name, dataset_version, dataset_schema, dataset",
    [({}, DATASET_NAME, DATASET_VERSION, SCHEMA, copy.deepcopy(EXAMPLES_DUPES))],
    indirect=["mock_basic_persistor"],
)
def test_basic_persistor_shuffle(
    mock_basic_persistor: Tuple[BasicPersistor, str],
    dataset_name: str,
    dataset_version: str,
    dataset_schema: DatasetSchema,
    dataset: List[Tuple[str, Dict[str, Any]]],
):
    """Test if the basic persistor saves the correct data and shuffles it into different partitions

    Ensure we read the right file locations, the right amount of bytes,
    and the ordering is correct.
    """
    # in order to assert that we are shuffling we are going to sub out the
    # persist partition function and get average distance on global index
    # if it is != 2 (ie: samples are adjacent in partitions) then shuffling has occured
    def mock_persist_wicker_partition(
        self,
        spark_partition_iter: Iterable[Tuple[str, ParsedExample]],
        schema: schema.DatasetSchema,
        s3_storage: FakeS3DataStorage,
        s3_path_factory: S3PathFactory,
        target_max_column_file_numrows: int = 50,
    ) -> Iterable[Tuple[str, PointerParsedExample]]:
        # set up the global sum and counter for calcing mean
        global_sum = 0
        global_counter = 0
        # we still have to do all of the regular logic to test writing
        column_bytes_file_writers: Dict[str, ColumnBytesFileWriter] = {}
        heavy_pointer_columns = schema.get_pointer_columns()
        metadata_columns = schema.get_non_pointer_columns()
        previous_value, previous_parition = None, None

        for partition, example in spark_partition_iter:
            # if the previous value is unset or the parition has changed
            if not previous_value or previous_parition != partition:
                previous_value = example["global_index"]
                previous_parition = partition
            # if we can calculate the distance because we are on same parition
            # and the previous value is not None
            else:
                current_diff = abs(example["global_index"] - previous_value)
                previous_value = example["global_index"]
                previous_parition = partition
                global_sum += current_diff
                global_counter += 1
            # Create ColumnBytesFileWriter lazily as required, for each partition
            if partition not in column_bytes_file_writers:
                column_bytes_file_writers[partition] = ColumnBytesFileWriter(
                    s3_storage,
                    s3_path_factory,
                    target_file_rowgroup_size=target_max_column_file_numrows,
                )

            # Write to ColumnBytesFileWriter and return only metadata + heavy-pointers
            parquet_metadata: Dict[str, Any] = {col: example[col] for col in metadata_columns}
            for col in heavy_pointer_columns:
                loc = column_bytes_file_writers[partition].add(col, example[col])
                parquet_metadata[col] = loc.to_bytes()
            yield partition, parquet_metadata

        # Flush all writers when finished
        for partition in column_bytes_file_writers:
            column_bytes_file_writers[partition].close()
        # assert that we are not at mean 2 and that we have shuffled successfully
        mean = global_sum / global_counter
        assert mean != 2.0

    # create the mock basic persistor
    with patch("wicker.core.persistance.AbstractDataPersistor.persist_wicker_partition", mock_persist_wicker_partition):
        mock_basic_persistor_obj, tempdir = mock_basic_persistor
        # persist and shuffle the dataset
        mock_basic_persistor_obj.persist_wicker_dataset(dataset_name, dataset_version, dataset_schema, dataset, False)
        # assert the dataset is correctly written
        assert_written_correctness(tempdir)
