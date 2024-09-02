import os
import tempfile
import unittest
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pyarrow as pa  # type: ignore
import pyarrow.fs as pafs  # type: ignore
import pyarrow.parquet as papq  # type: ignore
import pytest
from pytest_mock import MockFixture

from wicker.core.column_files import ColumnBytesFileWriter
from wicker.core.datasets import S3Dataset, SFDataset
from wicker.core.definitions import DatasetID, DatasetPartition
from wicker.core.storage import S3PathFactory
from wicker.schema import schema, serialization
from wicker.schema.schema import (
    BoolField,
    DatasetSchema,
    FloatField,
    IntField,
    SchemaField,
    SfNumpyField,
    StringField,
)
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


@contextmanager
def cwd(path):
    """Changes the current working directory, and returns to the previous directory afterwards"""
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


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
                s3_path_factory=None,
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
                s3_path_factory=None,
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


@pytest.mark.parametrize(
    (
        "dataset_name",
        "version",
        "dataset_partition_name",
        "table_name",
        "connection_parameters",
        "columns_to_load",
        "optional_condition",
        "primary_keys",
    ),
    [
        ("test", "0.0.0", "partition", "test_table", {}, ["column0", "column1"], "column0 = test", ["key1"]),
    ],
)
def test_sf_dataset_constructor(
    mocker: MockFixture,
    dataset_name: str,
    version: str,
    dataset_partition_name: str,
    table_name: str,
    columns_to_load: List[str],
    primary_keys: List[str],
    connection_parameters: Dict[str, str],
    optional_condition: str,
) -> None:
    mocker.patch("wicker.core.datasets.SFDataset.connection")
    mocker.patch("wicker.core.datasets.SFDataset.schema")
    dataset = SFDataset(
        dataset_name=dataset_name,
        dataset_version=version,
        dataset_partition_name=dataset_partition_name,
        table_name=table_name,
        columns_to_load=columns_to_load,
        primary_keys=primary_keys,
        connection_parameters=connection_parameters,
        optional_condition=optional_condition,
    )
    assert dataset._dataset_id.name == dataset_name
    assert dataset._dataset_id.version == version
    assert dataset._partition.dataset_id.name == dataset_name
    assert dataset._partition.dataset_id.version == version
    assert dataset._partition.partition == dataset_partition_name
    assert dataset._table_name == table_name
    assert dataset._columns_to_load == columns_to_load
    assert dataset._primary_keys == primary_keys
    assert dataset._connection_parameters == connection_parameters
    assert dataset._optional_condition == optional_condition
    assert dataset._dataset_definition.dataset_id.name == dataset_name
    assert dataset._dataset_definition.dataset_id.version == version


@pytest.fixture
def sf_dataset(mocker: MockFixture) -> SFDataset:
    return SFDataset("test", "0.0.0", "", "", {}, schema=mocker.MagicMock())


@pytest.mark.parametrize(
    ("connection", "connection_parameter", "expectation"),
    [
        (None, {}, True),
        (1, {}, 1),
        (1, None, 1),
    ],
)
def test_connection(
    mocker: MockFixture, sf_dataset: SFDataset, connection: Any, connection_parameter: Dict[str, str], expectation: Any
) -> None:
    mocker.patch("snowflake.connector.connect", return_value=expectation)
    sf_dataset._connection = connection
    sf_dataset._connection_parameters = connection_parameter
    assert sf_dataset.connection == expectation


@pytest.mark.parametrize(
    ("schema", "schema_fields", "primary_keys", "expectation"),
    [
        (DatasetSchema([], [], True), [], [], DatasetSchema([], [], True)),
        (None, [StringField("key1")], ["key1"], DatasetSchema([StringField("key1")], ["key1"])),
    ],
)
def test_schema(
    mocker: MockFixture,
    sf_dataset: SFDataset,
    schema: Any,
    schema_fields: List[SchemaField],
    primary_keys: List[str],
    expectation: DatasetSchema,
) -> None:
    mocker.patch.object(sf_dataset, "_get_schema_from_database")
    mocker.patch.object(sf_dataset, "_get_schema_fields", return_value=schema_fields)
    mocker.patch.object(sf_dataset, "_primary_keys", new=primary_keys)
    sf_dataset._schema = schema
    ret = sf_dataset.schema()
    assert ret == expectation


@pytest.mark.parametrize(
    ("input_table", "table", "schema", "expectation"),
    [
        (
            pa.Table.from_arrays([[1, 2], [3, 4]], names=["col1", "col2"]),
            None,
            None,
            pa.Table.from_arrays([[1, 2], [3, 4]], names=["col1", "col2"]),
        ),
        (
            None,
            pa.Table.from_arrays([[1, 2], [3, 4]], names=["col1", "col2"]),
            DatasetSchema([StringField("col1"), StringField("col2")], primary_keys=["col1"]),
            pa.Table.from_arrays([[1, 2], [3, 4]], names=["col1", "col2"]),
        ),
        (
            None,
            pa.Table.from_arrays([[1, 2], [3, 4], ["[1]", "[2]"]], names=["col1", "col2", "sf1"]),
            DatasetSchema(
                [StringField("col1"), StringField("col2"), SfNumpyField("sf1", (1, -1), "float")], primary_keys=["col1"]
            ),
            pa.Table.from_arrays([[1, 2], [3, 4], ["[1]".encode(), "[2]".encode()]], names=["col1", "col2", "sf1"]),
        ),
    ],
)
def test_arrow_table(
    mocker: MockFixture,
    sf_dataset: SFDataset,
    input_table: Optional[pa.Table],
    table: Optional[pa.Table],
    schema: DatasetSchema,
    expectation: pa.Table,
) -> None:
    mocker.patch.object(sf_dataset, "_arrow_table", new=input_table)
    mocker.patch.object(sf_dataset, "_get_data")
    mocker.patch.object(sf_dataset, "_get_lower_case_columns", return_value=table)
    mocker.patch.object(sf_dataset, "schema", return_value=schema)
    ret = sf_dataset.arrow_table()
    assert ret == expectation


@pytest.mark.parametrize(
    ("table", "expectation"),
    [
        (pa.Table.from_arrays([[1, 2]], names=["col1"]), 2),
        (pa.Table.from_arrays([[1, 2, 3, 4]], names=["col1"]), 4),
    ],
)
def test_len(mocker: MockFixture, sf_dataset: SFDataset, table: pa.Table, expectation: int) -> None:
    mocker.patch.object(sf_dataset, "arrow_table", return_value=table)
    assert len(sf_dataset) == expectation


@pytest.mark.parametrize(
    ("table", "columns_to_load", "schema", "expectations"),
    [
        (
            pa.Table.from_arrays([[1, 2]], names=["col1"]),
            ["col1"],
            DatasetSchema([IntField("col1")], ["col1"]),
            [{"col1": 1}, {"col1": 2}],
        ),
        (
            pa.Table.from_arrays([[1, 2], ["[1]", "[2]"]], names=["col1", "sf1"]),
            ["col1"],
            DatasetSchema([IntField("col1"), SfNumpyField("sf1", (1, -1), "int")], ["col1"]),
            [{"col1": 1}, {"col1": 2}],
        ),
        (
            pa.Table.from_arrays([[1, 2], ["[1]", "[2]"]], names=["col1", "sf1"]),
            ["col1", "sf1"],
            DatasetSchema([IntField("col1"), SfNumpyField("sf1", (1, -1), "int")], ["col1"]),
            [{"col1": 1, "sf1": np.array([1])}, {"col1": 2, "sf1": np.array([2])}],
        ),
    ],
)
def test_getitem(
    mocker: MockFixture,
    sf_dataset: SFDataset,
    table: pa.Table,
    columns_to_load: List[str],
    schema: DatasetSchema,
    expectations: List[Dict[str, Any]],
) -> None:
    mocker.patch.object(sf_dataset, "arrow_table", return_value=table)
    mocker.patch.object(sf_dataset, "schema", return_value=schema)
    sf_dataset._columns_to_load = columns_to_load
    for idx, expectation in enumerate(expectations):
        assert sf_dataset[idx] == expectation


@pytest.mark.parametrize(
    ("table", "columns_to_load", "expectation"),
    [
        (
            pa.Table.from_arrays([["COL1", "COL2"], ["str", "int"]], names=["name", "type"]),
            None,
            pa.Table.from_arrays(
                [["COL1", "COL2"], ["str", "int"], ["col1", "col2"]], names=["name", "type", "lowercase_name"]
            ),
        ),
        (
            pa.Table.from_arrays([["COL1", "COL2"], ["str", "int"]], names=["name", "type"]),
            ["col1"],
            pa.Table.from_arrays([["COL1"], ["str"], ["col1"]], names=["name", "type", "lowercase_name"]),
        ),
    ],
)
def test_get_schema_from_database(
    mocker: MockFixture,
    sf_dataset: SFDataset,
    table: pa.Table,
    columns_to_load: Optional[List[str]],
    expectation: pa.Table,
) -> None:
    base_conn_mock = mocker.MagicMock()
    conn_mock = mocker.MagicMock()
    cur_mock = mocker.MagicMock()
    mocker.patch.object(cur_mock, "fetch_arrow_all", return_value=table)
    mocker.patch.object(conn_mock, "__enter__", return_value=cur_mock)
    mocker.patch.object(base_conn_mock, "cursor", return_value=conn_mock)
    sf_dataset._connection = base_conn_mock
    sf_dataset._columns_to_load = columns_to_load
    assert sf_dataset._get_schema_from_database() == expectation


@pytest.mark.parametrize(
    ("table", "expectation"),
    [
        (
            pa.Table.from_arrays(
                [["col1", "col2", "col3", "col4"], ["VARCHAR", "NUMBER(10,2)", "NUMBER(10,0)", "VARIANT"]],
                names=["lowercase_name", "type"],
            ),
            [StringField("col1"), FloatField("col2"), IntField("col3"), SfNumpyField("col4", (1, -1), "float32")],
        ),
    ],
)
def test_get_schema_fields(sf_dataset: SFDataset, table: pa.Table, expectation: List[SchemaField]) -> None:
    assert sf_dataset._get_schema_fields(table) == expectation


@pytest.mark.parametrize(
    ("type", "expectation"),
    [
        ("varchar", StringField),
        ("boolean", BoolField),
        ("number(1,2)", FloatField),
        ("number(1,0)", IntField),
        ("variant", SfNumpyField),
    ],
)
def test_get_schema_type(sf_dataset: SFDataset, type: str, expectation: SchemaField) -> None:
    assert sf_dataset._get_schema_type(type) == expectation


@pytest.mark.parametrize(
    ("table", "expectation"),
    [
        (pa.Table.from_arrays([[1, 2]], names=["COL1"]), pa.Table.from_arrays([[1, 2]], names=["col1"])),
        (pa.Table.from_arrays([[1, 2]], names=["col1"]), pa.Table.from_arrays([[1, 2]], names=["col1"])),
    ],
)
def test_get_lower_case_columns(sf_dataset: SFDataset, table: pa.Table, expectation: pa.Table) -> None:
    assert sf_dataset._get_lower_case_columns(table) == expectation
