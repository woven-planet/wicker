import copy
import os
import random
import uuid
from typing import Any, Dict, List, Tuple

import pyarrow.fs as pafs
import pyarrow.parquet as papq
import pytest

from wicker import schema
from wicker.core.datasets import S3Dataset
from wicker.core.persistance import BasicPersistor
from wicker.core.storage import S3PathFactory
from wicker.schema.schema import DatasetSchema
from wicker.testing.storage import LocalDataStorage

DATASET_NAME = "dataset"
DATASET_VERSION = "0.0.1"
SCHEMA = schema.DatasetSchema(
    primary_keys=["bar", "foo"],
    fields=[
        schema.IntField("foo"),
        schema.StringField("bar"),
        schema.BytesField("bytescol"),
    ],
)
EXAMPLES = [
    (
        "train" if i % 2 == 0 else "test",
        {
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
    storage = request.param.get("storage", LocalDataStorage(root_path=tmpdir))
    path_factory = request.param.get("path_factory", S3PathFactory(s3_root_path=os.path.join(tmpdir, "datasets")))
    return BasicPersistor(storage, path_factory), tmpdir


def assert_written_correctness(persistor: BasicPersistor, data: List[Tuple[str, Dict[str, Any]]], tmpdir: str) -> None:
    """Asserts that all files are written as expected by the L5MLDatastore"""
    # Check that files are correctly written locally by Spark/Parquet with a _SUCCESS marker file
    prefix = persistor.s3_path_factory.root_path
    assert DATASET_NAME in os.listdir(os.path.join(tmpdir, prefix))
    assert DATASET_VERSION in os.listdir(os.path.join(tmpdir, prefix, DATASET_NAME))
    for partition in ["train", "test"]:
        print(os.listdir(os.path.join(tmpdir, prefix)))
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
        foobar = [(barval.as_py(), fooval.as_py()) for fooval, barval in zip(tbl["foo"], tbl["bar"])]
        assert foobar == sorted(foobar)

        # Also load the data from the dataset and check it.
        ds = S3Dataset(
            DATASET_NAME,
            DATASET_VERSION,
            partition,
            local_cache_path_prefix=tmpdir,
            storage=persistor.s3_storage,
            s3_path_factory=persistor.s3_path_factory,
            pa_filesystem=pafs.LocalFileSystem(),
        )

        ds_values = [ds[idx] for idx in range(len(ds))]
        # Sort the expected values by the primary keys.
        expected_values = sorted([value for p, value in data if p == partition], key=lambda v: (v["bar"], v["foo"]))
        assert ds_values == expected_values


@pytest.mark.parametrize(
    "mock_basic_persistor, dataset_name, dataset_version, dataset_schema, dataset",
    [({}, DATASET_NAME, DATASET_VERSION, SCHEMA, copy.deepcopy(EXAMPLES_DUPES))],
    indirect=["mock_basic_persistor"],
)
def test_basic_persistor(
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
    # create the mock basic persistor
    mock_basic_persistor_obj, tempdir = mock_basic_persistor
    # persist the dataset
    mock_basic_persistor_obj.persist_wicker_dataset(dataset_name, dataset_version, dataset_schema, dataset)
    # assert the dataset is correctly written
    assert_written_correctness(persistor=mock_basic_persistor_obj, data=dataset, tmpdir=tempdir)
