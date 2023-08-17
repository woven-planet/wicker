import copy
import os
import random
import tempfile
import unittest
import uuid

import pyarrow.parquet as papq
from pyspark.sql import SparkSession

from wicker import schema
from wicker.core.config import get_config
from wicker.core.errors import WickerDatastoreException
from wicker.plugins.spark import persist_wicker_dataset
from wicker.testing.storage import FakeS3DataStorage

DATASET_NAME = "dataset"
DATASET_VERSION = "0.0.1"
TEST_ROWS_NUM = 10000
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
            "foo": random.randint(0, TEST_ROWS_NUM),
            "bar": str(uuid.uuid4()),
            "bytescol": b"0",
        },
    )
    for i in range(TEST_ROWS_NUM)
]
# Examples with a duplicated key
EXAMPLES_DUPES = copy.deepcopy(EXAMPLES)
EXAMPLES_DUPES[5000] = EXAMPLES_DUPES[0]


class LocalWritingTestCase(unittest.TestCase):
    def assert_written_correctness(self, tmpdir: str, row_num: int = TEST_ROWS_NUM) -> None:
        """Asserts that all files are written as expected by the L5MLDatastore"""
        # Check that files are correctly written locally by Spark/Parquet with a _SUCCESS marker file
        prefix = get_config().aws_s3_config.s3_datasets_path.replace("s3://", "")
        self.assertIn(DATASET_NAME, os.listdir(os.path.join(tmpdir, prefix)))
        self.assertIn(DATASET_VERSION, os.listdir(os.path.join(tmpdir, prefix, DATASET_NAME)))
        for partition in ["train", "test"]:
            columns_path = os.path.join(tmpdir, prefix, "__COLUMN_CONCATENATED_FILES__")
            all_read_bytes = b""
            for filename in os.listdir(columns_path):
                concatenated_bytes_filepath = os.path.join(columns_path, filename)
                with open(concatenated_bytes_filepath, "rb") as bytescol_file:
                    all_read_bytes += bytescol_file.read()
            self.assertEqual(all_read_bytes, b"0" * row_num)

            # Load parquet file and assert ordering of primary_key
            self.assertIn(
                f"{partition}.parquet", os.listdir(os.path.join(tmpdir, prefix, DATASET_NAME, DATASET_VERSION))
            )
            tbl = papq.read_table(os.path.join(tmpdir, prefix, DATASET_NAME, DATASET_VERSION, f"{partition}.parquet"))
            foobar = [(barval.as_py(), fooval.as_py()) for fooval, barval in zip(tbl["foo"], tbl["bar"])]
            self.assertEqual(foobar, sorted(foobar))

    def test_simple_schema_local_writing(self) -> None:
        for local_reduction in (True, False):
            for sort in (True, False):
                with tempfile.TemporaryDirectory() as tmpdir:
                    fake_storage = FakeS3DataStorage(tmpdir=tmpdir)
                    spark_session = SparkSession.builder.appName("test").master("local[*]")
                    spark = spark_session.getOrCreate()
                    sc = spark.sparkContext
                    rdd = sc.parallelize(copy.deepcopy(EXAMPLES), 100)
                    persist_wicker_dataset(
                        DATASET_NAME,
                        DATASET_VERSION,
                        SCHEMA,
                        rdd,
                        fake_storage,
                        local_reduction=local_reduction,
                        sort=sort,
                    )
                    self.assert_written_correctness(tmpdir)

    def test_dupe_primary_keys_raises_exception(self) -> None:
        for local_reduction in (True, False):
            for sort in (True, False):
                with self.assertRaises(WickerDatastoreException) as e:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        fake_storage = FakeS3DataStorage(tmpdir=tmpdir)
                        spark_session = SparkSession.builder.appName("test").master("local[*]")
                        spark = spark_session.getOrCreate()
                        sc = spark.sparkContext
                        rdd = sc.parallelize(copy.deepcopy(EXAMPLES_DUPES), 100)
                        persist_wicker_dataset(
                            DATASET_NAME,
                            DATASET_VERSION,
                            SCHEMA,
                            rdd,
                            fake_storage,
                            local_reduction=local_reduction,
                            sort=sort,
                        )

                    self.assertIn(
                        "Error: dataset examples do not have unique primary key tuples",
                        str(e.exception),
                    )

    def test_simple_schema_local_writing_4_row_dataset(self) -> None:
        for local_reduction in (True, False):
            for sort in (True, False):
                with tempfile.TemporaryDirectory() as tmpdir:
                    small_row_cnt = 4
                    fake_storage = FakeS3DataStorage(tmpdir=tmpdir)
                    spark_session = SparkSession.builder.appName("test").master("local[*]")
                    spark = spark_session.getOrCreate()
                    sc = spark.sparkContext
                    rdd = sc.parallelize(copy.deepcopy(EXAMPLES)[:small_row_cnt], 1)
                    persist_wicker_dataset(
                        DATASET_NAME,
                        DATASET_VERSION,
                        SCHEMA,
                        rdd,
                        fake_storage,
                        local_reduction=local_reduction,
                        sort=sort,
                    )
                    self.assert_written_correctness(tmpdir, small_row_cnt)
