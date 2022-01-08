"""Spark plugin for writing a dataset with Spark only (no external metadata database required)

This plugin does an expensive global sorting step using Spark, which could be prohibitive
for large datasets.
"""


from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import pyarrow as pa
import pyarrow.compute as pc

try:
    import pyspark
except ImportError:
    raise RuntimeError(
        "pyspark is not detected in current environment, install Wicker with extra arguments:"
        " `pip install wicker[spark]`"
    )

from wicker import schema
from wicker.core.column_files import ColumnBytesFileWriter
from wicker.core.definitions import DatasetID
from wicker.core.shuffle import save_index
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.schema import dataparsing, serialization

SPARK_PARTITION_SIZE = 256

PrimaryKeyTuple = Tuple[Any, ...]
UnparsedExample = Dict[str, Any]
ParsedExample = Dict[str, Any]
PointerParsedExample = Dict[str, Any]


def _persist_spark_partition_wicker(
    spark_partition_iter: Iterable[Tuple[str, ParsedExample]],
    dataset_schema: schema.DatasetSchema,
    target_max_column_file_numrows: int = 50,
    s3_storage: S3DataStorage = S3DataStorage(),
    s3_path_factory: S3PathFactory = S3PathFactory(),
) -> Iterable[Tuple[str, PointerParsedExample]]:
    """Persists a Spark partition of examples with parsed bytes into S3Storage as ColumnBytesFiles,
    returning a new Spark partition of examples with heavy-pointers and metadata only.
    :param spark_partition_iter: Spark partition of `(partition_str, example)`, where `example` is a dictionary of
        parsed bytes that needs to be uploaded to S3
    :param dataset_schema: schema of dataset
    :param target_max_column_file_numrows: Maximum number of rows in column files. Defaults to 50.
    :param storage: S3DataStorage to use. Defaults to S3DataStorage().
    :param s3_path_factory: S3PathFactory to use. Defaults to S3PathFactory().
    :return: a Generator of `(partition_str, example)`, where `example` is a dictionary with heavy-pointers
        that point to ColumnBytesFiles in S3 in place of the parsed bytes
    """
    column_bytes_file_writers: Dict[str, ColumnBytesFileWriter] = {}
    heavy_pointer_columns = dataset_schema.get_pointer_columns()
    metadata_columns = dataset_schema.get_non_pointer_columns()

    for partition, example in spark_partition_iter:
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


def persist_wicker_dataset(
    dataset_name: str,
    dataset_version: str,
    dataset_schema: schema.DatasetSchema,
    rdd: pyspark.rdd.RDD[Tuple[str, UnparsedExample]],
    s3_storage: S3DataStorage = S3DataStorage(),
    s3_path_factory: S3PathFactory = S3PathFactory(),
) -> Dict[str, int]:
    """Persists a Spark RDD as a Wicker dataset. RDD must be provided as an RDD of Tuples of (partition, UnparsedExample),
    where `partition` is a string representing the dataset partition (e.g. train/test/eval) for that given row,
    and `UnparsedExample` is a Python Dict[str, Any] of the (non-validated) data to be written into the dataset.

    This function will perform validations, do a global sort based on primary keys, serialize the examples into bytes,
    save the data in S3 and persist the written data into the configured Wicker S3 bucket.

    :param dataset_name: name of the dataset
    :param dataset_version: version of the dataset
    :param dataset_schema: schema of the dataset
    :param rdd: RDD of data to be persisted as a Wicker dataset
    :return: A dictionary of partition name to size
    """
    # Write schema to S3
    schema_path = s3_path_factory.get_dataset_schema_path(DatasetID(name=dataset_name, version=dataset_version))
    s3_storage.put_object_s3(serialization.dumps(dataset_schema).encode("utf-8"), schema_path)

    # Parse each row to throw errors early if they fail validation
    def parse_row(data: UnparsedExample) -> ParsedExample:
        return dataparsing.parse_example(data, dataset_schema)

    rdd0 = rdd.mapValues(parse_row)

    # Make sure to cache the RDD to ease future computations, since it seems that sortBy and zipWithIndex
    # trigger actions and we want to avoid recomputing the source RDD at all costs
    rdd0 = rdd0.cache()
    dataset_size = rdd0.count()

    # Key each row with the partition + primary_keys
    # (partition, data) -> (primary_key_tup, (partition, data))
    def get_row_keys(partition_data_tup: Tuple[str, ParsedExample]) -> PrimaryKeyTuple:
        partition, data = partition_data_tup
        return (partition,) + tuple(data[pk] for pk in dataset_schema.primary_keys)

    rdd1 = rdd0.keyBy(get_row_keys)

    # Sort RDD by keys
    rdd2: pyspark.rdd.RDD[Tuple[Tuple[Any, ...], Tuple[str, ParsedExample]]] = rdd1.sortByKey(
        # TODO(jchia): Magic number, we should derive this based on row size
        numPartitions=dataset_size // SPARK_PARTITION_SIZE,
        ascending=True,
    )

    # Write data to Column Byte Files
    rdd3 = rdd2.values()
    rdd4 = rdd3.mapPartitions(
        lambda spark_iterator: _persist_spark_partition_wicker(
            spark_iterator,
            dataset_schema,
            # TODO(jchia): Magic number, we should derive this based on row size
            target_max_column_file_numrows=50,
            s3_storage=s3_storage,
            s3_path_factory=s3_path_factory,
        )
    )

    # For each dataset partition, persist the metadata as a Parquet file in S3
    def save_partition_tbl(partition_table_tuple: Tuple[str, pa.Table]) -> Tuple[str, int]:
        partition, pa_tbl = partition_table_tuple
        save_index(
            dataset_name, dataset_version, {partition: pa_tbl}, s3_storage=s3_storage, s3_path_factory=s3_path_factory
        )
        return (partition, pa_tbl.num_rows)

    rdd5 = rdd4.combineByKey(
        createCombiner=lambda data: pa.Table.from_pydict(
            {col: [data[col]] for col in dataset_schema.get_all_column_names()}
        ),
        mergeValue=lambda tbl, data: pa.Table.from_batches(
            [
                *tbl.to_batches(),  # type: ignore
                *pa.Table.from_pydict({col: [data[col]] for col in dataset_schema.get_all_column_names()}).to_batches(),
            ]
        ),
        mergeCombiners=lambda tbl1, tbl2: pa.Table.from_batches(
            [
                *tbl1.to_batches(),  # type: ignore
                *tbl2.to_batches(),  # type: ignore
            ]
        ),
    )
    rdd6 = rdd5.mapValues(
        lambda pa_tbl: pc.take(
            pa_tbl,
            pc.sort_indices(
                pa_tbl,
                sort_keys=[(pk, "ascending") for pk in dataset_schema.primary_keys],
            ),
        )
    )
    rdd7 = rdd6.map(save_partition_tbl)
    written = rdd7.collect()

    return {partition: size for partition, size in written}
