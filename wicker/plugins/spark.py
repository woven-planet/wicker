"""Spark plugin for writing a dataset with Spark only (no external metadata database required)

This plugin does an expensive global sorting step using Spark, which could be prohibitive
for large datasets.
"""


from typing import Any, Dict, Iterator, Tuple

import pyarrow as pa

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


def _persist_spark_partition_wicker(
    spark_partition_iter: Iterator[Tuple[str, Dict[str, Any]]],
    dataset_schema: schema.DatasetSchema,
    target_max_column_file_numrows: int = 50,
    target_column_file_size: int = int(250 * 1e6),
    storage: S3DataStorage = S3DataStorage(),
    s3_path_factory: S3PathFactory = S3PathFactory(),
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Persists a Spark partition of examples with parsed bytes into S3Storage as ColumnBytesFiles,
    returning a new Spark partition of examples with heavy-pointers and metadata only.

    :param spark_partition_iter: Spark partition of `(partition_str, example)`, where `example` is a dictionary of
        parsed bytes that needs to be uploaded to S3
    :param dataset_schema: schema of dataset
    :param target_max_column_file_numrows: Maximum number of rows in column files. Defaults to 50.
    :param target_column_file_size: Target file size for colunmn files. Defaults to 250MB.
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
                storage,
                s3_path_factory,
                target_file_size=target_column_file_size,
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
    rdd: pyspark.rdd.RDD[Tuple[str, Dict[str, Any]]],
    sort_by_primary_keys: bool = True,
    target_max_column_file_numrows: int = 50,
    target_column_file_size: int = int(250 * 1e6),
) -> None:
    """Persists a Spark RDD as a Wicker dataset. RDD must be provided as an RDD of Tuples of (partition, example),
    where `partition` is a string representing the dataset partition (e.g. train/test/eval) for that given row,
    and `example` is a Python Dict[str, Any] of the (non-validated) data to be written into the dataset.

    This function will perform validations, do a global sort based on primary keys, serialize the examples into bytes,
    save the data in S3 and persist the written data into the configured Wicker S3 bucket.

    :param dataset_name: name of the dataset
    :param dataset_version: version of the dataset
    :param dataset_schema: schema of the dataset
    :param rdd: RDD of data to be persisted as a Wicker dataset
    :param sort_by_primary_keys: Whether or not to sort the dataset by primary key, or persist it in whatever
        order the user has passed in the RDD as, defaults to True
    :param target_max_column_file_numrows: max number of rows per ColumnByteFile, defaults to 50
    :param target_column_file_size: target size of each ColumnByteFile, defaults to 250MB
    """
    # Write schema to S3
    s3_storage = S3DataStorage()
    s3_path_factory = S3PathFactory()
    schema_path = s3_path_factory.get_dataset_schema_path(DatasetID(name=dataset_name, version=dataset_version))
    s3_storage.put_object_s3(serialization.dumps(dataset_schema).encode("utf-8"), schema_path)

    # Parse each row to throw errors early if they fail validation
    def parse_row(partition_data_tup: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        partition, data = partition_data_tup
        return partition, dataparsing.parse_example(data, dataset_schema)

    rdd = rdd.map(parse_row)

    # Sort RDD by dataset partition and primary keys if required
    def sort_row_key(partition_data_tup: Tuple[str, Dict[str, Any]]) -> str:
        partition, data = partition_data_tup
        return "-".join([partition, *[str(data[key]) for key in dataset_schema.primary_keys]])

    if sort_by_primary_keys:
        rdd = rdd.sortBy(sort_row_key)

    # Write heavy bytes to S3 and return just an RDD of the metadata
    rdd = rdd.mapPartitions(
        lambda spark_iterator: _persist_spark_partition_wicker(
            spark_iterator,
            dataset_schema,
            target_max_column_file_numrows=target_max_column_file_numrows,
            target_column_file_size=target_column_file_size,
        )
    )

    # For each dataset partition, persist the metadata as a Parquet file in S3
    def save_partition_tbl(partition_table_tuple: Tuple[str, pa.Table]) -> None:
        partition, pa_tbl = partition_table_tuple
        return save_index(dataset_name, dataset_version, {partition: pa_tbl})

    rdd.combineByKey(
        createCombiner=lambda data: pa.Table.from_pydict(
            {col: [data[col]] for col in dataset_schema.get_all_column_names()}
        ),
        mergeValue=lambda tbl, data: pa.Table.from_batches(
            [
                *tbl.to_batches(),
                *pa.Table.from_pydict({col: [data[col]] for col in dataset_schema.get_all_column_names()}).to_batches(),
            ]
        ),
        mergeCombiners=lambda tbl1, tbl2: pa.Table.from_batches([*tbl1.to_batches(), *tbl2.to_batches()]),
    ).map(save_partition_tbl).collect()
