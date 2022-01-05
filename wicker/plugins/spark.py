"""Spark plugin for writing a dataset with Spark only (no external metadata database required)

This plugin does an expensive global sorting step using Spark, which could be prohibitive
for large datasets.
"""


from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

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
    spark_partition_iter: Iterable[Tuple[str, Dict[str, Any]]],
    dataset_schema: schema.DatasetSchema,
    target_max_column_file_numrows: int = 50,
    s3_storage: S3DataStorage = S3DataStorage(),
    s3_path_factory: S3PathFactory = S3PathFactory(),
) -> Iterable[Tuple[str, Dict[str, Any]]]:
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
    rdd: pyspark.rdd.RDD[Tuple[str, Dict[str, Any]]],
    s3_storage: S3DataStorage = S3DataStorage(),
    s3_path_factory: S3PathFactory = S3PathFactory(),
) -> Dict[str, int]:
    """Persists a Spark RDD as a Wicker dataset. RDD must be provided as an RDD of Tuples of (partition, example),
    where `partition` is a string representing the dataset partition (e.g. train/test/eval) for that given row,
    and `example` is a Python Dict[str, Any] of the (non-validated) data to be written into the dataset.

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
    def parse_row(data: Dict[str, Any]) -> Dict[str, Any]:
        return dataparsing.parse_example(data, dataset_schema)

    rdd = rdd.mapValues(parse_row)

    # Make sure to cache the RDD to ease future computations, since it seems that sortBy and zipWithIndex
    # trigger actions and we want to avoid recomputing the source RDD at all costs
    rdd = rdd.cache()

    # Materialize partitioning index to RDD to properly do a `.zip` later, which assumes that
    # the two RDDs have the same number of partitions and the same number of elements in each partition
    # See: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.zip.html
    #
    # (spark_partition_idx, (partition, data))
    def zipWithPartitionIndex(
        partition_idx: int, it: Iterable[Tuple[str, Dict[str, Any]]]
    ) -> Iterable[Tuple[int, Tuple[str, Dict[str, Any]]]]:
        for tup in it:
            yield (partition_idx, tup)

    rdd0 = rdd.mapPartitionsWithIndex(zipWithPartitionIndex)

    def get_row_keys(partition_data_tup: Tuple[str, Dict[str, Any]]) -> Tuple[Any, ...]:
        partition, data = partition_data_tup
        return (partition,) + tuple(data[pk] for pk in dataset_schema.primary_keys)

    # Obtain an RDD of the argsort of the primary keys. Must have same partitioning as our source
    # RDD to enable us to zip them together later on.
    #
    # 1. (spark_partition_idx, (partition, data)) -> (spark_partition_idx, primary_key)
    # 2. (spark_partition_idx, primary_key) -> ((spark_partition_idx, primary_key), original_idx)
    # 3. Sort by primary_key
    # 4. ((spark_partition_idx, primary_key), original_idx) ->
    #       (((spark_partition_idx, primary_key), original_idx), global_sorted_idx)
    # 5. Sort by original_idx
    # 6. (((spark_partition_idx, primary_key), original_idx), global_sorted_idx) ->
    #       (spark_partition_idx, global_sorted_idx)
    # 7. Repartition by spark_partition_idx
    # 8. (spark_partition_idx, global_sorted_idx) -> global_sorted_idx
    global_sorted_idx_rdd = rdd0
    global_sorted_idx_rdd0 = global_sorted_idx_rdd.mapValues(get_row_keys)
    global_sorted_idx_rdd1 = global_sorted_idx_rdd0.zipWithIndex()
    global_sorted_idx_rdd2 = global_sorted_idx_rdd1.sortBy(lambda x: x[0][1])  # type: ignore
    global_sorted_idx_rdd3 = global_sorted_idx_rdd2.zipWithIndex()
    global_sorted_idx_rdd4 = global_sorted_idx_rdd3.sortBy(lambda x: x[0][1])  # type: ignore
    global_sorted_idx_rdd5 = global_sorted_idx_rdd4.map(lambda x: (x[0][0][0], x[1]))
    global_sorted_idx_rdd6 = global_sorted_idx_rdd5.partitionBy(rdd.getNumPartitions(), partitionFunc=lambda x: x)
    global_sorted_idx_rdd7 = global_sorted_idx_rdd6.values()

    # Zip our argsort RDD with our source RDD, and use the argsort indices to repartition + sort our data.
    # Note that this performs a distributed shuffle of our heavy bytes data to bring them into contiguous order.
    # We then do a mapPartitions to write each partition into Wicker
    #
    # 1. (spark_partition_idx, (partition, data)) -> ((spark_partition_idx, (partition, data)), global_sorted_idx)
    # 2. ((spark_partition_idx, (partition, data)), global_sorted_idx) -> (global_sorted_idx, (partition, data))
    # 3. Repartition and sort by global_sorted_idx to obtain partitions of contiguous, globally sorted examples
    # 4. (global_sorted_idx, (partition, data)) -> (partition, data)
    # 5. Write each partition of (partition, data) into Wicker
    rdd1 = rdd0.zip(global_sorted_idx_rdd7)
    rdd2 = rdd1.map(lambda x: (x[1], x[0][1]))
    rdd3 = rdd2.repartitionAndSortWithinPartitions(  # type: ignore
        # TODO(jchia): Magic number, we should derive this based on row size
        partitionFunc=lambda x: x // 256,
        keyfunc=lambda x: x,
    )
    rdd4 = rdd3.values()
    rdd5 = rdd4.mapPartitions(
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

    rdd6 = rdd5.combineByKey(
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
    rdd7 = rdd6.map(save_partition_tbl)
    written = rdd7.collect()

    return {partition: size for partition, size in written}
