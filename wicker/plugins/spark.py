"""Spark plugin for writing a dataset with Spark only (no external metadata database required)

This plugin does an expensive global sorting step using Spark, which could be prohibitive
for large datasets.
"""


from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc

try:
    import pyspark
except ImportError:
    raise RuntimeError(
        "pyspark is not detected in current environment, install Wicker with extra arguments:"
        " `pip install wicker[spark]`"
    )

from operator import add

from wicker import schema
from wicker.core.column_files import ColumnBytesFileWriter
from wicker.core.definitions import DatasetID
from wicker.core.errors import WickerDatastoreException
from wicker.core.persistance import AbstractDataPersistor
from wicker.core.shuffle import save_index
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.schema import serialization

SPARK_PARTITION_SIZE = 256

PrimaryKeyTuple = Tuple[Any, ...]
UnparsedExample = Dict[str, Any]
ParsedExample = Dict[str, Any]
PointerParsedExample = Dict[str, Any]


# public facing api consistency function
def persist_wicker_dataset(
    dataset_name: str,
    dataset_version: str,
    dataset_schema: schema.DatasetSchema,
    rdd: pyspark.rdd.RDD[Tuple[str, UnparsedExample]],
    s3_storage: S3DataStorage = S3DataStorage(),
    s3_path_factory: S3PathFactory = S3PathFactory(),
) -> Optional[Dict[str, int]]:
    """
    Persist wicker dataset public facing api function, for api consistency.
    :param dataset_name: name of dataset persisted
    :type dataset_name: str
    :param dataset_version: version of dataset persisted
    :type dataset_version: str
    :param dataset_schema: schema of dataset to be persisted
    :type dataset_schema: DatasetSchema
    :param rdd: rdd of data to persist
    :type rdd: RDD
    :param s3_storage: s3 storage abstraction
    :type s3_storage: S3DataStorage
    :param s3_path_factory: s3 path abstraction
    :type s3_path_factory: S3PathFactory
    """
    return SparkPersistor(s3_storage, s3_path_factory).persist_wicker_dataset(
        dataset_name, dataset_version, dataset_schema, rdd
    )


class SparkPersistor(AbstractDataPersistor):
    def __init__(
        self,
        s3_storage: S3DataStorage = S3DataStorage(),
        s3_path_factory: S3PathFactory = S3PathFactory(),
        schema: Optional[schema.DatasetSchema] = None,
        dataset_name: Optional[str] = None,
        dataset_version: Optional[str] = None,
        rdd: Optional[pyspark.rdd.RDD[Tuple[str, UnparsedExample]]] = None,
    ) -> None:
        """
        Init a SparkPersistor

        :param s3_storage: The storage abstraction for S3
        :type s3_storage: S3DataStore
        :param s3_path_factory: The path factory for generating s3 paths
                                based on dataset name and version
        :type s3_path_factory: S3PathFactory
        :param schema: Schema of the data
        :type schema: Dataset schema or none
        :param dataset_name: Optionally start with name of dataset
        :type dataset_name: Str or none
        :param dataset_version: Optionally start with version of dataset
        :type dataset_version: Str or none
        :param rdd: Rdd containing the data
        :type rdd: RDD
        """
        super().__init__(s3_storage, s3_path_factory, schema, dataset_name, dataset_version)
        self._current_rdd = rdd

    def persist_wicker_dataset(
        self,
        dataset_name: str,
        dataset_version: str,
        dataset_schema: schema.DatasetSchema,
        dataset: pyspark.rdd.RDD[Tuple[str, UnparsedExample]],
    ) -> Optional[Dict[str, int]]:
        """Persists a Spark RDD as a Wicker dataset. RDD must be provided as an RDD of Tuples of (partition, UnparsedExample),
        where `partition` is a string representing the dataset partition (e.g. train/test/eval) for that given row,
        and `UnparsedExample` is a Python Dict[str, Any] of the (non-validated) data to be written into the dataset.

        This function will perform validations, do a global sort based on primary keys,
        serialize the examples into bytes, save the data in S3 and persist the written data
        into the configured Wicker S3 bucket.

        :param dataset_name: name of the dataset
        :param dataset_version: version of the dataset
        :param dataset_schema: schema of the dataset
        :param dataset: RDD of data to be persisted as a Wicker dataset
        :return: A dictionary of partition name to size
        """
        # Write schema to S3
        self._current_dataset_name = dataset_name
        self._current_dataset_version = dataset_version
        self._current_schema = dataset_schema
        self._current_rdd = dataset

        return self.persist_current_wicker_dataset()

    def persist_current_wicker_dataset(
        self,
    ) -> Optional[Dict[str, int]]:
        """
        Persist the current rdd dataset defined by name, version, schema, and data.
        """
        # check if variables have been set ie: not None
        if (
            not isinstance(self._current_dataset_name, str)
            or not isinstance(self._current_dataset_version, str)
            or not isinstance(self._current_schema, schema.DatasetSchema)
            or not isinstance(self._current_rdd, pyspark.rdd.RDD)
        ):
            logging.warning("Current dataset variables not all set, set all to proper not None values")
            return None

        # define locally for passing to spark rdd ops, breaks if relying on self
        # since it passes to spark engine and we lose self context
        current_schema: schema.DatasetSchema = self._current_schema
        s3_storage = self.s3_storage
        s3_path_factory = self.s3_path_factory
        current_dataset_name = self._current_dataset_name
        current_dataset_version = self._current_dataset_version
        current_rdd = self._current_rdd
        parse_row = self.parse_row
        get_row_keys = self.get_row_keys
        persist_spark_partition_wicker = self.persist_spark_partition_wicker
        save_partition_tbl = self.save_partition_tbl

        # put the schema up on to s3
        schema_path = s3_path_factory.get_dataset_schema_path(
            DatasetID(name=current_dataset_name, version=current_dataset_version)
        )
        s3_storage.put_object_s3(serialization.dumps(current_schema).encode("utf-8"), schema_path)

        # parse the rows and ensure validation passes, ie: rows actual data matches expected types
        rdd0 = current_rdd.mapValues(lambda row: parse_row(row, current_schema))  # type: ignore

        # Make sure to cache the RDD to ease future computations, since it seems that sortBy and zipWithIndex
        # trigger actions and we want to avoid recomputing the source RDD at all costs
        rdd0 = rdd0.cache()
        dataset_size = rdd0.count()

        rdd1 = rdd0.keyBy(lambda row: get_row_keys(row, current_schema))

        # Sort RDD by keys
        rdd2: pyspark.rdd.RDD[Tuple[Tuple[Any, ...], Tuple[str, ParsedExample]]] = rdd1.sortByKey(
            # TODO(jchia): Magic number, we should derive this based on row size
            numPartitions=dataset_size // SPARK_PARTITION_SIZE,
            ascending=True,
        )

        def set_partition(iterator: Iterable[PrimaryKeyTuple]) -> Iterable[int]:
            key_set = set(iterator)
            yield len(key_set)

        # the number of unique keys in rdd partitions
        # this is softer check than collecting all the keys in all partitions to check uniqueness9
        rdd_key_count: int = rdd2.map(lambda x: x[0]).mapPartitions(set_partition).reduce(add)
        num_unique_keys = rdd_key_count
        if dataset_size != num_unique_keys:
            raise WickerDatastoreException(
                f"""Error: dataset examples do not have unique primary key tuples.
                Dataset has has {dataset_size} examples but {num_unique_keys} unique primary keys"""
            )

        # persist the spark partition to S3Storage
        rdd3 = rdd2.values()
        rdd4 = rdd3.mapPartitions(
            lambda spark_iterator: persist_spark_partition_wicker(
                spark_iterator,
                current_schema,
                s3_storage,
                s3_path_factory,
                # TODO(jchia): Magic number, we should derive this based on row size
                target_max_column_file_numrows=50,
            )
        )

        # combine the rdd by the keys in the pyarrow table
        rdd5 = rdd4.combineByKey(
            createCombiner=lambda data: pa.Table.from_pydict(
                {col: [data[col]] for col in current_schema.get_all_column_names()}
            ),
            mergeValue=lambda tbl, data: pa.Table.from_batches(
                [
                    *tbl.to_batches(),  # type: ignore
                    *pa.Table.from_pydict(
                        {col: [data[col]] for col in current_schema.get_all_column_names()}
                    ).to_batches(),
                ]
            ),
            mergeCombiners=lambda tbl1, tbl2: pa.Table.from_batches(
                [
                    *tbl1.to_batches(),  # type: ignore
                    *tbl2.to_batches(),  # type: ignore
                ]
            ),
        )
        # create the partition tables
        rdd6 = rdd5.mapValues(
            lambda pa_tbl: pc.take(
                pa_tbl,
                pc.sort_indices(
                    pa_tbl,
                    sort_keys=[(pk, "ascending") for pk in current_schema.primary_keys],
                ),
            )
        )
        # save the parition table to s3
        rdd7 = rdd6.map(
            lambda partition_table: save_partition_tbl(
                partition_table, current_dataset_name, current_dataset_version, s3_storage, s3_path_factory
            )
        )
        written = rdd7.collect()

        return {partition: size for partition, size in written}

    @staticmethod
    def get_row_keys(partition_data_tup: Tuple[str, ParsedExample], schema: schema.DatasetSchema) -> PrimaryKeyTuple:
        """
        Get the keys of a row based on the parition tuple and the data schema.

        :param partition_data_tup: Tuple of partition id and ParsedExample row
        :type partition_data_tup: Tuple[str, ParsedExample]
        :return: Tuple of primary key values from parsed row and schema
        :rtype: PrimaryKeyTuple
        """
        partition, data = partition_data_tup
        return (partition,) + tuple(data[pk] for pk in schema.primary_keys)

    # Write data to Column Byte Files
    @staticmethod
    def persist_spark_partition_wicker(
        spark_partition_iter: Iterable[Tuple[str, ParsedExample]],
        schema: schema.DatasetSchema,
        s3_storage: S3DataStorage,
        s3_path_factory: S3PathFactory,
        target_max_column_file_numrows: int = 50,
    ) -> Iterable[Tuple[str, PointerParsedExample]]:
        """Persists a Spark partition of examples with parsed bytes into S3Storage as ColumnBytesFiles,
        returning a new Spark partition of examples with heavy-pointers and metadata only.
        :param spark_partition_iter: Spark partition of `(partition_str, example)`, where `example`
            is a dictionary of parsed bytes that needs to be uploaded to S3
        :param target_max_column_file_numrows: Maximum number of rows in column files. Defaults to 50.
        :return: a Generator of `(partition_str, example)`, where `example` is a dictionary with heavy-pointers
            that point to ColumnBytesFiles in S3 in place of the parsed bytes
        """
        column_bytes_file_writers: Dict[str, ColumnBytesFileWriter] = {}
        heavy_pointer_columns = schema.get_pointer_columns()
        metadata_columns = schema.get_non_pointer_columns()

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

    # sort the indices of the primary keys in ascending order
    @staticmethod
    def save_partition_tbl(
        partition_table_tuple: Tuple[str, pa.Table],
        current_dataset_name: str,
        current_dataset_version: str,
        s3_storage: S3DataStorage,
        s3_path_factory: S3PathFactory,
    ) -> Tuple[str, int]:
        """
        Save a partition table to s3 under the dataset name and version.

        :param partition_table_tuple: Tuple of partition id and pyarrow table to save
        :type partition_table_tuple: Tuple[str, pyarrow.Table]
        :return: A tuple containing the paritiion id and the num of saved rows
        :rtype: Tuple[str, int]
        """
        partition, pa_tbl = partition_table_tuple
        save_index(
            current_dataset_name,
            current_dataset_version,
            {partition: pa_tbl},
            s3_storage=s3_storage,
            s3_path_factory=s3_path_factory,
        )
        return (partition, pa_tbl.num_rows)
