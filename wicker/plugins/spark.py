"""Spark plugin for writing a dataset with Spark only (no external metadata database required)

This plugin does an expensive global sorting step using Spark, which could be prohibitive
for large datasets.
"""


from __future__ import annotations

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
from wicker.schema import dataparsing, serialization

SPARK_PARTITION_SIZE = 256

PrimaryKeyTuple = Tuple[Any, ...]
UnparsedExample = Dict[str, Any]
ParsedExample = Dict[str, Any]
PointerParsedExample = Dict[str, Any]


class SparkPersistor(AbstractDataPersistor):
    def __init__(
        self,
        s3_storage: S3DataStorage = S3DataStorage(),
        s3_path_factory: S3PathFactory = S3PathFactory(),
        current_schema: Optional[schema.DatasetSchema] = None,
        current_dataset_name: Optional[str] = None,
        current_dataset_version: Optional[str] = None,
        current_rdd: Optional[pyspark.rdd.RDD[Tuple[str, UnparsedExample]]] = None,
    ) -> None:
        """
        Init a SparkPersistor

        :param s3_storage: The storage abstraction for S3
        :type s3_storage: S3DataStore
        :param s3_path_factory: The path factory for generating s3 paths
                                based on dataset name and version
        :type s3_path_factory: S3PathFactory
        """
        super().__init__(s3_storage, s3_path_factory, current_schema, current_dataset_name, current_dataset_version)
        self._current_rdd = current_rdd

    def persist_wicker_dataset(
        self,
        dataset_name: str,
        dataset_version: str,
        dataset_schema: schema.DatasetSchema,
        dataset: pyspark.rdd.RDD[Tuple[str, UnparsedExample]],
    ) -> Dict[str, int]:
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
        self._current_schema = dataset_schema
        self._current_dataset_name = dataset_name
        self._current_dataset_version = dataset_version
        self._current_rdd = dataset

        return self.persist_current_wicker_dataset()

    def persist_current_wicker_dataset(
        self,
    ) -> Dict[str, int]:
        """
        Persist the current rdd dataset defined by name, version, schema, and data.
        """
        self.__validate_current_dataset_variables()
        schema_path = self.s3_path_factory.get_dataset_schema_path(
            DatasetID(name=self._current_dataset_name, version=self._current_dataset_version)
        )
        self.s3_storage.put_object_s3(serialization.dumps(self._current_schema).encode("utf-8"), schema_path)

        # parse the rows and ensure validation passes, ie: rows actual data matches expected types
        # ignore type since this is already validated above
        rdd0 = self._current_rdd.mapValues(self._parse_row)  # type: ignore
        '''
        # Make sure to cache the RDD to ease future computations, since it seems that sortBy and zipWithIndex
        # trigger actions and we want to avoid recomputing the source RDD at all costs
        rdd0 = rdd0.cache()
        dataset_size = rdd0.count()

        rdd1 = rdd0.keyBy(self._get_row_keys)

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

        # Write data to Column Byte Files
        rdd3 = rdd2.values()
        rdd4 = rdd3.mapPartitions(
            lambda spark_iterator: self._persist_spark_partition_wicker(
                spark_iterator,
                # TODO(jchia): Magic number, we should derive this based on row size
                target_max_column_file_numrows=50,
            )
        )

        # combine the rdd by the keys in the pyarrow table
        rdd5 = rdd4.combineByKey(
            createCombiner=lambda data: pa.Table.from_pydict(
                {col: [data[col]] for col in self._current_schema.get_all_column_names()}
            ),
            mergeValue=lambda tbl, data: pa.Table.from_batches(
                [
                    *tbl.to_batches(),  # type: ignore
                    *pa.Table.from_pydict(
                        {col: [data[col]] for col in self._current_schema.get_all_column_names()}
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
        # sort the indices of the primary keys in ascending order
        rdd6 = rdd5.mapValues(
            lambda pa_tbl: pc.take(
                pa_tbl,
                pc.sort_indices(
                    pa_tbl,
                    sort_keys=[(pk, "ascending") for pk in self._current_schema.primary_keys],
                ),
            )
        )
        # save the parition table to s3
        rdd7 = rdd6.map(self._save_partition_tbl)
        written = rdd7.collect()

        return {partition: size for partition, size in written}'''

    def _persist_spark_partition_wicker(
        self,
        spark_partition_iter: Iterable[Tuple[str, ParsedExample]],
        target_max_column_file_numrows: int = 50,
    ) -> Iterable[Tuple[str, PointerParsedExample]]:
        """Persists a Spark partition of examples with parsed bytes into S3Storage as ColumnBytesFiles,
        returning a new Spark partition of examples with heavy-pointers and metadata only.
        :param spark_partition_iter: Spark partition of `(partition_str, example)`, where `example` is a dictionary of
            parsed bytes that needs to be uploaded to S3
        :param target_max_column_file_numrows: Maximum number of rows in column files. Defaults to 50.
        :return: a Generator of `(partition_str, example)`, where `example` is a dictionary with heavy-pointers
            that point to ColumnBytesFiles in S3 in place of the parsed bytes
        """
        column_bytes_file_writers: Dict[str, ColumnBytesFileWriter] = {}
        heavy_pointer_columns = self._current_schema.get_pointer_columns()
        metadata_columns = self._current_schema.get_non_pointer_columns()

        for partition, example in spark_partition_iter:
            # Create ColumnBytesFileWriter lazily as required, for each partition
            if partition not in column_bytes_file_writers:
                column_bytes_file_writers[partition] = ColumnBytesFileWriter(
                    self.s3_storage,
                    self.s3_path_factory,
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

    def _parse_row(
        self,
        data_row: UnparsedExample,
    ) -> ParsedExample:
        """
        Parse a row to test for validation errors.

        :param data_row: Data row to be parsed
        :type data_row: UnparsedExample
        :return: parsed row containing the correct types associated with schema
        :rtype: ParsedExample
        """
        return dataparsing.parse_example(data_row, self._current_schema)

    def _get_row_keys(
        self,
        partition_data_tup: Tuple[str, ParsedExample],
    ) -> PrimaryKeyTuple:
        """
        Get the keys of a row based on the parition tuple and the data schema.

        :param partition_data_tup: Tuple of partition id and ParsedExample row
        :type partition_data_tup: Tuple[str, ParsedExample]
        :return: Tuple of primary key values from parsed row and schema
        :rtype: PrimaryKeyTuple
        """
        partition, data = partition_data_tup
        return (partition,) + tuple(data[pk] for pk in self._current_schema.primary_keys)

    def _save_partition_tbl(
        self,
        partition_table_tuple: Tuple[str, pa.Table],
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
            self._current_dataset_name,
            self._current_dataset_version,
            {partition: pa_tbl},
            s3_storage=self.s3_storage,
            s3_path_factory=self.s3_path_factory,
        )
        return (partition, pa_tbl.num_rows)

    def __validate_current_dataset_variables(self):
        cur_dataset_type_tuples = (
            ("_current_dataset_name", str),
            ("_current_dataset_version", str),
            ("_current_schema", schema.DatasetSchema),
            ("_current_rdd", pyspark.rdd.RDD),
        )
        for cur_var_name, cur_var_expc_type in cur_dataset_type_tuples:
            if not isinstance(self.__getattribute__(cur_var_name), cur_var_expc_type):
                raise ValueError(f"{cur_var_name} must be of type {cur_var_expc_type}.")
