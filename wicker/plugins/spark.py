"""Spark plugin for writing a dataset with Spark only (no external metadata database required)

This plugin does an expensive global sorting step using Spark, which could be prohibitive
for large datasets.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

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

from wicker import schema as schema_module
from wicker.core.definitions import DatasetID
from wicker.core.errors import WickerDatastoreException
from wicker.core.persistance import AbstractDataPersistor
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.schema import serialization

SPARK_PARTITION_SIZE = 256
MAX_COL_FILE_NUMROW = 50  # TODO(isaak-willett): Magic number, we should derive this based on row size

PrimaryKeyTuple = Tuple[Any, ...]
UnparsedExample = Dict[str, Any]
ParsedExample = Dict[str, Any]
PointerParsedExample = Dict[str, Any]


def persist_wicker_dataset(
    dataset_name: str,
    dataset_version: str,
    dataset_schema: schema_module.DatasetSchema,
    rdd: pyspark.rdd.RDD[Tuple[str, UnparsedExample]],
    s3_storage: S3DataStorage = S3DataStorage(),
    s3_path_factory: S3PathFactory = S3PathFactory(),
    local_reduction: bool = False,
    sort: bool = True,
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
    :param local_reduction: if true, reduce tables on main instance (no spark). This is useful if the spark reduction
        is not feasible for a large dataste.
    :type local_reduction: bool
    :param sort: if true, sort the resulting table by primary keys
    :type sort: bool
    """
    return SparkPersistor(s3_storage, s3_path_factory).persist_wicker_dataset(
        dataset_name,
        dataset_version,
        dataset_schema,
        rdd,
        local_reduction=local_reduction,
        sort=sort,
    )


class SparkPersistor(AbstractDataPersistor):
    def __init__(
        self,
        s3_storage: S3DataStorage = S3DataStorage(),
        s3_path_factory: S3PathFactory = S3PathFactory(),
    ) -> None:
        """
        Init a SparkPersistor

        :param s3_storage: The storage abstraction for S3
        :type s3_storage: S3DataStore
        :param s3_path_factory: The path factory for generating s3 paths
                                based on dataset name and version
        :type s3_path_factory: S3PathFactory
        """
        super().__init__(s3_storage, s3_path_factory)

    def persist_wicker_dataset(
        self,
        dataset_name: str,
        dataset_version: str,
        schema: schema_module.DatasetSchema,
        rdd: pyspark.rdd.RDD[Tuple[str, UnparsedExample]],
        local_reduction: bool = False,
        sort: bool = True,
    ) -> Optional[Dict[str, int]]:
        """
        Persist the current rdd dataset defined by name, version, schema, and data.
        """
        # check if variables have been set ie: not None
        if (
            not isinstance(dataset_name, str)
            or not isinstance(dataset_version, str)
            or not isinstance(schema, schema_module.DatasetSchema)
            or not isinstance(rdd, pyspark.rdd.RDD)
        ):
            raise ValueError("Current dataset variables not all set, set all to proper not None values")

        # define locally for passing to spark rdd ops, breaks if relying on self
        # since it passes to spark engine and we lose self context
        s3_storage = self.s3_storage
        s3_path_factory = self.s3_path_factory
        parse_row = self.parse_row
        get_row_keys = self.get_row_keys
        persist_wicker_partition = self.persist_wicker_partition
        save_partition_tbl = self.save_partition_tbl

        # put the schema up on to s3
        schema_path = s3_path_factory.get_dataset_schema_path(DatasetID(name=dataset_name, version=dataset_version))
        s3_storage.put_object_s3(serialization.dumps(schema).encode("utf-8"), schema_path)

        # parse the rows and ensure validation passes, ie: rows actual data matches expected types
        rdd0 = rdd.mapValues(lambda row: parse_row(row, schema))  # type: ignore

        # Make sure to cache the RDD to ease future computations, since it seems that sortBy and zipWithIndex
        # trigger actions and we want to avoid recomputing the source RDD at all costs
        rdd0 = rdd0.cache()
        dataset_size = rdd0.count()

        rdd1 = rdd0.keyBy(lambda row: get_row_keys(row, schema))

        # Sort RDD by keys
        rdd2: pyspark.rdd.RDD[Tuple[Tuple[Any, ...], Tuple[str, ParsedExample]]] = rdd1.sortByKey(
            # TODO(jchia): Magic number, we should derive this based on row size
            numPartitions=max(1, dataset_size // SPARK_PARTITION_SIZE),
            ascending=True,
        )

        def set_partition(iterator: Iterable[PrimaryKeyTuple]) -> Iterable[int]:
            key_set = set(iterator)
            yield len(key_set)

        # the number of unique keys in rdd partitions
        # this is softer check than collecting all the keys in all partitions to check uniqueness
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
            lambda spark_iterator: persist_wicker_partition(
                dataset_name,
                spark_iterator,
                schema,
                s3_storage,
                s3_path_factory,
                target_max_column_file_numrows=MAX_COL_FILE_NUMROW,
            )
        )

        if not local_reduction:
            # combine the rdd by the keys in the pyarrow table
            rdd5 = rdd4.combineByKey(
                createCombiner=lambda data: pa.Table.from_pydict(
                    {col: [data[col]] for col in schema.get_all_column_names()}
                ),
                mergeValue=lambda tbl, data: pa.Table.from_batches(
                    [
                        *tbl.to_batches(),  # type: ignore
                        *pa.Table.from_pydict({col: [data[col]] for col in schema.get_all_column_names()}).to_batches(),
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
            if sort:
                rdd6 = rdd5.mapValues(
                    lambda pa_tbl: pc.take(
                        pa_tbl,
                        pc.sort_indices(
                            pa_tbl,
                            sort_keys=[(pk, "ascending") for pk in schema.primary_keys],
                        ),
                    )
                )
            else:
                rdd6 = rdd5

            # save the parition table to s3
            rdd7 = rdd6.map(
                lambda partition_table: save_partition_tbl(
                    partition_table, dataset_name, dataset_version, s3_storage, s3_path_factory
                )
            )
            rdd_list = rdd7.collect()
            written = {partition: size for partition, size in rdd_list}
        else:
            # In normal operation, rdd5 may have thousands of partitions at the start of operation,
            # however because there are typically only at most three dataset splits (train,test,val),
            # the output of rdd5 has only three partitions with any data. Empriically this has lead to
            # issues related to driver memory. As a work around, we can instead perform this reduction
            # manually outside of the JVM without much of a performance penalty since spark's parallelism
            # was not being taken advantage of anyway.

            # We are going to iterate over each partition, grab its pyarrow table, and merge them.
            # to avoid localIterator from running rdd4 sequentially, we first cache it and trigger an action.
            rdd4 = rdd4.cache()
            _ = rdd4.count()

            # the rest of this is adapted from the map task persist
            merged_dicts: Dict[str, Dict[str, List[Any]]] = {}
            for partition_key, row in rdd4.toLocalIterator():
                current_dict: Dict[str, List[Any]] = merged_dicts.get(partition_key, {})
                for col in row.keys():
                    if col in current_dict:
                        current_dict[col].append(row[col])
                    else:
                        current_dict[col] = [row[col]]
                merged_dicts[partition_key] = current_dict

            arrow_dict = {}
            for partition_key, data_dict in merged_dicts.items():
                data_table = pa.Table.from_pydict(data_dict)
                if sort:
                    arrow_dict[partition_key] = pc.take(
                        pa.Table.from_pydict(data_dict),
                        pc.sort_indices(data_table, sort_keys=[(pk, "ascending") for pk in schema.primary_keys]),
                    )
                else:
                    arrow_dict[partition_key] = data_table

            written = {}
            for partition_key, pa_table in arrow_dict.items():
                self.save_partition_tbl(
                    (partition_key, pa_table), dataset_name, dataset_version, self.s3_storage, self.s3_path_factory
                )
                written[partition_key] = pa_table.num_rows

        return written

    @staticmethod
    def get_row_keys(
        partition_data_tup: Tuple[str, ParsedExample], schema: schema_module.DatasetSchema
    ) -> PrimaryKeyTuple:
        """
        Get the keys of a row based on the parition tuple and the data schema.

        :param partition_data_tup: Tuple of partition id and ParsedExample row
        :type partition_data_tup: Tuple[str, ParsedExample]
        :return: Tuple of primary key values from parsed row and schema
        :rtype: PrimaryKeyTuple
        """
        partition, data = partition_data_tup
        return (partition,) + tuple(data[pk] for pk in schema.primary_keys)
