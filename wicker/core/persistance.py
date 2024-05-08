import abc
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc

from wicker import schema as schema_module
from wicker.core.column_files import ColumnBytesFileWriter
from wicker.core.definitions import DatasetID
from wicker.core.shuffle import save_index
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.schema import dataparsing, serialization

MAX_COL_FILE_NUMROW = 50  # TODO(isaak-willett): Magic number, we should derive this based on row size

UnparsedExample = Dict[str, Any]
ParsedExample = Dict[str, Any]
PointerParsedExample = Dict[str, Any]


class AbstractDataPersistor(abc.ABC):
    """
    Abstract class for persisting data onto a user defined cloud or local instance.

    Only s3 is supported right now but plan to support other data stores
    (BigQuery, Azure, Postgres)
    """

    def __init__(
        self,
        s3_storage: S3DataStorage = S3DataStorage(),
        s3_path_factory: S3PathFactory = S3PathFactory(),
    ) -> None:
        """
        Init a Persister

        :param s3_storage: The storage abstraction for S3
        :type s3_storage: S3DataStore
        :param s3_path_factory: The path factory for generating s3 paths
                                based on dataset name and version
        :type s3_path_factory: S3PathFactory
        """
        super().__init__()
        self.s3_storage = s3_storage
        self.s3_path_factory = s3_path_factory

    @abc.abstractmethod
    def persist_wicker_dataset(
        self,
        dataset_name: str,
        dataset_version: str,
        dataset_schema: schema_module.DatasetSchema,
        dataset: Any,
    ) -> Optional[Dict[str, int]]:
        """
        Persist a user specified dataset defined by name, version, schema, and data.

        :param dataset_name: Name of the dataset
        :type dataset_name: str
        :param dataset_version: Version of the dataset
        :type: dataset_version: str
        :param dataset_schema: Schema of the dataset
        :type dataset_schema: wicker.schema.schema.DatasetSchema
        :param dataset: Data of the dataset
        :type dataset: User defined
        """
        raise NotImplementedError("Method, persist_wicker_dataset, needs to be implemented in inhertiance class.")

    @staticmethod
    def parse_row(data_row: UnparsedExample, schema: schema_module.DatasetSchema) -> ParsedExample:
        """
        Parse a row to test for validation errors.

        :param data_row: Data row to be parsed
        :type data_row: UnparsedExample
        :return: parsed row containing the correct types associated with schema
        :rtype: ParsedExample
        """
        return dataparsing.parse_example(data_row, schema)

        # Write data to Column Byte Files

    @staticmethod
    def persist_wicker_partition(
        dataset_name: str,
        spark_partition_iter: Iterable[Tuple[str, ParsedExample]],
        schema: schema_module.DatasetSchema,
        s3_storage: S3DataStorage,
        s3_path_factory: S3PathFactory,
        target_max_column_file_numrows: int = 50,
    ) -> Iterable[Tuple[str, PointerParsedExample]]:
        """Persists a Spark partition of examples with parsed bytes into S3Storage as ColumnBytesFiles,
        returning a new Spark partition of examples with heavy-pointers and metadata only.
        :param dataset_name: dataset name
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
                    dataset_name=dataset_name,
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

    @staticmethod
    def save_partition_tbl(
        partition_table_tuple: Tuple[str, pa.Table],
        dataset_name: str,
        dataset_version: str,
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
            dataset_name,
            dataset_version,
            {partition: pa_tbl},
            s3_storage=s3_storage,
            s3_path_factory=s3_path_factory,
        )
        return (partition, pa_tbl.num_rows)


def persist_wicker_dataset(
    dataset_name: str,
    dataset_version: str,
    dataset_schema: schema_module.DatasetSchema,
    dataset: Any,
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
    return BasicPersistor(s3_storage, s3_path_factory).persist_wicker_dataset(
        dataset_name, dataset_version, dataset_schema, dataset
    )


class BasicPersistor(AbstractDataPersistor):
    """
    Basic persistor class that persists wicker data on s3 in a non sorted manner.

    We will move to supporting other features like shuffling, other data engines, etc...
    """

    def __init__(
        self, s3_storage: S3DataStorage = S3DataStorage(), s3_path_factory: S3PathFactory = S3PathFactory()
    ) -> None:
        super().__init__(s3_storage, s3_path_factory)

    def persist_wicker_dataset(
        self, dataset_name: str, dataset_version: str, dataset_schema: schema_module.DatasetSchema, dataset: Any
    ) -> Optional[Dict[str, int]]:
        """
        Persist a user defined dataset, pushing data to s3 in a basic manner

        :param dataset_name: Name of the dataset
        :type dataset_name: str
        :param dataset_version: Version of the dataset
        :type: dataset_version: str
        :param dataset_schema: Schema of the dataset
        :type dataset_schema: wicker.schema.schema.DatasetSchema
        :param dataset: Data of the dataset
        :type dataset: User defined
        """
        # what needs to be done within this function
        # 1. Check if the variables are set
        # check if variables have been set ie: not None
        if (
            not isinstance(dataset_name, str)
            or not isinstance(dataset_version, str)
            or not isinstance(dataset_schema, schema_module.DatasetSchema)
        ):
            raise ValueError("Current dataset variables not all set, set all to proper not None values")

        # 2. Put the schema up on s3
        schema_path = self.s3_path_factory.get_dataset_schema_path(
            DatasetID(name=dataset_name, version=dataset_version)
        )
        self.s3_storage.put_object_s3(serialization.dumps(dataset_schema).encode("utf-8"), schema_path)

        # 3. Validate the rows and ensure data is well formed, sort while doing
        dataset_0 = [(row[0], self.parse_row(row[1], dataset_schema)) for row in dataset]

        # 4. Sort the dataset if not sorted
        sorted_dataset_0 = sorted(dataset_0, key=lambda tup: tup[0])

        # 6. Persist the partitions to S3
        metadata_iterator = self.persist_wicker_partition(
            dataset_name,
            sorted_dataset_0,
            dataset_schema,
            self.s3_storage,
            self.s3_path_factory,
            MAX_COL_FILE_NUMROW,
        )

        # 7. Create the parition table, need to combine keys in a way we can form table
        # split into k dicts where k is partition number and the data is a list of values
        # for each key for all the dicts in the partition
        merged_dicts: Dict[str, Dict[str, List[Any]]] = {}
        for partition_key, row in metadata_iterator:
            current_dict: Dict[str, List[Any]] = merged_dicts.get(partition_key, {})
            for col in row.keys():
                if col in current_dict:
                    current_dict[col].append(row[col])
                else:
                    current_dict[col] = [row[col]]
            merged_dicts[partition_key] = current_dict
        # convert each of the dicts to a pyarrow table in the same way SparkPersistor
        # converts, needed to ensure parity between the two
        arrow_dict = {}
        for partition_key, data_dict in merged_dicts.items():
            data_table = pa.Table.from_pydict(data_dict)
            arrow_dict[partition_key] = pc.take(
                pa.Table.from_pydict(data_dict),
                pc.sort_indices(data_table, sort_keys=[(pk, "ascending") for pk in dataset_schema.primary_keys]),
            )

        # 8. Persist the partition table to s3
        written_dict = {}
        for partition_key, pa_table in arrow_dict.items():
            self.save_partition_tbl(
                (partition_key, pa_table), dataset_name, dataset_version, self.s3_storage, self.s3_path_factory
            )
            written_dict[partition_key] = pa_table.num_rows

        return written_dict
