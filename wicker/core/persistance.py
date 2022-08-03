import abc
from typing import Any, Dict, Optional

from wicker import schema
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.schema import dataparsing

UnparsedExample = Dict[str, Any]
ParsedExample = Dict[str, Any]


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
        dataset_schema: schema.DatasetSchema,
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
    def parse_row(data_row: UnparsedExample, schema: schema.DatasetSchema) -> ParsedExample:
        """
        Parse a row to test for validation errors.

        :param data_row: Data row to be parsed
        :type data_row: UnparsedExample
        :return: parsed row containing the correct types associated with schema
        :rtype: ParsedExample
        """
        return dataparsing.parse_example(data_row, schema)
