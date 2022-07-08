import abc
import warnings
from typing import Any, Dict, Optional

from wicker import schema
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.plugins.spark import UnparsedExample


class AbstractDataPersistor(abc.ABC):
    """
    Abstract class for persisting data onto a user defined cloud or local instance.

    Only s3 is supported right now but plan to support other data stores
    (BigQuery, Whatever the hell Azure is called, Postgres)
    """

    def __init__(
        self,
        s3_storage: S3DataStorage = S3DataStorage(),
        s3_path_factory: S3PathFactory = S3PathFactory(),
        current_schema: Optional[schema.DatasetSchema] = None,
        current_dataset_name: Optional[str] = None,
        current_dataset_version: Optional[str] = None,
    ) -> None:
        """
        Init a Persister

        :param s3_storage: The storage abstraction for S3
        :type s3_storage: S3DataStore
        :param s3_path_factory: The path factory for generating s3 paths
                                based on dataset name and version
        :type s3_path_factory: S3PathFactory
        :param current_schema: Dataschema to be set initially, no setting makes
            empty schema
        :type current_schema: wicker.schema.schema.DatasetSchema
        :param current_dataset_name: Name of the dataset to be set initially
            empty sets to unassigned
        :type current_dataset_name: str
        :param current_dataset_version: Version of the dataset to be set intitially
            empty sets to unassigned
        :type current_dataset_version: str
        """
        super().__init__()
        self.s3_storage = s3_storage
        self.s3_path_factory = s3_path_factory
        self._current_schema = current_schema
        self._current_dataset_name = (current_dataset_name,)
        self._current_dataset_version = current_dataset_version

    @abc.abstractmethod
    def persist_wicker_dataset(
        self,
        dataset_name: str,
        dataset_version: str,
        dataset_schema: schema.DatasetSchema,
        dataset: Any,
    ) -> Dict[str, int]:
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
        raise NotImplementedError("Method, persist_wicker_dataset, needs to" "be implemented in inhertiance class.")

    @abc.abstractmethod
    def persist_current_wicker_dataset(
        self,
    ) -> Dict[str, int]:
        """
        Persist the current dataset defined by name, version, schema, and data.
        """
        raise NotImplementedError(
            "Method, persist_current_wicker_dataset, needs" "to be implemented in inheritance class"
        )

    def _parse_row(self, data_row: UnparsedExample):
        """
        Parse a row to test for validation errors.

        :param data_row: Data row to be parsed
        :type data_row: UnparsedExample
        :return: parsed row containing the correct types associated with schema
        :rtype: ParsedExample
        """
        if self._current_schema is not None:
            return schema.dataparsing.parse_example(data_row, self._current_schema)
        warnings.warn("_current_schema not set, could not parse")
