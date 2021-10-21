from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any, Dict, TypeVar

# flake8 linting struggles with the module name `schema` and DatasetDefinition dataclass field name `schema`
from wicker import schema  # noqa: 401

Example = TypeVar("Example")
ExampleMetadata = Dict[str, Any]


DATASET_ID_REGEX = re.compile(r"^(?P<name>[0-9a-zA-Z_]+)/(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$")


@dataclasses.dataclass(frozen=True)
class DatasetID:
    """Representation of the unique identifier of a dataset

    `name` should be alphanumeric and contain no spaces, only underscores
    `version` should be a semantic version (e.g. 1.0.0)
    """

    name: str
    version: str

    @classmethod
    def from_str(cls, s: str) -> DatasetID:
        """Parses a DatasetID from a string"""
        match = DATASET_ID_REGEX.match(s)
        if not match:
            raise ValueError(f"{s} is not a valid DatasetID")
        return cls(name=match["name"], version=match["version"])

    def __str__(self) -> str:
        """Helper function to return the representation of the dataset version as a path-like string"""
        return f"{self.name}/{self.version}"

    @staticmethod
    def validate_dataset_id(name: str, version: str) -> None:
        """Validates the name and version of a dataset"""
        if not re.match(r"^[0-9a-zA-Z_]+$", name):
            raise ValueError(
                f"Provided dataset name {name} must be alphanumeric and contain no spaces, only underscores"
            )
        if not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", version):
            raise ValueError(
                f"Provided dataset version {version} should be a semantic version without any prefixes/suffixes"
            )

    def __post_init__(self) -> None:
        DatasetID.validate_dataset_id(self.name, self.version)


@dataclasses.dataclass(frozen=True)
class DatasetDefinition:
    """Representation of the definition of a dataset (immutable once dataset is added)

    `name` should be alphanumeric and contain no spaces, only underscores
    `version` should be a semantic version (e.g. 1.0.0)
    """

    dataset_id: DatasetID
    schema: schema.DatasetSchema

    @property
    def identifier(self) -> DatasetID:
        return DatasetID(name=self.dataset_id.name, version=self.dataset_id.version)

    def __post_init__(self) -> None:
        DatasetID.validate_dataset_id(self.dataset_id.name, self.dataset_id.version)


@dataclasses.dataclass(frozen=True)
class DatasetPartition:
    """Representation of the definition of a partition within dataset

    The partition here is meant to represent the common train/val/test splits of a dataset, but
    could also represent partitions for other use cases.

    `partition` should be alphanumeric and contain no spaces, only underscores
    """

    dataset_id: DatasetID
    partition: str

    def __str__(self) -> str:
        """Helper function to return the representation of the partition as a path-like string"""
        return f"{self.dataset_id.name}/{self.dataset_id.version}/{self.partition}"


class DatasetState(enum.Enum):
    """Representation of the state of a dataset"""

    STAGED = "PENDING"
    COMMITTED = "COMMITTED"
