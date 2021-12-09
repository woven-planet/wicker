"""Classes handling the shuffling of data in S3 when committing a dataset.

When committing a dataset, we sort the data by primary_key before materializing in S3
as Parquet files.

1. The ShuffleJob is the unit of work, and it is just an ordered set of Examples that should
be bundled together in one Parquet file

2. The ShuffleJobFactory produces ShuffleJobs, using a DatasetWriter object to retrieve the
written examples for divvying up as ShuffleJobs.

3. ShuffleWorkers receive the ShuffleJobs and perform the act of retrieving the data and
persisting the data into S3 as Parquet files (one for each ShuffleJob)
"""

import dataclasses

from typing import List

from wicker.core.definitions import DatasetPartition
from wicker.schema import schema
from wicker.core.writer import DatasetWriter


DEFAULT_COLUMN_FILE_SIZE_UPPER_BOUND_BYTES = 250000000
DEFAULT_COLUMN_FILE_ROWGROUP_SIZE_TARGET = 50


@dataclasses.dataclass
class ShuffleJob:
    """Represents all the shuffling operations that will happen for a given partition (train/eval/test) on a given
    compute shard."""

    dataset_partition: DatasetPartition
    schema: schema.DatasetSchema
    examples: List[str]

    target_column_file_size: int
    target_parquet_rowgroup_size: int


class ShuffleJobFactory:

    def __init__(
        self,
        writer: DatasetWriter,
        target_column_file_size: int = DEFAULT_COLUMN_FILE_SIZE_UPPER_BOUND_BYTES,
        target_parquet_rowgroup_size: int = DEFAULT_COLUMN_FILE_ROWGROUP_SIZE_TARGET,
    ):
        self.writer = writer

        # Factory configurations
        self.target_column_file_size = target_column_file_size
        self.target_parquet_rowgroup_size = target_parquet_rowgroup_size

    def 
