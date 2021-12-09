import os
import pickle
from typing import Any, Dict, Generator, Optional, Tuple

import pynamodb
from pynamodb.attributes import NumberAttribute, UnicodeAttribute
from pynamodb.models import Model
from retry import retry

from wicker.core.definitions import DatasetDefinition
from wicker.core.writer import AsyncDatasetWriter, ExampleKey

# DANGER: If these constants are ever changed, this is a backward-incompatible change.
# Make sure that all the writers and readers of dynamodb are in sync when changing.
NUM_DYNAMODB_SHARDS = 32
HASH_PREFIX_LENGTH = 4
DYNAMODB_QUERY_PAGINATION_LIMIT = 1000


class DynamoDBExampleDBRow(Model):
    class Meta:
        table_name = "l5ml-datastore"
        region = "us-west-2"  # TODO(jchia: Figure out how to pipe this in as a parameter)

    dataset_id = UnicodeAttribute(hash_key=True)
    example_id = UnicodeAttribute(range_key=True)
    row_data_path = UnicodeAttribute()
    row_size = NumberAttribute()


def _key_to_row_id_and_shard_id(example_key: ExampleKey) -> Tuple[str, int]:
    """Convert an ExampleKey into a DynamoDB HashKey and RangeKey in a deterministic way.

    HashKeys help to increase write throughput by allowing us to write to different partitions.
    RangeKeys are how the rows are sorted within partitions by DynamoDB.
    """
    hash = example_key.hash()
    shard = int(hash[0], 16) % NUM_DYNAMODB_SHARDS
    return hash, shard


def _dataset_shard_name(dataset_definition: DatasetDefinition, shard_id: int) -> str:
    return f"{dataset_definition.identifier}_shard{shard_id:02d}"


class DynamodbAsyncDatasetWriter(AsyncDatasetWriter):
    def _save_row_impl(self, key: ExampleKey, data: Dict[str, Any]) -> None:
        """Subclasses should implement this method to save each individual row

        :param key: key of row
        :param data: validated data for row
        """
        hashed_row_key = key.hash()
        pickled_row = pickle.dumps(data)  # TODO(jchia): Do we want a more sophisticated storage format here?
        row_s3_path = os.path.join(
            self.s3_path_factory.get_temporary_row_files_path(self.dataset_definition.identifier),
            hashed_row_key,
        )

        # Write to S3 and write a pointer in DynamoDB
        self.s3_storage.put_object_s3(pickled_row, row_s3_path)
        example_id, shard_id = _key_to_row_id_and_shard_id(key)
        entry = DynamoDBExampleDBRow(
            dataset_id=_dataset_shard_name(self.dataset_definition, shard_id),
            example_id=example_id,
            row_data_path=row_s3_path,
            row_size=len(pickled_row),
        )
        entry.save()

    @retry(pynamodb.exceptions.QueryError, tries=10, backoff=2, delay=4, jitter=(0, 2))
    def _scan_unordered(self) -> Generator[ExampleKey, None, None]:
        """Scans the dataset for a non-sorted list of ExampleKeys for a given partition. Should be fast (O(minutes))
        to perform for each partition as this is called on a single machine to assign chunks to jobs to run.
        """
        for shard_id in range(NUM_DYNAMODB_SHARDS):
            # Run a paginated query for all data from the shard, yielding data until the query runs empty
            hash_key = _dataset_shard_name(self.dataset_definition, shard_id)
            last_evaluated_key = None
            while True:
                query_results = DynamoDBExampleDBRow.query(
                    hash_key,
                    consistent_read=True,
                    last_evaluated_key=last_evaluated_key,
                    limit=DYNAMODB_QUERY_PAGINATION_LIMIT,
                )
                yield from query_results
                if query_results.last_evaluated_key is None:
                    break
                last_evaluated_key = query_results.last_evaluated_key
