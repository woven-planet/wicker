import dataclasses
import heapq
from typing import Generator, List, Tuple

try:
    import pynamodb  # type: ignore
except ImportError:
    raise RuntimeError(
        "pynamodb is not detected in current environment, install Wicker with extra arguments:"
        " `pip install wicker[dynamodb]`"
    )
from pynamodb.attributes import NumberAttribute, UnicodeAttribute
from pynamodb.models import Model
from retry import retry

from wicker.core.config import get_config
from wicker.core.definitions import DatasetID
from wicker.core.writer import (
    AbstractDatasetWriterMetadataDatabase,
    ExampleKey,
    MetadataDatabaseScanRow,
)

# DANGER: If these constants are ever changed, this is a backward-incompatible change.
# Make sure that all the writers and readers of dynamodb are in sync when changing.
NUM_DYNAMODB_SHARDS = 32
HASH_PREFIX_LENGTH = 4
DYNAMODB_QUERY_PAGINATION_LIMIT = 1000


@dataclasses.dataclass(frozen=True)
class DynamoDBConfig:
    table_name: str
    region: str


def get_dynamodb_config() -> DynamoDBConfig:
    raw_config = get_config().raw
    if "dynamodb_config" not in raw_config:
        raise RuntimeError("Could not find 'dynamodb' parameters in config")
    if "table_name" not in raw_config["dynamodb_config"]:
        raise RuntimeError("Could not find 'table_name' parameter in config.dynamodb_config")
    if "region" not in raw_config["dynamodb_config"]:
        raise RuntimeError("Could not find 'region' parameter in config.dynamodb_config")
    return DynamoDBConfig(
        table_name=raw_config["dynamodb_config"]["table_name"],
        region=raw_config["dynamodb_config"]["region"],
    )


class DynamoDBExampleDBRow(Model):
    class Meta:
        table_name = get_dynamodb_config().table_name
        region = get_dynamodb_config().region

    dataset_id = UnicodeAttribute(hash_key=True)
    example_id = UnicodeAttribute(range_key=True)
    partition = UnicodeAttribute()
    row_data_path = UnicodeAttribute()
    row_size = NumberAttribute()


def _key_to_row_id_and_shard_id(example_key: ExampleKey) -> Tuple[str, int]:
    """Deterministically convert an ExampleKey into a row_id and a shard_id which are used as
    DynamoDB RangeKeys and HashKeys respectively.

    HashKeys help to increase read/write throughput by allowing us to use different partitions.
    RangeKeys are how the rows are sorted within partitions by DynamoDB.

    We completely randomize the hash and range key to optimize for write throughput, but this means
    that sorting needs to be done entirely client-side in our application.
    """
    partition_example_id = f"{example_key.partition}//{'/'.join([str(key) for key in example_key.primary_key_values])}"
    hash = example_key.hash()
    shard = int(hash, 16) % NUM_DYNAMODB_SHARDS
    return partition_example_id, shard


def _dataset_shard_name(dataset_id: DatasetID, shard_id: int) -> str:
    """Get the name of the DynamoDB partition for a given dataset_definition and shard number"""
    return f"{dataset_id}_shard{shard_id:02d}"


class DynamodbMetadataDatabase(AbstractDatasetWriterMetadataDatabase):
    def save_row_metadata(self, dataset_id: DatasetID, key: ExampleKey, location: str, row_size: int) -> None:
        """Saves a row in the metadata database, marking it as having been uploaded to S3 and
        ready for shuffling.

        :param dataset_id: The ID of the dataset to save to
        :param key: The key of the example
        :param location: The location of the example in S3
        :param row_size: The size of the file in S3
        """
        partition_example_id, shard_id = _key_to_row_id_and_shard_id(key)
        entry = DynamoDBExampleDBRow(
            dataset_id=_dataset_shard_name(dataset_id, shard_id),
            example_id=partition_example_id,
            partition=key.partition,
            row_data_path=location,
            row_size=row_size,
        )
        entry.save()

    def scan_sorted(self, dataset_id: DatasetID) -> Generator[MetadataDatabaseScanRow, None, None]:
        """Scans the MetadataDatabase for a **SORTED** list of ExampleKeys for a given dataset. Should be fast O(minutes)
        to perform as this will be called from a single machine to assign chunks to jobs to run.

        :param dataset: The dataset to scan the metadata database for
        :return: a Generator of MetadataDatabaseScanRow in **SORTED** primary_key order
        """

        @retry(pynamodb.exceptions.QueryError, tries=10, backoff=2, delay=4, jitter=(0, 2))
        def shard_iterator(shard_id: int) -> Generator[DynamoDBExampleDBRow, None, None]:
            """Yields DynamoDBExampleDBRows from a given shard to exhaustion, sorted by example_id in ascending order
            DynamoDBExampleDBRows are yielded in sorted order of the Dynamodb RangeKey, which is the example_id
            """
            hash_key = _dataset_shard_name(dataset_id, shard_id)
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

        # Individual shards have their rows already in sorted order
        # Elements are popped off each shard to exhaustion and put into a minheap
        # We yield from the heap to exhaustion to provide a stream of globally sorted example_ids
        heap: List[Tuple[str, int, DynamoDBExampleDBRow]] = []
        shard_iterators = {shard_id: shard_iterator(shard_id) for shard_id in range(NUM_DYNAMODB_SHARDS)}
        for shard_id, iterator in shard_iterators.items():
            try:
                row = next(iterator)
                heapq.heappush(heap, (row.example_id, shard_id, row))
            except StopIteration:
                pass
        while heap:
            _, shard_id, row = heapq.heappop(heap)
            try:
                nextrow = next(shard_iterators[shard_id])
                heapq.heappush(heap, (nextrow.example_id, shard_id, nextrow))
            except StopIteration:
                pass
            yield MetadataDatabaseScanRow(
                partition=row.partition,
                row_data_path=row.row_data_path,
                row_size=row.row_size,
            )
