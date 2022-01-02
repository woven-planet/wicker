"""Spark plugin for writing a dataset with Spark only (no external metadata database required)

This plugin does an expensive global sorting step using Spark, which could be prohibitive
for large datasets.
"""


from __future__ import annotations

import collections
import dataclasses
from typing import Any, Dict, Iterable, Tuple

import pyarrow as pa

try:
    import pyspark
except ImportError:
    raise RuntimeError(
        "pyspark is not detected in current environment, install Wicker with extra arguments:"
        " `pip install wicker[spark]`"
    )

from wicker import schema
from wicker.core.definitions import DatasetDefinition, DatasetID
from wicker.core.shuffle import ShuffleJob, ShuffleJobFactory, ShuffleWorker, save_index
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.core.writer import (
    AbstractDatasetWriterMetadataDatabase,
    DatasetWriter,
    DatasetWriterBackend,
)
from wicker.plugins.dynamodb import DynamodbMetadataDatabase
from wicker.schema import serialization


@dataclasses.dataclass
class _ShuffleWorkerResults:
    partition: str
    pa_table: pa.Table


def persist_wicker_dataset(
    dataset_name: str,
    dataset_version: str,
    dataset_schema: schema.DatasetSchema,
    rdd: pyspark.rdd.RDD[Tuple[str, Dict[str, Any]]],
    # TODO(jchia): Extend this to support other databases
    metadata_database: AbstractDatasetWriterMetadataDatabase = DynamodbMetadataDatabase(),
) -> Dict[str, int]:
    """Persists a Spark RDD as a Wicker dataset. RDD must be provided as an RDD of Tuples of (partition, example),
    where `partition` is a string representing the dataset partition (e.g. train/test/eval) for that given row,
    and `example` is a Python Dict[str, Any] of the (non-validated) data to be written into the dataset.

    This function will perform validations, do a global sort based on primary keys, serialize the examples into bytes,
    save the data in S3 and persist the written data into the configured Wicker S3 bucket.

    :param dataset_name: name of the dataset
    :param dataset_version: version of the dataset
    :param dataset_schema: schema of the dataset
    :param rdd: RDD of data to be persisted as a Wicker dataset
    :param metadata_database: Metadata database to use as intermediate storage for shuffle
    :return: A dictionary of partition name to size
    """
    # Write schema to S3
    s3_storage = S3DataStorage()
    s3_path_factory = S3PathFactory()
    schema_path = s3_path_factory.get_dataset_schema_path(DatasetID(name=dataset_name, version=dataset_version))
    s3_storage.put_object_s3(serialization.dumps(dataset_schema).encode("utf-8"), schema_path)

    # Write rows to S3 storage for shuffle
    def add_examples(partition_data_tups: Iterable[Tuple[str, Dict[str, Any]]]) -> Iterable[Any]:
        with DatasetWriter(
            dataset_definition=DatasetDefinition(
                DatasetID(name=dataset_name, version=dataset_version),
                schema=dataset_schema,
            ),
            metadata_database=metadata_database,
        ) as writer:
            for partition, data in partition_data_tups:
                writer.add_example(partition, data)
        yield

    rdd.mapPartitions(add_examples).collect()

    # Run shuffling on the same cluster by downloading from the metadata_database and S3
    def run_shuffling_job(job: ShuffleJob) -> _ShuffleWorkerResults:
        worker = ShuffleWorker(storage=S3DataStorage(), s3_path_factory=S3PathFactory())
        return _ShuffleWorkerResults(
            pa_table=worker.process_job(job),
            partition=job.dataset_partition.partition,
        )

    backend = DatasetWriterBackend(s3_path_factory, s3_storage, metadata_database)
    job_factory = ShuffleJobFactory(backend)
    shuffle_jobs = list(job_factory.build_shuffle_jobs(DatasetID(name=dataset_name, version=dataset_version)))
    shuffle_results = rdd.context.parallelize(shuffle_jobs, len(shuffle_jobs)).map(run_shuffling_job).collect()

    results_by_partition = collections.defaultdict(list)
    for result in shuffle_results:
        results_by_partition[result.partition].append(result.pa_table)

    tables_by_partition: Dict[str, pa.Table] = {}
    for partition in results_by_partition:
        tables_by_partition[partition] = pa.concat_tables(results_by_partition[partition])

    save_index(
        dataset_name,
        dataset_version,
        tables_by_partition,
        s3_storage=s3_storage,
        s3_path_factory=s3_path_factory,
    )
    return {partition_name: len(tables_by_partition[partition_name]) for partition_name in tables_by_partition}
