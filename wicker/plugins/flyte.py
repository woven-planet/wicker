import collections
import dataclasses
import json
import tempfile
from typing import Dict, List, Type, cast

try:
    import flytekit  # type: ignore
except ImportError:
    raise RuntimeError(
        "flytekit is not detected in current environment, install Wicker with extra arguments:"
        " `pip install wicker[flyte]`"
    )
import pyarrow as pa  # type: ignore
from flytekit.extend import TypeEngine, TypeTransformer  # type: ignore

from wicker.core.definitions import DatasetID, DatasetPartition
from wicker.core.shuffle import ShuffleJob, ShuffleJobFactory, ShuffleWorker, save_index
from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.core.writer import DatasetWriterBackend
from wicker.plugins.dynamodb import DynamodbMetadataDatabase

###
# Custom type definitions to clean up passing of data between Flyte tasks
###


class ShuffleJobTransformer(TypeTransformer[ShuffleJob]):
    _TYPE_INFO = flytekit.BlobType(
        format="binary",
        dimensionality=flytekit.BlobType.BlobDimensionality.SINGLE,
    )

    def __init__(self):
        super(ShuffleJobTransformer, self).__init__(name="shufflejob-transform", t=ShuffleJob)

    def get_literal_type(self, t: Type[ShuffleJob]) -> flytekit.LiteralType:
        """
        This is useful to tell the Flytekit type system that ``ShuffleJob`` actually refers to what corresponding type
        In this example, we say its of format binary (do not try to introspect) and there are more than one files in it
        """
        return flytekit.LiteralType(blob=self._TYPE_INFO)

    @staticmethod
    def _shuffle_jobs_to_bytes(job: ShuffleJob) -> bytes:
        return json.dumps(
            {
                "dataset_partition": {
                    "dataset_id": {
                        "name": job.dataset_partition.dataset_id.name,
                        "version": job.dataset_partition.dataset_id.version,
                    },
                    "partition": job.dataset_partition.partition,
                },
                "files": job.files,
            }
        ).encode("utf-8")

    @staticmethod
    def _shuffle_jobs_from_bytes(b: bytes) -> ShuffleJob:
        data = json.loads(b.decode("utf-8"))
        return ShuffleJob(
            dataset_partition=DatasetPartition(
                dataset_id=DatasetID(
                    name=data["dataset_partition"]["dataset_id"]["name"],
                    version=data["dataset_partition"]["dataset_id"]["version"],
                ),
                partition=data["dataset_partition"]["partition"],
            ),
            files=[(path, size) for path, size in data["files"]],
        )

    def to_literal(
        self,
        ctx: flytekit.FlyteContext,
        python_val: ShuffleJob,
        python_type: Type[ShuffleJob],
        expected: flytekit.LiteralType,
    ) -> flytekit.Literal:
        """
        This method is used to convert from given python type object ``MyDataset`` to the Literal representation
        """
        # Step 1: lets upload all the data into a remote place recommended by Flyte
        remote_path = ctx.file_access.get_random_remote_path()
        with tempfile.NamedTemporaryFile() as tmpfile:
            tmpfile.write(self._shuffle_jobs_to_bytes(python_val))
            tmpfile.flush()
            tmpfile.seek(0)
            ctx.file_access.upload(tmpfile.name, remote_path)
        # Step 2: lets return a pointer to this remote_path in the form of a literal
        return flytekit.Literal(
            scalar=flytekit.Scalar(
                blob=flytekit.Blob(uri=remote_path, metadata=flytekit.BlobMetadata(type=self._TYPE_INFO))
            )
        )

    def to_python_value(
        self, ctx: flytekit.FlyteContext, lv: flytekit.Literal, expected_python_type: Type[ShuffleJob]
    ) -> ShuffleJob:
        """
        In this function we want to be able to re-hydrate the custom object from Flyte Literal value
        """
        # Step 1: lets download remote data locally
        local_path = ctx.file_access.get_random_local_path()
        ctx.file_access.download(lv.scalar.blob.uri, local_path)
        # Step 2: create the ShuffleJob object
        with open(local_path, "rb") as f:
            return self._shuffle_jobs_from_bytes(f.read())


@dataclasses.dataclass
class ShuffleWorkerResults:
    partition: str
    pa_table: pa.Table


class ShuffleWorkerResultsTransformer(TypeTransformer[ShuffleWorkerResults]):
    _TYPE_INFO = flytekit.BlobType(
        format="binary",
        dimensionality=flytekit.BlobType.BlobDimensionality.SINGLE,
    )

    def __init__(self):
        super(ShuffleWorkerResultsTransformer, self).__init__(
            name="shuffleworkerresults-transform",
            t=ShuffleWorkerResults,
        )

    def get_literal_type(self, t: Type[ShuffleWorkerResults]) -> flytekit.LiteralType:
        """
        This is useful to tell the Flytekit type system that ``ShuffleWorkerResults`` actually refers to
        what corresponding type In this example, we say its of format binary (do not try to introspect) and
        there are more than one files in it
        """
        return flytekit.LiteralType(blob=self._TYPE_INFO)

    def to_literal(
        self,
        ctx: flytekit.FlyteContext,
        python_val: ShuffleWorkerResults,
        python_type: Type[ShuffleWorkerResults],
        expected: flytekit.LiteralType,
    ) -> flytekit.Literal:
        """
        This method is used to convert from given python type object ``ShuffleWorkerResults``
        to the Literal representation
        """
        # Step 1: lets upload all the data into a remote place recommended by Flyte
        remote_path = ctx.file_access.get_random_remote_path()
        local_path = ctx.file_access.get_random_local_path()
        with pa.ipc.new_stream(
            local_path, python_val.pa_table.schema.with_metadata({"partition": python_val.partition})
        ) as stream:
            stream.write(python_val.pa_table)
        ctx.file_access.upload(local_path, remote_path)
        # Step 2: lets return a pointer to this remote_path in the form of a literal
        return flytekit.Literal(
            scalar=flytekit.Scalar(
                blob=flytekit.Blob(uri=remote_path, metadata=flytekit.BlobMetadata(type=self._TYPE_INFO))
            )
        )

    def to_python_value(
        self, ctx: flytekit.FlyteContext, lv: flytekit.Literal, expected_python_type: Type[ShuffleWorkerResults]
    ) -> ShuffleWorkerResults:
        """
        In this function we want to be able to re-hydrate the custom object from Flyte Literal value
        """
        # Step 1: lets download remote data locally
        local_path = ctx.file_access.get_random_local_path()
        ctx.file_access.download(lv.scalar.blob.uri, local_path)
        # Step 2: create the ShuffleWorkerResults object
        with pa.ipc.open_stream(local_path) as reader:
            return ShuffleWorkerResults(
                pa_table=pa.Table.from_batches([b for b in reader]),
                partition=reader.schema.metadata[b"partition"].decode("utf-8"),
            )


TypeEngine.register(ShuffleJobTransformer())
TypeEngine.register(ShuffleWorkerResultsTransformer())


###
# Task and Workflow definitions
###


@flytekit.task(requests=flytekit.Resources(mem="2Gi", cpu="1"), retries=2)
def initialize_dataset(
    schema_json_str: str,
    dataset_id: str,
) -> str:
    """Write the schema to the storage."""
    s3_path_factory = S3PathFactory()
    s3_storage = S3DataStorage()
    schema_path = s3_path_factory.get_dataset_schema_path(DatasetID.from_str(dataset_id))
    s3_storage.put_object_s3(schema_json_str.encode("utf-8"), schema_path)
    return schema_json_str


@flytekit.task(requests=flytekit.Resources(mem="8Gi", cpu="2"), retries=2)
def create_shuffling_jobs(
    schema_json_str: str,
    dataset_id: str,
    worker_max_working_set_size: int = 16384,
) -> List[ShuffleJob]:
    """Read the DynamoDB and return a list of shuffling jobs to distribute.

    The job descriptions are stored into files managed by Flyte.
    :param dataset_id: string representation of the DatasetID we need to process (dataset name + version).
    :param schema_json_str: string representation of the dataset schema
    :param max_rows_per_worker: Maximum number of rows to assign per working set, defaults to 16384 but can be
        increased if dataset sizes are so large that we want to use fewer workers and don't mind the committing
        step taking longer per-worker.
    :return: a list of shuffling jobs to do.
    """
    # TODO(jchia): Dynamically decide on what backends to use for S3 and MetadataDatabase instead of hardcoding here
    backend = DatasetWriterBackend(S3PathFactory(), S3DataStorage(), DynamodbMetadataDatabase())
    job_factory = ShuffleJobFactory(backend, worker_max_working_set_size=worker_max_working_set_size)
    return list(job_factory.build_shuffle_jobs(DatasetID.from_str(dataset_id)))


@flytekit.task(requests=flytekit.Resources(mem="8Gi", cpu="2"), retries=4, cache=True, cache_version="v1")
def run_shuffling_job(job: ShuffleJob) -> ShuffleWorkerResults:
    """Run one shuffling job
    :param job: the ShuffleJob for this worker to run
    :return: pyarrow table containing only metadata and pointers to the ColumnBytesFiles in S3 for
        bytes columns in the dataset.
    """
    worker = ShuffleWorker(storage=S3DataStorage(), s3_path_factory=S3PathFactory())
    return ShuffleWorkerResults(
        pa_table=worker.process_job(job),
        partition=job.dataset_partition.partition,
    )


@flytekit.task(requests=flytekit.Resources(mem="8Gi", cpu="2"), retries=2)
def finalize_shuffling_jobs(dataset_id: str, shuffle_results: List[ShuffleWorkerResults]) -> Dict[str, int]:
    """Aggregate the indexes from the various shuffling jobs and publish the final parquet files for the dataset.
    :param dataset_id: string representation of the DatasetID we need to process (dataset name + version).
    :param shuffled_jobs_files: list of files containing the pandas Dataframe generated by the shuffling jobs.
    :return: A dictionary mapping partition_name -> size_of_partition.
    """
    results_by_partition = collections.defaultdict(list)
    for result in shuffle_results:
        results_by_partition[result.partition].append(result.pa_table)

    tables_by_partition: Dict[str, pa.Table] = {}
    for partition in results_by_partition:
        tables_by_partition[partition] = pa.concat_tables(results_by_partition[partition])

    dataset_id_obj = DatasetID.from_str(dataset_id)
    save_index(
        dataset_id_obj.name,
        dataset_id_obj.version,
        tables_by_partition,
        s3_storage=S3DataStorage(),
        s3_path_factory=S3PathFactory(),
    )
    return {partition_name: len(tables_by_partition[partition_name]) for partition_name in tables_by_partition}


@flytekit.workflow  # type: ignore
def WickerDataShufflingWorkflow(
    dataset_id: str,
    schema_json_str: str,
    worker_max_working_set_size: int = 16384,
) -> Dict[str, int]:
    """Pipeline finalizing a wicker dataset.
    :param dataset_id: string representation of the DatasetID we need to process (dataset name + version).
    :param schema_json_str: string representation of the schema, serialized as JSON
    :return: A dictionary mapping partition_name -> size_of_partition.
    """
    schema_json_str_committed = initialize_dataset(
        dataset_id=dataset_id,
        schema_json_str=schema_json_str,
    )
    jobs = create_shuffling_jobs(
        schema_json_str=schema_json_str_committed,
        dataset_id=dataset_id,
        worker_max_working_set_size=worker_max_working_set_size,
    )
    shuffle_results = flytekit.map_task(run_shuffling_job, metadata=flytekit.TaskMetadata(retries=1))(job=jobs)
    result = cast(Dict[str, int], finalize_shuffling_jobs(dataset_id=dataset_id, shuffle_results=shuffle_results))
    return result
