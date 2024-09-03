import abc
import logging
import os
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import boto3
import pyarrow  # type: ignore
import pyarrow.fs as pafs  # type: ignore
import pyarrow.parquet as papq  # type: ignore

from wicker.core.column_files import (
    ColumnBytesFileCache,
    ColumnBytesFileLocationV1,
    ColumnBytesFileReader,
)
from wicker.core.config import get_config  # type: ignore
from wicker.core.definitions import DatasetDefinition, DatasetID, DatasetPartition
from wicker.core.multi_cloud.gcloud.gcs import (
    generate_manifest_file,
    get_non_existant_s3_file_set,
    launch_gcs_transfer_job,
    push_manifest_to_gcp,
)
from wicker.core.parsing import list_combine, multiproc_file_parse, thread_file_parse
from wicker.core.storage import (
    AbstractDataStorage,
    FileSystemDataStorage,
    S3DataStorage,
    S3PathFactory,
    WickerPathFactory,
)
from wicker.schema import dataloading, serialization
from wicker.schema.schema import DatasetSchema

# How long to wait before timing out on filelocks in seconds
FILE_LOCK_TIMEOUT_SECONDS = 300

logger = logging.getLogger(__name__)


def thread_func_head_size(buckets_keys_chunks_local: List[Tuple[str, str]]):
    return thread_file_parse(buckets_keys_chunks_local, iterate_bucket_key_chunk_for_size, sum)


def thread_func_non_existant_gcloud(buckets_keys_chunks_local: List[Tuple[str, str]]):
    return thread_file_parse(buckets_keys_chunks_local, get_non_existant_s3_file_set, list_combine)


def iterate_bucket_key_chunk_for_size(bucket_key_locs: List[Tuple[str, str]]) -> int:  # type: ignore
    """Iterate on chunk of s3 files to get local length of bytes.

    Args:
        bucket_key_locs: List of Tuple[str, str] containing the s3 bucket and key to check size.

    Returns:
        int: total amount of bytes in the file chunk list

    """
    local_len = 0

    # create the s3 resource locally and don't pass in. Boto3 docs state to do this in each thread
    # and not pass around.
    s3_resource = boto3.resource("s3")
    idx = 0
    for bucket_key_loc in bucket_key_locs:
        bucket_loc, key_loc = bucket_key_loc
        # get the byte length for the object
        byte_length = s3_resource.Object(bucket_loc, key_loc).content_length
        local_len += byte_length
        idx += 1
        if idx == 10:
            break
    return local_len


class AbstractDataset(abc.ABC):
    """Interface for a Map-style (non-streaming) Dataset"""

    def __init__(
        self,
        dataset_name: str,
        dataset_partition_name: str,
        dataset_version: str,
        path_factory: WickerPathFactory,
        pa_filesystem: pafs.FileSystem,
        storage: AbstractDataStorage,
        columns_to_load: Optional[List[str]] = None,
        filelock_timeout_seconds: int = FILE_LOCK_TIMEOUT_SECONDS,
        local_cache_path_prefix: Optional[str] = os.getenv("TMPDIR", "/tmp"),
        treat_objects_as_bytes: bool = False,
    ) -> None:
        """Init an AbstractDataset object.

        :param dataset_name: name of the dataset
        :param dataset_partition_name: partition name
        :param dataset_version: version of the dataset
        :param path_factory: WickerPathFactory for pulling consistent paths.
        :param pa_filesystem: Pyarrow filesystem for reading the parquet files and tables.
        :param storage: AbstractDataStorage for data access.
        :param columns_to_load: list of columns to load, defaults to None which loads all columns
        :param filelock_timeout_seconds: number of seconds after which to timeout on waiting for downloads,
            defaults to FILE_LOCK_TIMEOUT_SECONDS
        :param local_cache_path_prefix: Path to local cache path, if None don't create cache
        :param treat_objects_as_bytes: If set, don't try to decode ObjectFields and keep them as binary data.
        """
        super().__init__()
        self._arrow_table: Optional[pyarrow.Table] = None
        self._columns_to_load = columns_to_load
        self._filelock_timeout_seconds = filelock_timeout_seconds
        self._local_cache_path_prefix = local_cache_path_prefix
        self._pa_filesystem = pa_filesystem
        self._path_factory = path_factory
        self._storage = storage
        self._treat_objects_as_bytes = treat_objects_as_bytes
        self._column_bytes_file_reader = self.__create_column_bytes_file_reader(
            dataset_name=dataset_name,
            filelock_timeout_seconds=filelock_timeout_seconds,
            local_cache_path_prefix=local_cache_path_prefix,
            path_factory=path_factory,
            storage=storage,
        )

        self._dataset_id = DatasetID(name=dataset_name, version=dataset_version)
        self._dataset_definition = DatasetDefinition(
            self._dataset_id,
            schema=self.schema,
        )
        self._partition = DatasetPartition(dataset_id=self._dataset_id, partition=dataset_partition_name)

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Gets the ith Example from the given dataset/version/partition"""
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset/version/partition"""
        pass

    @abc.abstractmethod
    def arrow_table(self) -> pyarrow.Table:
        """Return the pyarrow table with all the metadata fields and pointers of the dataset."""
        pass

    @cached_property
    def heavy_pointer_files(self) -> List[str]:
        """Get list of heavy pointer files in the dataset"""
        schema = self.schema
        heavy_pointer_cols = []
        for col_name in schema.get_all_column_names():
            column = schema.get_column(col_name)
            if column is not None and column.is_heavy_pointer:
                heavy_pointer_cols.append(col_name)

        return heavy_pointer_cols

    @cached_property
    def schema(self) -> DatasetSchema:
        """Return the schema of the dataset."""
        schema_data = None
        schema_path = self._path_factory._get_dataset_schema_path(self._dataset_id)
        local_path = schema_path
        # only fetch the file locally if we have a cache prefix
        if self._local_cache_path_prefix is not None:
            local_path = self._storage.fetch_file(
                schema_path, self._local_cache_path_prefix, timeout_seconds=self._filelock_timeout_seconds
            )
        with open(local_path, "rb") as f:
            schema_data = serialization.loads(
                f.read().decode("utf-8"), treat_objects_as_bytes=self._treat_objects_as_bytes
            )
        return schema_data

    @staticmethod
    def __create_column_bytes_file_reader(
        dataset_name: Optional[str],
        filelock_timeout_seconds: int,
        local_cache_path_prefix: Optional[str],
        path_factory: WickerPathFactory,
        storage: AbstractDataStorage,
    ) -> Union[ColumnBytesFileCache, ColumnBytesFileReader]:
        """Create a column bytes file reader class.

        Creates an instance of a ColumnBytesFileReader or ColumnBytesFileCache
        depending on whether a local_cache_path_prefix is not none. Access
        pattern is the same between the two classes but one is a read through cache
        and the other is not, usage dependent on if you need your own cache or rely
        on bucket mount cache in GCSFuse or S3 Mount Point.

        Args:
            dataset_name (Optional[str]): Name of the dataset.
            filelock_timeout_seconds (int): Time that you wait for file lock.
            local_cache_path_prefix (Optional[str]): Optional path of the prefix, determines
                if you use a cache or not. If specified as None no cache is used.
            path_factory (WickerPathFactory): Path factory for locating column bytes files
                in dataset path.
            storage (AbstractDataStorage): Data storage to use for grabbing files.

        Returns:
            Union[ColumnBytesFileCache, ColumnBytesFileReader]: reader class for reading data
            with or without cache.
        """
        # if we have a cache prefix create a cache
        if local_cache_path_prefix is not None:
            logging.info(f"Cache passed at path - {local_cache_path_prefix}, creating read through cache on top.")
            # weird mypy problem, can't define baseclass and have it pick up the child correctly
            column_bytes_file_class: Union[ColumnBytesFileCache, ColumnBytesFileReader] = ColumnBytesFileCache(
                filelock_timeout_seconds=filelock_timeout_seconds,
                local_cache_path_prefix=local_cache_path_prefix,
                path_factory=path_factory,
                storage=storage,
            )
        else:
            logging.info("No cache passed, reading without caching.")
            column_bytes_file_class = ColumnBytesFileReader(dataset_name=dataset_name, path_factory=path_factory)
        return column_bytes_file_class


class FileSystemDataset(AbstractDataset):
    """Implementation of a Map-based dataset on local file system or mounted drive"""

    def __init__(
        self,
        dataset_name: str,
        dataset_partition_name: str,
        dataset_version: str,
        columns_to_load: Optional[List[str]] = None,
        filelock_timeout_seconds: int = FILE_LOCK_TIMEOUT_SECONDS,
        filesystem_root_path: Optional[str] = None,
        local_cache_path_prefix: Optional[str] = os.getenv("TMPDIR", "/tmp"),
        pa_filesystem: Optional[pafs.LocalFileSystem] = None,
        path_factory: Optional[WickerPathFactory] = None,
        storage: Optional[FileSystemDataStorage] = None,
        treat_objects_as_bytes: bool = False,
    ):
        """Initializes a FileSystemDataset.

        :param dataset_name: name of the dataset
        :param dataset_partition_name: partition name
        :param dataset_version: version of the dataset
        :param columns_to_load: list of columns to load, defaults to None which loads all columns
        :param filelock_timeout_seconds: number of seconds after which to timeout on waiting for downloads,
            defaults to FILE_LOCK_TIMEOUT_SECONDS
        :param filesystem_root_path: path to the root of the wicker file system. If path factory is none,
            this must be set.
        :param local_cache_path_prefix: Path to local cache path, if None don't create cache
        :param pa_filesystem: Pyarrow filesystem for reading the parquet files and tables.
        :param path_factory: Optional WickerPathFactory for pulling consistent paths.
        :param storage: Optional FileSystemDataStorage object for pulling files from filesystem
        :param treat_objects_as_bytes: If set, don't try to decode ObjectFields and keep them as binary data.
        """
        if path_factory is None and filesystem_root_path is None:
            raise ValueError("Need to pass either path factory of wicker dataset or root of the tree.")
        # ignore type failure here as we handle the case where they're both none above
        path_factory = (
            path_factory
            if path_factory is not None
            else WickerPathFactory(root_path=filesystem_root_path)  # type: ignore
        )
        pa_filesystem = pafs.LocalFileSystem() if pa_filesystem is None else pa_filesystem
        storage = storage if storage is not None else FileSystemDataStorage()

        super().__init__(
            columns_to_load=columns_to_load,
            dataset_name=dataset_name,
            dataset_partition_name=dataset_partition_name,
            dataset_version=dataset_version,
            filelock_timeout_seconds=filelock_timeout_seconds,
            local_cache_path_prefix=local_cache_path_prefix,
            path_factory=path_factory,
            pa_filesystem=pa_filesystem,
            storage=storage,
            treat_objects_as_bytes=treat_objects_as_bytes,
        )

    def arrow_table(self) -> pyarrow.Table:
        """Grab and load arrow table from expected path.

        Returns:
            pyarrow.Table: Arrow table object for the loaded dataset.
        """
        path = self._path_factory._get_dataset_partition_path(self._partition)
        if not self._arrow_table:
            self._arrow_table = papq.read_table(path, columns=self._columns_to_load, filesystem=self._pa_filesystem)
        return self._arrow_table

    def __len__(self) -> int:
        """Get length of arrow table inferring it is the same as the dataset.

        Returns:
            int: length of arrow table.
        """
        return len(self.arrow_table())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get data item at index within arrow table.

        Pulls from either cache or data store the item in the dataset at index specified.

        Args:
            idx (int): idx in arrow table to grab data.

        Returns:
            Dict[str, Any]: Row of data defined through schema object.
        """
        tbl = self.arrow_table()
        columns = self._columns_to_load if self._columns_to_load is not None else tbl.column_names
        row = {col: tbl[col][idx].as_py() for col in columns}
        return dataloading.load_example(
            self._column_bytes_file_reader.resolve_pointers(row, self.schema),
            self.schema,
        )


class S3Dataset(AbstractDataset):
    """Implementation for a Map-based dataset"""

    def __init__(
        self,
        dataset_name: str,
        dataset_version: str,
        dataset_partition_name: str,
        local_cache_path_prefix: str = os.getenv("TMPDIR", "/tmp"),
        columns_to_load: Optional[List[str]] = None,
        storage: Optional[S3DataStorage] = None,
        s3_path_factory: Optional[S3PathFactory] = None,
        pa_filesystem: Optional[pafs.FileSystem] = None,
        filelock_timeout_seconds: int = FILE_LOCK_TIMEOUT_SECONDS,
        treat_objects_as_bytes: bool = False,
        filters=None,
    ):
        """Initializes an S3Dataset

        :param dataset_name: name of the dataset
        :param dataset_version: version of the dataset
        :param dataset_partition_name: partition name
        :param columns_to_load: list of columns to load, defaults to None which loads all columns
        :param data_service: S3DataService instance to use, defaults to None which uses default initializations
        :param filelock_timeout_seconds: number of seconds after which to timeout on waiting for downloads,
            defaults to FILE_LOCK_TIMEOUT_SECONDS
        :param treat_objects_as_bytes: If set, don't try to decode ObjectFields and keep them as binary data
        :param filters: Only returns rows which match the filter. Defaults to None, i.e., returns all rows.
        :type filters: pyarrow.compute.Expression, List[Tuple], or List[List[Tuple]], optional
        .. seealso:: `filters in <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html>`__ # noqa
        """
        pa_filesystem = (
            pafs.S3FileSystem(region=get_config().aws_s3_config.region) if pa_filesystem is None else pa_filesystem
        )
        s3_path_factory = s3_path_factory if s3_path_factory is not None else S3PathFactory()
        storage = storage if storage is not None else S3DataStorage()

        super().__init__(
            columns_to_load=columns_to_load,
            dataset_name=dataset_name,
            dataset_partition_name=dataset_partition_name,
            dataset_version=dataset_version,
            filelock_timeout_seconds=filelock_timeout_seconds,
            local_cache_path_prefix=local_cache_path_prefix,
            path_factory=s3_path_factory,
            pa_filesystem=pa_filesystem,
            storage=storage,
            treat_objects_as_bytes=treat_objects_as_bytes,
        )
        self.filters = filters

    def arrow_table(self) -> pyarrow.Table:
        path = self._path_factory._get_dataset_partition_path(self._partition, prefix_to_trim="s3://")
        if not self._arrow_table:
            self._arrow_table = papq.read_table(
                path, columns=self._columns_to_load, filesystem=self._pa_filesystem, filters=self.filters
            )
        return self._arrow_table

    def __len__(self) -> int:
        return len(self.arrow_table())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tbl = self.arrow_table()
        columns = self._columns_to_load if self._columns_to_load is not None else tbl.column_names
        row = {col: tbl[col][idx].as_py() for col in columns}
        return dataloading.load_example(
            self._column_bytes_file_reader.resolve_pointers(row, self.schema),
            self.schema,
        )

    def copy_partition_to_gcloud(self) -> bool:
        # get the total set of col files in the ds
        heavy_pointer_buckets_keys = self.heavy_pointer_buckets_keys

        # get the total set that do not exist on gcloud
        # do this in case previous transfer failed and we pick up midway
        cut_down = list(heavy_pointer_buckets_keys)[:50]
        files_to_move = multiproc_file_parse(cut_down, thread_func_non_existant_gcloud, list_combine)

        # when you have the file list create the gcloud transfer service
        # manifest file
        manifest_file_local_path = "./manifest.csv"
        generate_manifest_file(files_to_move=files_to_move, manifest_dest_path=manifest_file_local_path)

        gcs_file_location_path = push_manifest_to_gcp(
            dataset_name=self._dataset_id.name,
            dataset_partition=self._partition.partition,
            dataset_version=self._dataset_id.version,
            manifest_file_local_path=manifest_file_local_path,
        )

        launch_code = launch_gcs_transfer_job(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            description="test job",
            manifest_location=gcs_file_location_path,
            project_id="wp-dev-eai-n321",
        )
        return launch_code

    @cached_property
    def heavy_pointer_buckets_keys(self) -> Set[Tuple[str, str]]:
        """Get the bucket key pairs for each heavy point file."""
        logging.info("Processing through heavy pointers")

        buckets_keys = set()
        arrow_table = self.arrow_table()
        for heavy_pntr_col in self.heavy_pointer_files[:5]:
            logging.info(f"Evaulating {heavy_pntr_col} for column file locations")
            # Each individual row only knows which column file it goes to, so we have to
            # neccesarily parse all rows :( to get the column files. This should be cached
            # as metadata but that would require re-curating the datasets.
            for location_bytes in arrow_table[heavy_pntr_col].to_pylist():
                location = ColumnBytesFileLocationV1.from_bytes(location_bytes)
                # type ignore because this is guaranteed to have a S3PathFactory as the
                # path factory attr
                path = self._path_factory.get_column_concatenated_bytes_s3path_from_uuid(  # type: ignore
                    location.file_id.bytes, dataset_name=self._dataset_id.name
                )
                bucket, key = path.replace("s3://", "").split("/", 1)
                buckets_keys.add((bucket, key))
        return buckets_keys

    def _get_parquet_dir_size(self) -> int:
        """Get the parquet path and find all the files within, count their bytes

        Returns:
            int: bytes in parquet directory
        """
        # bytes size of arrow table not bytes in arrow table
        # bytes in arrow table is a method of arrow table but it doesn't
        # reflect the size of the file sizes stored on s3 just the loaded data
        arrow_path = self._path_factory._get_dataset_partition_path(self._partition, prefix_to_trim="s3://")
        bucket, key = arrow_path.replace("s3://", "").split("/", 1)

        def get_folder_size(bucket, prefix):
            total_size = 0
            for obj in boto3.resource("s3").Bucket(bucket).objects.filter(Prefix=prefix):
                total_size += obj.size
            return total_size

        return get_folder_size(bucket, key)

    def _get_dataset_partition_size(self) -> int:
        """Gets total size of the dataset partition in bytes.

        Returns:
            int: total dataset partition size in bytes
        """

        # intialize with size of parquet dir
        logging.info("Parsing parquet and arrow dir for size.")
        par_dir_bytes = self._get_parquet_dir_size()

        buckets_keys_list = list(self.heavy_pointer_buckets_keys)
        column_files_byte_size = multiproc_file_parse(
            buckets_keys=buckets_keys_list,
            function_for_process=thread_func_head_size,
            result_collapse_func=list_combine,
        )
        return column_files_byte_size + par_dir_bytes

    @cached_property
    def dataset_size(self) -> int:
        """Total dataset partition size in bytes

        Returns:
            int: total dataset size in bytes
        """
        return self._get_dataset_partition_size()
