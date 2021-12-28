import pickle
import tempfile
import unittest
import unittest.mock
from typing import IO, Dict, List, Tuple

from wicker.core.column_files import ColumnBytesFileLocationV1
from wicker.core.config import get_config
from wicker.core.definitions import DatasetDefinition, DatasetID, DatasetPartition
from wicker.core.shuffle import ShuffleJob, ShuffleJobFactory, ShuffleWorker
from wicker.core.writer import MetadataDatabaseScanRow
from wicker.schema import dataparsing, schema, serialization
from wicker.testing.codecs import Vector, VectorCodec
from wicker.testing.storage import FakeS3DataStorage

DATASET_NAME = "dataset1"
DATASET_VERSION = "0.0.1"
FAKE_DATASET_SCHEMA = schema.DatasetSchema(
    fields=[
        schema.IntField("timestamp"),
        schema.StringField("car_id"),
        schema.ObjectField("vector", VectorCodec(0)),
    ],
    primary_keys=["car_id", "timestamp"],
)
FAKE_EXAMPLE = {
    "timestamp": 1,
    "car_id": "somecar",
    "vector": Vector([0, 0, 0]),
}
FAKE_DATASET_ID = DatasetID(name=DATASET_NAME, version=DATASET_VERSION)
FAKE_DATASET_DEFINITION = DatasetDefinition(
    dataset_id=FAKE_DATASET_ID,
    schema=FAKE_DATASET_SCHEMA,
)


class TestShuffleJobFactory(unittest.TestCase):
    def test_shuffle_job_factory_no_entries(self) -> None:
        """Tests the functionality of a shuffle_job_factory when provided with a mock backend"""
        mock_rows: List[MetadataDatabaseScanRow] = []
        mock_dataset_writer_backend = unittest.mock.MagicMock()
        mock_dataset_writer_backend._metadata_db.scan_sorted.return_value = (row for row in mock_rows)
        job_factory = ShuffleJobFactory(mock_dataset_writer_backend)
        self.assertEqual(list(job_factory.build_shuffle_jobs(FAKE_DATASET_ID)), [])

    def test_shuffle_job_factory_one_entry(self) -> None:
        """Tests the functionality of a shuffle_job_factory when provided with a mock backend"""
        mock_rows: List[MetadataDatabaseScanRow] = [
            MetadataDatabaseScanRow(partition="train", row_data_path="somepath", row_size=1337)
        ]
        mock_dataset_writer_backend = unittest.mock.MagicMock()
        mock_dataset_writer_backend._metadata_db.scan_sorted.return_value = (row for row in mock_rows)
        job_factory = ShuffleJobFactory(mock_dataset_writer_backend)
        self.assertEqual(
            list(job_factory.build_shuffle_jobs(FAKE_DATASET_ID)),
            [
                ShuffleJob(
                    dataset_partition=DatasetPartition(
                        dataset_id=FAKE_DATASET_ID,
                        partition="train",
                    ),
                    files=[("somepath", 1337)],
                )
            ],
        )

    def test_shuffle_job_factory_one_partition(self) -> None:
        """Tests the functionality of a shuffle_job_factory when provided with a mock backend"""
        partition = "train"
        mock_rows: List[MetadataDatabaseScanRow] = [
            MetadataDatabaseScanRow(partition=partition, row_data_path=f"somepath{i}", row_size=i) for i in range(10)
        ]
        mock_dataset_writer_backend = unittest.mock.MagicMock()
        mock_dataset_writer_backend._metadata_db.scan_sorted.return_value = (row for row in mock_rows)
        job_factory = ShuffleJobFactory(mock_dataset_writer_backend)
        self.assertEqual(
            list(job_factory.build_shuffle_jobs(FAKE_DATASET_ID)),
            [
                ShuffleJob(
                    dataset_partition=DatasetPartition(
                        dataset_id=FAKE_DATASET_ID,
                        partition=partition,
                    ),
                    files=[(row.row_data_path, row.row_size) for row in mock_rows],
                )
            ],
        )

    def test_shuffle_job_factory_one_partition_two_working_sets(self) -> None:
        """Tests the functionality of a shuffle_job_factory when provided with a mock backend"""
        partition = "train"
        worker_max_working_set_size = 5
        mock_rows: List[MetadataDatabaseScanRow] = [
            MetadataDatabaseScanRow(partition=partition, row_data_path=f"somepath{i}", row_size=i) for i in range(10)
        ]
        mock_dataset_writer_backend = unittest.mock.MagicMock()
        mock_dataset_writer_backend._metadata_db.scan_sorted.return_value = (row for row in mock_rows)
        job_factory = ShuffleJobFactory(
            mock_dataset_writer_backend, worker_max_working_set_size=worker_max_working_set_size
        )
        self.assertEqual(
            list(job_factory.build_shuffle_jobs(FAKE_DATASET_ID)),
            [
                ShuffleJob(
                    dataset_partition=DatasetPartition(
                        dataset_id=FAKE_DATASET_ID,
                        partition=partition,
                    ),
                    files=[(row.row_data_path, row.row_size) for row in mock_rows[:worker_max_working_set_size]],
                ),
                ShuffleJob(
                    dataset_partition=DatasetPartition(
                        dataset_id=FAKE_DATASET_ID,
                        partition=partition,
                    ),
                    files=[(row.row_data_path, row.row_size) for row in mock_rows[worker_max_working_set_size:]],
                ),
            ],
        )

    def test_shuffle_job_factory_multi_partition_multi_working_sets(self) -> None:
        """Tests the functionality of a shuffle_job_factory when provided with a mock backend"""
        partitions = ("train", "test")
        num_rows = 10
        num_batches = 2
        worker_max_working_set_size = num_rows // num_batches
        mock_rows: List[MetadataDatabaseScanRow] = [
            MetadataDatabaseScanRow(partition=partition, row_data_path=f"somepath{i}", row_size=i)
            for partition in partitions
            for i in range(10)
        ]
        mock_dataset_writer_backend = unittest.mock.MagicMock()
        mock_dataset_writer_backend._metadata_db.scan_sorted.return_value = (row for row in mock_rows)
        job_factory = ShuffleJobFactory(
            mock_dataset_writer_backend, worker_max_working_set_size=worker_max_working_set_size
        )
        self.assertEqual(
            list(job_factory.build_shuffle_jobs(FAKE_DATASET_ID)),
            [
                ShuffleJob(
                    dataset_partition=DatasetPartition(
                        dataset_id=FAKE_DATASET_ID,
                        partition=partition,
                    ),
                    files=[
                        (row.row_data_path, row.row_size)
                        for row in mock_rows[
                            worker_max_working_set_size
                            * (batch + (partition_index * num_batches)) : worker_max_working_set_size
                            * (batch + (partition_index * num_batches) + 1)
                        ]
                    ],
                )
                for partition_index, partition in enumerate(partitions)
                for batch in range(2)
            ],
        )


class TestShuffleWorker(unittest.TestCase):
    def setUp(self) -> None:
        self.boto3_patcher = unittest.mock.patch("wicker.core.shuffle.boto3")
        self.boto3_mock = self.boto3_patcher.start()
        self.uploaded_column_bytes_files: Dict[Tuple[str, str], bytes] = {}

    def tearDown(self) -> None:
        self.boto3_patcher.stop()
        self.uploaded_column_bytes_files.clear()

    @staticmethod
    def download_fileobj_mock(bucket: str, key: str, bio: IO) -> None:
        bio.write(
            pickle.dumps(
                dataparsing.parse_example(
                    FAKE_EXAMPLE,
                    FAKE_DATASET_SCHEMA,
                )
            )
        )
        bio.seek(0)
        return None

    def test_process_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # The threaded workers each construct their own S3DataStorage from a boto3 client to download
            # file data in parallel.  We mock those out here by mocking out the boto3 client itself.
            self.boto3_mock.session.Session().client().download_fileobj.side_effect = self.download_fileobj_mock

            fake_job = ShuffleJob(
                dataset_partition=DatasetPartition(
                    dataset_id=FAKE_DATASET_ID,
                    partition="test",
                ),
                files=[(f"s3://somebucket/path/{i}", i) for i in range(10)],
            )

            fake_storage = FakeS3DataStorage(tmpdir=tmpdir)
            fake_storage.put_object_s3(
                serialization.dumps(FAKE_DATASET_SCHEMA).encode("utf-8"),
                f"{get_config().aws_s3_config.s3_datasets_path}/{FAKE_DATASET_ID.name}"
                f"/{FAKE_DATASET_ID.version}/avro_schema.json",
            )
            worker = ShuffleWorker(storage=fake_storage)
            shuffle_results = worker.process_job(fake_job)

            self.assertEqual(shuffle_results["timestamp"].to_pylist(), [FAKE_EXAMPLE["timestamp"] for _ in range(10)])
            self.assertEqual(shuffle_results["car_id"].to_pylist(), [FAKE_EXAMPLE["car_id"] for _ in range(10)])
            for location_bytes in shuffle_results["vector"].to_pylist():
                location = ColumnBytesFileLocationV1.from_bytes(location_bytes)
                path = worker.s3_path_factory.get_column_concatenated_bytes_s3path_from_uuid(location.file_id.bytes)
                self.assertTrue(fake_storage.check_exists_s3(path))
                data = fake_storage.fetch_obj_s3(path)[location.byte_offset : location.byte_offset + location.data_size]
                self.assertEqual(VectorCodec(0).decode_object(data), FAKE_EXAMPLE["vector"])
