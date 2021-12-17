import unittest
from typing import List

from wicker.core.definitions import DatasetDefinition, DatasetID, DatasetPartition
from wicker.core.shuffle import ShuffleJob, ShuffleJobFactory
from wicker.core.writer import MetadataDatabaseScanRow
from wicker.schema import schema
from wicker.testing.codecs import VectorCodec

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
        self.assertEqual(list(job_factory.build_shuffle_jobs(FAKE_DATASET_DEFINITION)), [])

    def test_shuffle_job_factory_one_entry(self) -> None:
        """Tests the functionality of a shuffle_job_factory when provided with a mock backend"""
        mock_rows: List[MetadataDatabaseScanRow] = [
            MetadataDatabaseScanRow(partition="train", row_data_path="somepath", row_size=1337)
        ]
        mock_dataset_writer_backend = unittest.mock.MagicMock()
        mock_dataset_writer_backend._metadata_db.scan_sorted.return_value = (row for row in mock_rows)
        job_factory = ShuffleJobFactory(mock_dataset_writer_backend)
        self.assertEqual(
            list(job_factory.build_shuffle_jobs(FAKE_DATASET_DEFINITION)),
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
            list(job_factory.build_shuffle_jobs(FAKE_DATASET_DEFINITION)),
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
            list(job_factory.build_shuffle_jobs(FAKE_DATASET_DEFINITION)),
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
