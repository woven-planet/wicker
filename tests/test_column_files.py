import os
import tempfile
import unittest
import uuid
from unittest.mock import MagicMock

from wicker.core.column_files import (
    ColumnBytesFileCache,
    ColumnBytesFileLocationV1,
    ColumnBytesFileReader,
    ColumnBytesFileWriter,
)
from wicker.core.definitions import DatasetID, DatasetPartition
from wicker.core.storage import S3PathFactory, WickerPathFactory
from wicker.testing.storage import FakeS3DataStorage

FAKE_SHARD_ID = "fake_shard_id"
FAKE_DATA_PARTITION = DatasetPartition(dataset_id=DatasetID(name="name", version="0.0.1"), partition="partition")
FAKE_BYTES = b"foo"
FAKE_BYTES2 = b"0123456789"
FAKE_COL = "col0"
FAKE_COL2 = "col1"


class TestColumnBytesFileWriter(unittest.TestCase):
    def test_write_empty(self) -> None:
        mock_storage = MagicMock()
        path_factory = S3PathFactory()
        with ColumnBytesFileWriter(
            storage=mock_storage,
            s3_path_factory=path_factory,
        ):
            pass
        mock_storage.put_object_s3.assert_not_called()

    def test_write_one_column_one_row(self) -> None:
        path_factory = S3PathFactory()
        mock_storage = MagicMock()
        with ColumnBytesFileWriter(
            storage=mock_storage,
            s3_path_factory=path_factory,
        ) as ccb:
            info = ccb.add(FAKE_COL, FAKE_BYTES)
            self.assertEqual(info.byte_offset, 0)
            self.assertEqual(info.data_size, len(FAKE_BYTES))
        mock_storage.put_file_s3.assert_called_once_with(
            unittest.mock.ANY,
            os.path.join(
                path_factory.get_column_concatenated_bytes_files_path(),
                str(info.file_id),
            ),
        )

    def test_write_one_column_multi_row(self) -> None:
        path_factory = S3PathFactory()
        mock_storage = MagicMock()
        with ColumnBytesFileWriter(
            storage=mock_storage,
            s3_path_factory=path_factory,
        ) as ccb:
            info = ccb.add(FAKE_COL, FAKE_BYTES)
            self.assertEqual(info.byte_offset, 0)
            self.assertEqual(info.data_size, len(FAKE_BYTES))

            next_info = ccb.add(FAKE_COL, FAKE_BYTES)
            self.assertEqual(next_info.byte_offset, len(FAKE_BYTES))
            self.assertEqual(next_info.data_size, len(FAKE_BYTES))
        mock_storage.put_file_s3.assert_called_once_with(
            unittest.mock.ANY,
            os.path.join(
                path_factory.get_column_concatenated_bytes_files_path(),
                str(info.file_id),
            ),
        )

    def test_write_multi_column_multi_row(self) -> None:
        path_factory = S3PathFactory()
        mock_storage = MagicMock()
        with ColumnBytesFileWriter(
            storage=mock_storage,
            s3_path_factory=path_factory,
        ) as ccb:
            info1 = ccb.add(FAKE_COL, FAKE_BYTES)
            self.assertEqual(info1.byte_offset, 0)
            self.assertEqual(info1.data_size, len(FAKE_BYTES))

            info1 = ccb.add(FAKE_COL, FAKE_BYTES)
            self.assertEqual(info1.byte_offset, len(FAKE_BYTES))
            self.assertEqual(info1.data_size, len(FAKE_BYTES))

            info2 = ccb.add(FAKE_COL2, FAKE_BYTES)
            self.assertEqual(info2.byte_offset, 0)
            self.assertEqual(info2.data_size, len(FAKE_BYTES))

            info2 = ccb.add(FAKE_COL2, FAKE_BYTES)
            self.assertEqual(info2.byte_offset, len(FAKE_BYTES))
            self.assertEqual(info2.data_size, len(FAKE_BYTES))

            info2 = ccb.add(FAKE_COL2, FAKE_BYTES)
            self.assertEqual(info2.byte_offset, len(FAKE_BYTES) * 2)
            self.assertEqual(info2.data_size, len(FAKE_BYTES))

        mock_storage.put_file_s3.assert_any_call(
            unittest.mock.ANY,
            os.path.join(
                path_factory.get_column_concatenated_bytes_files_path(),
                str(info1.file_id),
            ),
        )
        mock_storage.put_file_s3.assert_any_call(
            unittest.mock.ANY,
            os.path.join(
                path_factory.get_column_concatenated_bytes_files_path(),
                str(info2.file_id),
            ),
        )

    def test_write_large_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_factory = S3PathFactory()
            storage = FakeS3DataStorage(tmpdir=tmpdir)
            with ColumnBytesFileWriter(storage=storage, s3_path_factory=path_factory, target_file_size=10) as ccb:
                info1 = ccb.add(FAKE_COL, FAKE_BYTES)
                self.assertEqual(info1.byte_offset, 0)
                self.assertEqual(info1.data_size, len(FAKE_BYTES))
                info1 = ccb.add(FAKE_COL, FAKE_BYTES)
                self.assertEqual(info1.byte_offset, len(FAKE_BYTES))
                self.assertEqual(info1.data_size, len(FAKE_BYTES))
                info1 = ccb.add(FAKE_COL, FAKE_BYTES)
                self.assertEqual(info1.byte_offset, len(FAKE_BYTES) * 2)
                self.assertEqual(info1.data_size, len(FAKE_BYTES))
                info1 = ccb.add(FAKE_COL, FAKE_BYTES)
                self.assertEqual(info1.byte_offset, len(FAKE_BYTES) * 3)
                self.assertEqual(info1.data_size, len(FAKE_BYTES))
                info2 = ccb.add(FAKE_COL, FAKE_BYTES)
                self.assertEqual(info2.byte_offset, 0)
                self.assertEqual(info2.data_size, len(FAKE_BYTES))
                self.assertNotEqual(info1.file_id, info2.file_id)

                info1 = ccb.add(FAKE_COL2, FAKE_BYTES2)
                self.assertEqual(info1.byte_offset, 0)
                self.assertEqual(info1.data_size, len(FAKE_BYTES2))
                info2 = ccb.add(FAKE_COL2, FAKE_BYTES2)
                self.assertEqual(info2.byte_offset, 0)
                self.assertEqual(info2.data_size, len(FAKE_BYTES2))
                self.assertNotEqual(info1.file_id, info2.file_id)
                info3 = ccb.add(FAKE_COL2, FAKE_BYTES2)
                self.assertEqual(info3.byte_offset, 0)
                self.assertEqual(info3.data_size, len(FAKE_BYTES2))
                self.assertNotEqual(info2.file_id, info3.file_id)

    def test_write_manyrows_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_factory = S3PathFactory()
            storage = FakeS3DataStorage(tmpdir=tmpdir)
            with ColumnBytesFileWriter(
                storage=storage, s3_path_factory=path_factory, target_file_rowgroup_size=1
            ) as ccb:
                info1 = ccb.add(FAKE_COL, FAKE_BYTES)
                self.assertEqual(info1.byte_offset, 0)
                self.assertEqual(info1.data_size, len(FAKE_BYTES))
                info2 = ccb.add(FAKE_COL, FAKE_BYTES)
                self.assertEqual(info2.byte_offset, 0)
                self.assertEqual(info2.data_size, len(FAKE_BYTES))
                self.assertNotEqual(info1.file_id, info2.file_id)

                info1 = ccb.add(FAKE_COL2, FAKE_BYTES2)
                self.assertEqual(info1.byte_offset, 0)
                self.assertEqual(info1.data_size, len(FAKE_BYTES2))
                info2 = ccb.add(FAKE_COL2, FAKE_BYTES2)
                self.assertEqual(info2.byte_offset, 0)
                self.assertEqual(info2.data_size, len(FAKE_BYTES2))
                self.assertNotEqual(info1.file_id, info2.file_id)
                info3 = ccb.add(FAKE_COL2, FAKE_BYTES2)
                self.assertEqual(info3.byte_offset, 0)
                self.assertEqual(info3.data_size, len(FAKE_BYTES2))
                self.assertNotEqual(info2.file_id, info3.file_id)


class TestColumnBytesFileCacheAndReader(unittest.TestCase):
    def test_write_one_column_one_row_and_read(self) -> None:
        path_factory = S3PathFactory()
        mock_storage = MagicMock()
        # First, write a row to column bytes mock storage.
        with ColumnBytesFileWriter(
            storage=mock_storage,
            s3_path_factory=path_factory,
        ) as ccb:
            info = ccb.add(FAKE_COL, FAKE_BYTES)
            self.assertEqual(info.byte_offset, 0)
            self.assertEqual(info.data_size, len(FAKE_BYTES))
        s3_path = os.path.join(
            path_factory.get_column_concatenated_bytes_files_path(),
            str(info.file_id),
        )
        mock_storage.put_file_s3.assert_called_once_with(unittest.mock.ANY, s3_path)

        # Now, verify that we can read it back from mock storage.
        local_path = os.path.join("/tmp", s3_path.split("s3://fake_data/")[1])
        mock_storage.fetch_file = MagicMock(return_value=local_path)

        cbf_cache = ColumnBytesFileCache(
            storage=mock_storage,
            path_factory=path_factory,
        )

        # Mock the helper function that just opens a file and returns its contents.
        read_column_bytes_file_mock = MagicMock(return_value=FAKE_BYTES)
        cbf_cache._read_column_bytes_file = read_column_bytes_file_mock  # type: ignore

        # Read back the column file.
        bytes_read = cbf_cache.read(info)
        mock_storage.fetch_file.assert_called_once_with(s3_path, "/tmp", timeout_seconds=-1)
        self.assertEqual(len(bytes_read), len(FAKE_BYTES))
        self.assertEqual(bytes_read, FAKE_BYTES)

        # Now let's verify that we can use the reader directly to read the local column bytes.
        wicker_path_factory = WickerPathFactory("/tmp")
        cbf_reader = ColumnBytesFileReader(wicker_path_factory)

        # Reset the reader function mock so we can check that it was called just once.
        read_column_bytes_file_mock = MagicMock(return_value=FAKE_BYTES)
        cbf_reader._read_column_bytes_file = read_column_bytes_file_mock  # type: ignore

        bytes_read = cbf_reader.read(info)
        read_column_bytes_file_mock.assert_called_once_with(
            column_bytes_file_info=info,
            column_bytes_file_path=local_path,
        )
        self.assertEqual(len(bytes_read), len(FAKE_BYTES))
        self.assertEqual(bytes_read, FAKE_BYTES)


class TestCCBInfo(unittest.TestCase):
    def test_to_string(self) -> None:
        ccb_info = ColumnBytesFileLocationV1(uuid.uuid4(), 100, 100)
        ccb_info_parsed = ColumnBytesFileLocationV1.from_bytes(ccb_info.to_bytes())
        self.assertEqual(ccb_info, ccb_info_parsed)
