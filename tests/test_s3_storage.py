import os
import tempfile
from typing import Any, Dict
from unittest import TestCase, mock

from botocore.exceptions import ClientError  # type: ignore
from botocore.stub import Stubber  # type: ignore

from wicker.core.config import WickerConfig
from wicker.core.storage import S3DataStorage, S3PathFactory


class TestS3DataStorage(TestCase):
    def test_bucket_key_from_s3_path(self) -> None:
        """Unit test for the S3DataStorage bucket_key_from_s3_path function"""
        data_storage = S3DataStorage()

        s3_url = "s3://hello/world"
        bucket, key = data_storage.bucket_key_from_s3_path(s3_url)
        self.assertEqual(bucket, "hello")
        self.assertEqual(key, "world")

        s3_url = "s3://hello/"
        bucket, key = data_storage.bucket_key_from_s3_path(s3_url)
        self.assertEqual(bucket, "hello")
        self.assertEqual(key, "")

        s3_url = "s3://"
        bucket, key = data_storage.bucket_key_from_s3_path(s3_url)
        self.assertEqual(bucket, "")
        self.assertEqual(key, "")

        s3_url = "s3://hello/world/"
        bucket, key = data_storage.bucket_key_from_s3_path(s3_url)
        self.assertEqual(bucket, "hello")
        self.assertEqual(key, "world/")

    def test_check_exists_s3(self) -> None:
        """Unit test for the check_exists_s3 function."""
        data_storage = S3DataStorage()
        input_path = "s3://foo/bar/baz/dummy"

        with Stubber(data_storage.client) as stubber:
            response = {}  # type: ignore
            expected_params = {"Bucket": "foo", "Key": "bar/baz/dummy"}
            stubber.add_response("head_object", response, expected_params)
            self.assertTrue(data_storage.check_exists_s3(input_path))

    def test_check_exists_s3_nonexisting(self) -> None:
        """Unit test for the check_exists_s3 function."""
        data_storage = S3DataStorage()
        input_path = "s3://foo/bar/baz/dummy"

        with Stubber(data_storage.client) as stubber:
            stubber.add_client_error(
                expected_params={"Bucket": "foo", "Key": "bar/baz/dummy"},
                method="head_object",
                service_error_code="404",
                service_message="The specified key does not exist.",
            )

            # The check_exists_s3 function catches the exception when the key does not exist
            self.assertFalse(data_storage.check_exists_s3(input_path))

    def test_put_object_s3(self) -> None:
        """Unit test for the put_object_s3 function."""
        data_storage = S3DataStorage()
        object_bytes = b"this is my object"
        input_path = "s3://foo/bar/baz/dummy"

        with Stubber(data_storage.client) as stubber:
            response = {}  # type: ignore
            expected_params = {
                "Body": object_bytes,
                "Bucket": "foo",
                "Key": "bar/baz/dummy",
            }
            stubber.add_response("put_object", response, expected_params)
            data_storage.put_object_s3(object_bytes, input_path)

    def test_put_file_s3(self) -> None:
        """Unit test for the put_file_s3 function"""
        data_storage = S3DataStorage()
        object_bytes = b"this is my object"
        input_path = "s3://foo/bar/baz/dummy"

        with tempfile.NamedTemporaryFile() as tmpfile:
            tmpfile.write(object_bytes)
            tmpfile.flush()

            with Stubber(data_storage.client) as stubber:
                response = {}  # type: ignore
                stubber.add_response("put_object", response, None)
                data_storage.put_file_s3(tmpfile.name, input_path)

    @staticmethod
    def download_file_side_effect(*args, **kwargs) -> None:  # type: ignore
        """Helper function to patch the S3 download_file function with a side-effect that creates an
        empty file at the correct path in order to mock the download"""
        input_path = str(kwargs["filename"])
        with open(input_path, "w"):
            pass

    # Stubber does not have a stub function for S3 client download_file function, so patch it
    @mock.patch("boto3.s3.transfer.S3Transfer.download_file")
    def test_fetch_file_s3(self, download_file: mock.Mock) -> None:
        """Unit test for the fetch_file_s3 function."""
        data_storage = S3DataStorage()
        input_path = "s3://foo/bar/baz/dummy"
        with tempfile.TemporaryDirectory() as local_prefix:
            # Add a side-effect to create the file to download at the correct local path
            download_file.side_effect = self.download_file_side_effect

            local_path = data_storage.fetch_file_s3(input_path, local_prefix)
            download_file.assert_called_once_with(
                bucket="foo",
                key="bar/baz/dummy",
                filename=f"{local_prefix}/bar/baz/dummy",
                extra_args=None,
                callback=None,
            )
            self.assertTrue(os.path.isfile(local_path))

    # Stubber does not have a stub function for S3 client download_file function, so patch it
    @mock.patch("boto3.s3.transfer.S3Transfer.download_file")
    def test_fetch_file_s3_on_nonexistent_file(self, download_file: mock.Mock) -> None:
        """Unit test for the fetch_file_s3 function for a non-existent file in S3."""
        data_storage = S3DataStorage()
        input_path = "s3://foo/bar/barbazz/dummy"
        local_prefix = "/tmp"

        response = {"Error": {"Code": "404"}}
        side_effect = ClientError(response, "unexpected")
        download_file.side_effect = side_effect

        with self.assertRaises(ClientError):
            data_storage.fetch_file_s3(input_path, local_prefix)


class TestS3PathFactory(TestCase):
    @mock.patch("wicker.core.storage.get_config")
    def test_get_column_concatenated_bytes_files_path(self, mock_get_config: mock.Mock) -> None:
        """Unit test for the S3PathFactory get_column_concatenated_bytes_files_path
        function"""
        # If store_concatenated_bytes_files_in_dataset is False, return the default path
        dummy_config: Dict[str, Any] = {
            "aws_s3_config": {
                "s3_datasets_path": "s3://dummy_bucket/wicker/",
                "region": "us-east-1",
                "boto_config": {"max_pool_connections": 10, "read_timeout_s": 140, "connect_timeout_s": 140},
            },
            "dynamodb_config": {"table_name": "fake-table-name", "region": "us-west-2"},
            "storage_download_config": {
                "retries": 2,
                "timeout": 150,
                "retry_backoff": 5,
                "retry_delay_s": 4,
            },
        }
        mock_get_config.return_value = WickerConfig.from_json(dummy_config)

        path_factory = S3PathFactory()
        self.assertEqual(
            path_factory.get_column_concatenated_bytes_files_path(),
            "s3://dummy_bucket/wicker/__COLUMN_CONCATENATED_FILES__",
        )

        # If store_concatenated_bytes_files_in_dataset is True, return the dataset-specific path
        dummy_config["aws_s3_config"]["store_concatenated_bytes_files_in_dataset"] = True
        mock_get_config.return_value = WickerConfig.from_json(dummy_config)
        dataset_name = "dummy_dataset"
        self.assertEqual(
            S3PathFactory().get_column_concatenated_bytes_files_path(dataset_name=dataset_name),
            f"s3://dummy_bucket/wicker/{dataset_name}/__COLUMN_CONCATENATED_FILES__",
        )

        # If the store_concatenated_bytes_files_in_dataset option is True but no
        # dataset_name, raise ValueError
        with self.assertRaises(ValueError):
            S3PathFactory().get_column_concatenated_bytes_files_path()
