"""
Gcloud unfortunately has no moto equivalent. This is the location where
we create a few mocked classes similar to moto to ensure we are at the
least calling functions correctly.

It is important to remember that while we can mock the exact calls we are
making we cannot mock the underlying behavior 1-1 reducing the use of these
tests dramatically.
"""
import logging
import os
from pathlib import Path


class MockedGCSBlob:
    """
    Mocked blob object for abstracting operations on single file similar to
    gcloud storage Blob. Keep updated or make updates for api consistency.
    """
    def __init__(
        self,
        bucket: "MockedGCSBucket",
        name: str
    ) -> None:
        self.__bucket = bucket
        self.__name = name


    def exists(
        self,
        client: "MockedGCSClient"
    ) -> bool:
        return client.check_file_exists(
            bucket = self.__bucket.bucket_name,
            file_path = self.__name
        )
        


class MockedGCSBucket:
    """
    Mocked bucket object that abstracts comms to client similar to gcloud storage bucket.
    Update to keep and/or make the apis consistent.
    """

    def __init__(
        self,
        bucket_name: str
    ):
        self.__bucket_name = bucket_name

    def write_file(
        self,
        client: "MockedGCSClient",
        data: bytes,
        file_path: str
    ):
        client.write_data(
            bucket=self.__bucket_name,
            data=data,
            full_path=file_path
        )

    @property
    def bucket_name(self):
        return self.__bucket_name


class MockedGCSClient:
    """
    Mocked client object, manages the mocked file tree.
    """

    def __init__(
        self,
        mock_file_root: str
    ):
        self.__mock_file_root = mock_file_root
        self.__bucket_path_registry = {}

    def check_file_exists(
        self,
        bucket: str,
        file_path: str
    ) -> bool:
        if bucket not in self.__bucket_path_registry:
            logging.warning("Bucket does not exist.")
            return False

        full_file_path = os.path.join(self.__bucket_path_registry[bucket], file_path)
        return os.path.exists(full_file_path)

    def create_bucket(
        self,
        mock_bucket_name: str
    ) -> None:
        if mock_bucket_name in self.__bucket_path_registry: 
            logging.warning("Mocked bucket already exists, returning")
            return
        
        mocked_fs_path = os.path.join(self.__mock_file_root, mock_bucket_name)
        os.makedirs(mocked_fs_path, exist_ok=True)
        self.__bucket_path_registry[mock_bucket_name] = mocked_fs_path
        logging.info(f"Created mocked bucket at {mocked_fs_path}")

    def write_data(
        self,
        bucket: str,
        data: bytes,
        full_path: str
    ) -> None:
        if not bucket in self.__bucket_path_registry:
            logging.warning("Bucket not created yet, run create bucket. Cannot create file.")
            return
        
        full_path_on_mock = os.path.join(self.__bucket_path_registry[bucket], full_path)
        if os.path.exists(full_path_on_mock):
            logging.warning("File path already on system, overwriting")
            os.remove(full_path_on_mock)

        mocked_dir = Path(full_path_on_mock).parent
        os.makedirs(mocked_dir, exist_ok=True)
        with open(full_path_on_mock, "wb") as open_file:
            open_file.write(data)

