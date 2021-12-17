import os
from typing import Dict

from wicker.core.storage import S3DataStorage


class FakeS3DataStorage(S3DataStorage):
    def __init__(self) -> None:
        self.files: Dict[str, bytes] = {}
        self.clear_stats()

    def clear_stats(self) -> None:
        self.num_read_requests = 0
        self.num_write_requests = 0
        self.bytes_read = 0
        self.bytes_written = 0

    def check_exists_s3(self, input_path: str) -> bool:
        self.num_read_requests += 1
        return input_path in self.files

    def fetch_obj_s3(self, input_path: str) -> bytes:
        if input_path not in self.files:
            raise KeyError(f"File {input_path} not found in the fake s3 storage.")
        return self.files[input_path]

    def fetch_file_s3(self, input_path: str, local_prefix: str, timeout_seconds: int = 120) -> str:
        if input_path not in self.files:
            raise KeyError(f"File {input_path} not found in the fake s3 storage.")
        bucket, key = self.bucket_key_from_s3_path(input_path)
        local_path = os.path.join(local_prefix, key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.isfile(local_path):
            with open(local_path, "wb") as f:
                f.write(self.files[input_path])
            self.bytes_read += len(self.files[input_path])
            self.num_read_requests += 1
        return local_path

    def fetch_partial_file_s3(
        self, input_path: str, local_prefix: str, offset: int, size: int, timeout_seconds: int = 120
    ) -> str:
        if input_path not in self.files:
            raise KeyError(f"File {input_path} not found in the fake s3 storage.")
        bucket, key = self.bucket_key_from_s3_path(input_path)
        local_path = os.path.join(local_prefix, key) + f".{offset}_{size}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        assert offset + size <= len(self.files[input_path])
        if not os.path.isfile(local_path):
            with open(local_path, "wb") as f:
                f.write(self.files[input_path][offset : offset + size])
            self.bytes_read += size
            self.num_read_requests += 1
        return local_path

    def put_object_s3(self, object_bytes: bytes, s3_path: str) -> None:
        self.num_write_requests += 1
        self.bytes_written += len(object_bytes)
        self.files[s3_path] = object_bytes

    def put_file_s3(self, local_path: str, s3_path: str) -> None:
        self.num_write_requests += 1
        with open(local_path, "rb") as f:
            self.files[s3_path] = f.read()
            self.bytes_written += len(self.files[s3_path])
