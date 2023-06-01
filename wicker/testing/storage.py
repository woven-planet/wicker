import os
import shutil
from pathlib import Path
from typing import Any, Dict

import pyarrow.fs as pafs

from wicker.core.storage import S3DataStorage


class FakeS3DataStorage(S3DataStorage):
    def __init__(self, tmpdir: str = "/tmp") -> None:
        self._tmpdir = tmpdir

    def __getstate__(self) -> Dict[Any, Any]:
        return {"tmpdir": self._tmpdir}

    def __setstate__(self, state: Dict[Any, Any]) -> None:
        self._tmpdir = state["tmpdir"]
        return

    def _get_local_path(self, path: str) -> str:
        return os.path.join(self._tmpdir, path.replace("s3://", ""))

    def check_exists_s3(self, input_path: str) -> bool:
        return os.path.exists(self._get_local_path(input_path))

    def fetch_obj_s3(self, input_path: str) -> bytes:
        if not self.check_exists_s3(input_path):
            raise KeyError(f"File {input_path} not found in the fake s3 storage.")
        with open(self._get_local_path(input_path), "rb") as f:
            return f.read()

    def fetch_file_s3(self, input_path: str, local_prefix: str, timeout_seconds: int = 120) -> str:
        if not self.check_exists_s3(input_path):
            raise KeyError(f"File {input_path} not found in the fake s3 storage.")
        bucket, key = self.bucket_key_from_s3_path(input_path)
        dest_path = os.path.join(local_prefix, key)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if not os.path.isfile(dest_path):
            shutil.copy2(self._get_local_path(input_path), dest_path)
        return dest_path

    def put_object_s3(self, object_bytes: bytes, s3_path: str) -> None:
        full_tmp_path = self._get_local_path(s3_path)
        os.makedirs(os.path.dirname(full_tmp_path), exist_ok=True)
        with open(full_tmp_path, "wb") as f:
            f.write(object_bytes)

    def put_file_s3(self, local_path: str, s3_path: str) -> None:
        full_tmp_path = self._get_local_path(s3_path)
        os.makedirs(os.path.dirname(full_tmp_path), exist_ok=True)
        shutil.copy2(local_path, full_tmp_path)


class LocalDataStorage(S3DataStorage):
    def __init__(self, root_path: str):
        super().__init__()
        self._root_path = Path(root_path)
        self._fs = pafs.LocalFileSystem()

    @property
    def filesystem(self) -> pafs.FileSystem:
        return self._fs

    def _create_path(self, path: str) -> None:
        """Ensures the given path exists."""
        self._fs.create_dir(path, recursive=True)

    # Override.
    def check_exists_s3(self, input_path: str) -> bool:
        file_info = self._fs.get_file_info(input_path)
        return file_info.type != pafs.FileType.NotFound

    # Override.
    def fetch_file_s3(self, input_path: str, local_prefix: str, timeout_seconds: int = 120) -> str:
        # This raises if the input path is not relative to the root.
        relative_input_path = Path(input_path).relative_to(self._root_path)

        target_path = os.path.join(local_prefix, str(relative_input_path))
        self._create_path(os.path.dirname(target_path))
        self._fs.copy_file(input_path, target_path)
        return target_path

    # Override.
    def fetch_partial_file_s3(
        self, input_path: str, local_prefix: str, offset: int, size: int, timeout_seconds: int = 120
    ) -> str:
        raise NotImplementedError("fetch_partial_file_s3")

    # Override.
    def put_object_s3(self, object_bytes: bytes, s3_path: str) -> None:
        self._create_path(os.path.dirname(s3_path))
        with self._fs.open_output_stream(s3_path) as ostream:
            ostream.write(object_bytes)

    # Override.
    def put_file_s3(self, local_path: str, s3_path: str) -> None:
        self._create_path(os.path.dirname(s3_path))
        self._fs.copy_file(local_path, s3_path)
