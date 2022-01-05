import os
import shutil
from typing import Any, Dict

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
