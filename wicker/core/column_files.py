"""This module contains the code required to interact with column files

column files are simple files that are just a simple continuous stream of bytes. They cannot
be read without the corresponding index that is stored in a separate metadata store, which contains the bytes
offsets to access the individual rows within the file.
"""

from __future__ import annotations

import dataclasses
import os
import struct
import tempfile
import uuid
from types import TracebackType
from typing import IO, Dict, Literal, Optional, Tuple, Type

from wicker.core.storage import S3DataStorage, S3PathFactory


@dataclasses.dataclass(order=True)
class ColumnBytesFileLocationV1:
    """Location serialized as:

    1byte protocol version | 16bytes file ID | 8bytes byte offset | 4bytes size of data in bytes
    """

    file_id: uuid.UUID
    byte_offset: int
    data_size: int

    PROTOCOL_VERSION = 1

    # Packed as (all little-endian):
    #   char(1)
    #   char[](16)
    #   unsigned_long_long(8)
    #   unsigned_int(8)
    STRUCT_PACK_FMT = "<c16sQI"

    def to_bytes(self) -> bytes:
        return struct.pack(
            ColumnBytesFileLocationV1.STRUCT_PACK_FMT,
            ColumnBytesFileLocationV1.PROTOCOL_VERSION.to_bytes(1, "little"),
            self.file_id.bytes,
            self.byte_offset,
            self.data_size,
        )

    @classmethod
    def from_bytes(cls, b: bytes) -> ColumnBytesFileLocationV1:
        protocol_version = int.from_bytes(b[0:1], "little")
        if protocol_version != 1:
            raise ValueError(f"Unable to parse ColumnBytesFileLocation with protocol_version={protocol_version}")
        _, file_id, byte_offset, data_size = struct.unpack(ColumnBytesFileLocationV1.STRUCT_PACK_FMT, b)
        return cls(
            file_id=uuid.UUID(bytes=file_id),
            byte_offset=byte_offset,
            data_size=data_size,
        )


@dataclasses.dataclass
class ColumnBytesFileWriteBuffer:
    file_id: uuid.UUID
    write_count: int
    buffer: IO[bytes]

    def append_to_buffer(self, data: bytes) -> Tuple[int, int]:
        """Appends data to the buffer, returning the start and size of the data

        :param data: data to append
        :return: (start, size) of the data in the buffer
        """
        start_byte_offset = self.buffer.tell()
        bytes_written = self.buffer.write(data)
        self.write_count += 1
        return start_byte_offset, bytes_written


class ColumnBytesFileWriter:
    """A class that writes columns into ColumnBytesFile files, with one file per column"""

    def __init__(
        self,
        storage: S3DataStorage = S3DataStorage(),
        s3_path_factory: S3PathFactory = S3PathFactory(),
        target_file_size: Optional[int] = None,
        target_file_rowgroup_size: Optional[int] = None,
    ) -> None:
        """Create a ColumnBytesFileWriter

        :param storage: storage client to use, defaults to S3DataStorage()
        :param s3_path_factory: path factory to use, defaults to S3PathFactory()
        :param target_file_size: If set, open a new binary file when a column file gets larger than this many bytes.
        :param target_file_rowgroup_size: If set, open a new binary file when a column file has more rows than this
        """
        self.storage = storage
        self.s3_path_factory = s3_path_factory
        # {column_name: (file_id, write_count, <filehandle_to_tmp_file>)}
        self.write_buffers: Dict[str, ColumnBytesFileWriteBuffer] = {}
        self.target_file_size = target_file_size
        self.target_file_rowgroup_size = target_file_rowgroup_size

    def __enter__(self) -> ColumnBytesFileWriter:
        return self

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> Literal[False]:
        self.close()
        return False

    def add(self, column_name: str, data: bytes) -> ColumnBytesFileLocationV1:
        """Adds some data into the write buffer

        :param column_name: name of the column to write to
        :param data: data to write into column
        :return: info about location in the binary file
        """
        # Initialize if first element to be added
        if column_name not in self.write_buffers:
            self.write_buffers[column_name] = self._get_new_buffer()

        # Update the io.BytesIO buffer
        write_buffer = self.write_buffers[column_name]
        start_byte_offset, bytes_written = write_buffer.append_to_buffer(data)
        file_size_reached_limit = (
            self.target_file_size is not None and start_byte_offset + bytes_written >= self.target_file_size
        )
        file_num_rows_reached_limit = (
            self.target_file_rowgroup_size is not None and write_buffer.write_count >= self.target_file_rowgroup_size
        )
        if file_size_reached_limit or file_num_rows_reached_limit:
            self._write_column(column_name)

        return ColumnBytesFileLocationV1(
            file_id=write_buffer.file_id,
            byte_offset=start_byte_offset,
            data_size=bytes_written,
        )

    def close(self) -> None:
        """Writes the data in the buffer to S3"""
        for column_name in self.write_buffers:
            self._write_column(column_name)

    def _get_new_buffer(self) -> ColumnBytesFileWriteBuffer:
        return ColumnBytesFileWriteBuffer(
            file_id=uuid.uuid4(),
            write_count=0,
            buffer=tempfile.NamedTemporaryFile(),
        )

    def _write_column(self, column_name: str) -> None:
        columns_root_path = self.s3_path_factory.get_column_concatenated_bytes_files_path()
        write_buffer = self.write_buffers[column_name]
        path = os.path.join(columns_root_path, str(write_buffer.file_id))
        if write_buffer.buffer.tell() > 0:
            write_buffer.buffer.flush()
            write_buffer.buffer.seek(0)
            self.storage.put_file_s3(
                write_buffer.buffer.name,
                path,
            )
        # Get a new UUID so that the next chunk of this column does not overwrite the previous chunks.
        write_buffer.buffer.close()
        self.write_buffers[column_name] = self._get_new_buffer()
