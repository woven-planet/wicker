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
from typing import IO, Any, Dict, List, Literal, Optional, Tuple, Type

from wicker.core.storage import S3DataStorage, S3PathFactory
from wicker.schema import schema, validation


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


class ColumnBytesFileCache:
    """A read-through caching abstraction for accessing ColumnBytesFiles"""

    def __init__(
        self,
        local_cache_path_prefix: str = "/tmp",
        filelock_timeout_seconds: int = -1,
        path_factory: S3PathFactory = S3PathFactory(),
        storage: Optional[S3DataStorage] = None,
    ):
        """Initializes a ColumnBytesFileCache

        :param local_cache_path_prefix: root to path on disk to use as a disk cache, defaults to "/tmp"
        :type local_cache_path_prefix: str, optional
        :param filelock_timeout_seconds: number of seconds after which to timeout on waiting for downloads,
            defaults to -1 which indicates to wait forever
        :type filelock_timeout_seconds: int, optional
        """
        self._s3_storage = storage if storage is not None else S3DataStorage()
        self._root_path = local_cache_path_prefix
        self._filelock_timeout_seconds = filelock_timeout_seconds
        self._columns_root_path = path_factory.get_column_concatenated_bytes_files_path()

    def read(
        self,
        cbf_info: ColumnBytesFileLocationV1,
    ) -> bytes:
        column_concatenated_bytes_file_path = os.path.join(self._columns_root_path, str(cbf_info.file_id))

        local_path = self._s3_storage.fetch_file_s3(
            column_concatenated_bytes_file_path,
            self._root_path,
            timeout_seconds=self._filelock_timeout_seconds,
        )

        with open(local_path, "rb") as f:
            f.seek(cbf_info.byte_offset)
            return f.read(cbf_info.data_size)

    def resolve_pointers(
        self,
        example: validation.AvroRecord,
        schema: schema.DatasetSchema,
    ) -> validation.AvroRecord:
        visitor = ResolvePointersVisitor(example, schema, self)
        return visitor.resolve_pointers()


class ResolvePointersVisitor(schema.DatasetSchemaVisitor[Any]):
    """schema.DatasetSchemaVisitor class that will resolve all heavy pointers in an example by
    downloading the appropriate ColumnBytesFile files and retrieving the appropriate bytes
    """

    def __init__(
        self,
        example: validation.AvroRecord,
        schema: schema.DatasetSchema,
        cbf_cache: ColumnBytesFileCache,
    ):
        self.cbf_cache = cbf_cache

        # Pointers to original data (data should be kept immutable)
        self._schema = schema
        self._example = example

        # Pointers to help keep visitor state during tree traversal
        self._current_data: Any = self._example
        self._current_path: Tuple[str, ...] = tuple()

    def resolve_pointers(self) -> Dict[str, Any]:
        """Resolve all heavy pointers"""
        # Since the original input example is non-None, the loaded example will be non-None also
        example: Dict[str, Any] = self._schema.schema_record._accept_visitor(self)
        return example

    def process_record_field(self, field: schema.RecordField) -> Optional[validation.AvroRecord]:
        """Visit an schema.RecordField schema field"""
        current_data = validation.validate_dict(self._current_data, field.required, self._current_path)
        if current_data is None:
            return current_data

        # Process nested fields by setting up the visitor's state and visiting each node
        processing_path = self._current_path
        processing_example = current_data
        loaded = {}

        # When reading records, the client might restrict the columns to load to a subset of the
        # full columns, so check if the key is actually present in the example being processed
        for nested_field in field.fields:
            if nested_field.name in processing_example:
                self._current_path = processing_path + (nested_field.name,)
                self._current_data = processing_example[nested_field.name]
                loaded[nested_field.name] = nested_field._accept_visitor(self)
        return loaded

    def process_array_field(self, field: schema.ArrayField) -> Optional[List[Any]]:
        current_data = validation.validate_field_type(self._current_data, list, field.required, self._current_path)
        if current_data is None:
            return current_data

        # Process array elements by setting up the visitor's state and visiting each element
        processing_path = self._current_path
        loaded = []

        # Arrays may contain None values if the element field declares that it is not required
        for element_index, element in enumerate(current_data):
            self._current_path = processing_path + (f"elem[{element_index}]",)
            self._current_data = element
            loaded.append(field.element_field._accept_visitor(self))
        return loaded

    def process_int_field(self, field: schema.IntField) -> Any:
        return self._current_data

    def process_long_field(self, field: schema.LongField) -> Any:
        return self._current_data

    def process_string_field(self, field: schema.StringField) -> Any:
        return self._current_data

    def process_bool_field(self, field: schema.BoolField) -> Any:
        return self._current_data

    def process_float_field(self, field: schema.FloatField) -> Any:
        return self._current_data

    def process_double_field(self, field: schema.DoubleField) -> Any:
        return self._current_data

    def process_object_field(self, field: schema.ObjectField) -> Any:
        if not field.is_heavy_pointer:
            return self._current_data
        data = validation.validate_field_type(self._current_data, bytes, field.required, self._current_path)
        if data is None:
            return data
        cbf_info = ColumnBytesFileLocationV1.from_bytes(data)
        return self.cbf_cache.read(cbf_info)
