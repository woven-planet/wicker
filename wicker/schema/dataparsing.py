from typing import Any, Dict, List, Optional, Tuple

from wicker.core.definitions import ExampleMetadata
from wicker.core.errors import WickerSchemaException
from wicker.schema import schema, validation


def parse_example(example: Dict[str, Any], schema: schema.DatasetSchema) -> validation.AvroRecord:
    """Parses an example according to the provided schema, converting known types such as
    numpy arrays, torch tensors etc into bytes for storage on disk

    The returned dictionary will be in the same shape as defined in the schema, with keys
    corresponding to the schema names, and values validated against the schema's fields.

    :param example: example to parse
    :type example: Dict[str, Any]
    :param schema: schema to parse against
    :type schema: schema.DatasetSchema
    :return: parsed example
    :rtype: Dict[str, Any]
    """
    parse_example_visitor = ParseExampleVisitor(example, schema)
    return parse_example_visitor.parse_example()


def parse_example_metadata(example: Dict[str, Any], schema: schema.DatasetSchema) -> ExampleMetadata:
    """Parses ExampleMetadata from an example according to the provided schema

    The returned dictionary will be in the same shape as defined in the schema, with keys
    corresponding to the schema names, and values validated against the schema's fields.

    Certain field types will be ignored when parsing examples as they are not considered part of the example metadata:

        1. BytesField
        ... More to be implemented (e.g. AvroNumpyField etc)

    :param example: example to parse
    :type example: Dict[str, Any]
    :param schema: schema to parse against
    :type schema: schema.DatasetSchema
    """
    parse_example_metadata_visitor = ParseExampleMetadataVisitor(example, schema)
    return parse_example_metadata_visitor.parse_example()


# A special exception to indicate that a field should be skipped
class _SkipFieldException(Exception):
    pass


class ParseExampleVisitor(schema.DatasetSchemaVisitor[Any]):
    """schema.DatasetSchemaVisitor class that will validate and transform an Example
    in accordance to a provided schema.DatasetSchema

    The original example is not modified in-place, and we incur the cost of copying
    primitives (e.g. bytes, strings)
    """

    def __init__(self, example: Dict[str, Any], schema: schema.DatasetSchema):
        # Pointers to original data (data should be kept immutable)
        self._schema = schema
        self._example = example

        # Pointers to help keep visitor state during tree traversal
        self._current_data: Any = self._example
        self._current_path: Tuple[str, ...] = tuple()

    def parse_example(self) -> Dict[str, Any]:
        """Parses an example into a form that is suitable for storage in an Avro format"""
        # Since the original input example is non-None, the parsed example will be non-None also
        example: Dict[str, Any] = self._schema.schema_record._accept_visitor(self)
        return example

    def process_record_field(self, field: schema.RecordField) -> Optional[validation.AvroRecord]:
        """Visit an schema.RecordField schema field"""
        val = validation.validate_field_type(self._current_data, dict, field.required, self._current_path)
        if val is None:
            return val

        # Add keys to the example for any non-required fields that were left unset in the raw data
        for optional_field in [f for f in field.fields if not f.required]:
            if optional_field.name not in val:
                val[optional_field.name] = None

        # Validate that data matches schema exactly
        schema_key_names = {nested_field.name for nested_field in field.fields}
        if val.keys() != schema_key_names:
            raise WickerSchemaException(
                f"Error at path {'.'.join(self._current_path)}: "
                f"Example missing keys: {list(schema_key_names - val.keys())} "
                f"and has extra keys: {list(val.keys() - schema_key_names)}"
            )

        # Process nested fields by setting up the visitor's state and visiting each node
        res = {}
        processing_path = self._current_path
        processing_example = self._current_data

        for nested_field in field.fields:
            self._current_path = processing_path + (nested_field.name,)
            self._current_data = processing_example[nested_field.name]
            try:
                res[nested_field.name] = nested_field._accept_visitor(self)
            except _SkipFieldException:
                pass
        return res

    def process_int_field(self, field: schema.IntField) -> Optional[int]:
        return validation.validate_field_type(self._current_data, int, field.required, self._current_path)

    def process_long_field(self, field: schema.LongField) -> Optional[int]:
        return validation.validate_field_type(self._current_data, int, field.required, self._current_path)

    def process_string_field(self, field: schema.StringField) -> Optional[str]:
        return validation.validate_field_type(self._current_data, str, field.required, self._current_path)

    def process_bool_field(self, field: schema.BoolField) -> Optional[bool]:
        return validation.validate_field_type(self._current_data, bool, field.required, self._current_path)

    def process_float_field(self, field: schema.FloatField) -> Optional[float]:
        return validation.validate_field_type(self._current_data, float, field.required, self._current_path)

    def process_double_field(self, field: schema.DoubleField) -> Optional[float]:
        return validation.validate_field_type(self._current_data, float, field.required, self._current_path)

    def process_object_field(self, field: schema.ObjectField) -> Optional[bytes]:
        data = validation.validate_field_type(
            self._current_data, field.codec.object_type(), field.required, self._current_path
        )
        if data is None:
            return None
        return field.codec.validate_and_encode_object(data)

    def process_array_field(self, field: schema.ArrayField) -> Optional[List[Any]]:
        val = validation.validate_field_type(self._current_data, list, field.required, self._current_path)
        if val is None:
            return val

        # Process array elements by setting up the visitor's state and visiting each element
        res = []
        processing_path = self._current_path
        processing_example = self._current_data

        # Arrays may contain None values if the element field declares that it is not required
        for element_index, element in enumerate(processing_example):
            self._current_path = processing_path + (f"elem[{element_index}]",)
            self._current_data = element
            # Allow _SkipFieldExceptions to propagate up and skip this array field
            res.append(field.element_field._accept_visitor(self))
        return res


class ParseExampleMetadataVisitor(ParseExampleVisitor):
    """Specialization of ParseExampleVisitor which skips over certain fields that are now parsed as metadata"""

    def process_object_field(self, field: schema.ObjectField) -> Optional[bytes]:
        # Raise a special error to indicate that this field should be skipped
        raise _SkipFieldException()
