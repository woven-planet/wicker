from typing import Any, Dict, List, Optional, Tuple, TypeVar

from wicker.schema import schema, validation

T = TypeVar("T")


def load_example(example: validation.AvroRecord, schema: schema.DatasetSchema) -> Dict[str, Any]:
    """Loads an example according to the provided schema, converting data based on their column
    information into their corresponding in-memory representations (e.g. numpy arrays, torch tensors)

    The returned dictionary will be in the same shape as defined in the schema, with keys
    corresponding to the schema names, and values validated/transformed against the schema's fields.

    :param example: example to load
    :type example: validation.AvroRecord
    :param schema: schema to load against
    :type schema: schema.DatasetSchema
    :return: loaded example
    :rtype: Dict[str, Any]
    """
    load_example_visitor = LoadExampleVisitor(example, schema)
    return load_example_visitor.load_example()


class LoadExampleVisitor(schema.DatasetSchemaVisitor[Any]):
    """schema.DatasetSchemaVisitor class that will validate and load an Example
    in accordance to a provided schema.DatasetSchema
    """

    def __init__(self, example: validation.AvroRecord, schema: schema.DatasetSchema):
        # Pointers to original data (data should be kept immutable)
        self._schema = schema
        self._example = example

        # Pointers to help keep visitor state during tree traversal
        self._current_data: Any = self._example
        self._current_path: Tuple[str, ...] = tuple()

    def load_example(self) -> Dict[str, Any]:
        """Loads an example from its Avro format into its in-memory representations"""
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

    def process_object_field(self, field: schema.ObjectField) -> Optional[Any]:
        data = validation.validate_field_type(self._current_data, bytes, field.required, self._current_path)
        if data is None:
            return data
        return field.codec.decode_object(data)

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
