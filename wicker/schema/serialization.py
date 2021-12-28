import json
import re
from typing import Any, Dict, Type, Union

from wicker.core.errors import WickerSchemaException
from wicker.schema import codecs, schema
from wicker.schema.schema import PRIMARY_KEYS_TAG

JSON_SCHEMA_VERSION = 2
DATASET_ID_REGEX = re.compile(r"^(?P<name>[0-9a-zA-Z_]+)/v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$")


def dumps(schema: schema.DatasetSchema, pretty: bool = True) -> str:
    """Dumps a schema as JSON

    :param schema: schema to dump
    :type schema: schema.DatasetSchema
    :param pretty: whether to dump the schema as a prettified JSON, defaults to False
    :type pretty: bool, optional
    :return: JSON string
    :rtype: str
    """
    visitor = AvroDatasetSchemaSerializer()
    jdata = schema.schema_record._accept_visitor(visitor)
    jdata.update(jdata["type"])
    jdata["_json_version"] = JSON_SCHEMA_VERSION
    if pretty:
        return json.dumps(jdata, sort_keys=True, indent=4)
    return json.dumps(jdata)


def loads(schema_str: str, treat_objects_as_bytes: bool = False) -> schema.DatasetSchema:
    """Loads a DatasetSchema from a JSON str

    :param schema_str: JSON string
    :param treat_objects_as_bytes: If set, don't load codecs for object types, instead just treat them as bytes.
      This is useful for code that needs to work with datasets in a generic way, but does not need to actually decode
      the data.
    :return: Deserialized DatasetSchema object
    """
    # Parse string as JSON
    try:
        schema_dict = json.loads(schema_str)
    except json.decoder.JSONDecodeError:
        raise WickerSchemaException(f"Unable load string as JSON: {schema_str}")

    # Construct the DatasetSchema
    try:
        fields = [_loads(d, treat_objects_as_bytes=treat_objects_as_bytes) for d in schema_dict["fields"]]
        return schema.DatasetSchema(
            fields=fields,
            primary_keys=json.loads(schema_dict.get(PRIMARY_KEYS_TAG, "[]")),
            allow_empty_primary_keys=True,  # For backward compatibility. Clean me AVSW-78939.
        )
    except KeyError:
        raise WickerSchemaException(f"Malformed serialization of DatasetSchema: {schema_str}")


def _loads(schema_dict: Dict[str, Any], treat_objects_as_bytes: bool) -> schema.SchemaField:
    """Recursively parses a schema dictionary into the appropriate SchemaField"""
    type_ = schema_dict["type"]
    required = True

    # Nullable types are represented in Avro schemas as Union (list) types with "null" as the
    # first element (by convention)
    if isinstance(type_, list):
        assert type_[0] == "null"
        required = False
        type_ = type_[1]

    if isinstance(type_, dict) and type_["type"] == "record":
        return schema.RecordField(
            name=schema_dict["name"],
            fields=[_loads(f, treat_objects_as_bytes=treat_objects_as_bytes) for f in type_["fields"]],
            description=schema_dict["_description"],
            required=required,
        )
    if isinstance(type_, dict) and type_["type"] == "array":
        # ArrayField columns are limited to contain only dicts or simple types (and not nested arrays)
        element_dict = type_.copy()
        element_dict["type"] = type_["items"]
        element_field = _loads(element_dict, treat_objects_as_bytes=treat_objects_as_bytes)
        return schema.ArrayField(
            element_field=element_field,
            required=required,
        )
    return _loads_base_types(type_, required, schema_dict, treat_objects_as_bytes=treat_objects_as_bytes)


def _loads_base_types(
    type_: str, required: bool, schema_dict: Dict[str, Any], treat_objects_as_bytes: bool = False
) -> schema.SchemaField:
    if type_ == "int":
        return schema.IntField(
            name=schema_dict["name"],
            description=schema_dict["_description"],
            required=required,
        )
    elif type_ == "long":
        return schema.LongField(
            name=schema_dict["name"],
            description=schema_dict["_description"],
            required=required,
        )
    elif type_ == "string":
        return schema.StringField(
            name=schema_dict["name"],
            description=schema_dict["_description"],
            required=required,
        )
    elif type_ == "boolean":
        return schema.BoolField(
            name=schema_dict["name"],
            description=schema_dict["_description"],
            required=required,
        )
    elif type_ == "float":
        return schema.FloatField(
            name=schema_dict["name"],
            description=schema_dict["_description"],
            required=required,
        )
    elif type_ == "double":
        return schema.DoubleField(
            name=schema_dict["name"],
            description=schema_dict["_description"],
            required=required,
        )
    elif type_ == "bytes":
        l5ml_metatype = schema_dict["_l5ml_metatype"]
        if l5ml_metatype == "object":
            codec_name = schema_dict["_codec_name"]
            if treat_objects_as_bytes:
                codec: codecs.Codec = _PassThroughObjectCodec(codec_name, json.loads(schema_dict["_codec_params"]))
            else:
                try:
                    codec_cls: Type[codecs.Codec] = codecs.Codec.codec_registry[codec_name]
                except KeyError:
                    raise WickerSchemaException(
                        f"Could not find a registered codec with name {codec_name} "
                        f"for field {schema_dict['name']}. Please define a subclass of ObjectField.Codec and define "
                        "the codec_name static method."
                    )
                codec = codec_cls.load_codec_from_dict(json.loads(schema_dict["_codec_params"]))
            return schema.ObjectField(
                name=schema_dict["name"],
                codec=codec,
                description=schema_dict["_description"],
                required=required,
                is_heavy_pointer=schema_dict.get("_is_heavy_pointer", False),
            )
        raise WickerSchemaException(f"Unhandled _l5ml_metatype for avro bytes type: {l5ml_metatype}")
    raise WickerSchemaException(f"Unhandled type: {type_}")


class _PassThroughObjectCodec(codecs.Codec):
    """The _PassThroughObjectCodec class is a placeholder for any object codec, it is used when we need to parse
    any possible schema properly, but we don't need to read the data. This codec does not decode/encode the data,
    instead is just acts as an identity function. However, it keeps track of all the attributes of the original codec
    as they were stored in the loaded schema, so that if we save that schema again, we do not lose any information.
    """

    def __init__(self, codec_name: str, codec_attributes: Dict[str, Any]):
        self._codec_name_value = codec_name
        self._codec_attributes = codec_attributes

    @staticmethod
    def _codec_name() -> str:
        # The static method does not make any sense here, because this class is a placeholder for any codec class.
        # Instead we will overload the get_codec_name function.
        return ""

    def get_codec_name(self) -> str:
        return self._codec_name_value

    def save_codec_to_dict(self) -> Dict[str, Any]:
        return self._codec_attributes

    @staticmethod
    def load_codec_from_dict(data: Dict[str, Any]) -> codecs.Codec:
        pass

    def validate_and_encode_object(self, obj: bytes) -> bytes:
        return obj

    def decode_object(self, data: bytes) -> bytes:
        return data

    def object_type(self) -> Type[Any]:
        return bytes


class AvroDatasetSchemaSerializer(schema.DatasetSchemaVisitor[Dict[str, Any]]):
    """A visitor class that serializes an AvroDatasetSchema as Avro-compatible JSON"""

    def process_schema_field(self, field: schema.SchemaField, avro_type: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Common processing across all field types"""
        # The way to declare nullable fields in Avro schemas is to declare Avro Union (list) types.
        # The default type (usually null) should be listed first:
        # https://avro.apache.org/docs/current/spec.html#Unions
        field_type = avro_type if field.required else ["null", avro_type]
        return {
            "name": field.name,
            "type": field_type,
            "_description": field.description,
            **field.custom_field_tags,
        }

    def process_record_field(self, field: schema.RecordField) -> Dict[str, Any]:
        record_type = {
            "type": "record",
            "fields": [nested_field._accept_visitor(self) for nested_field in field.fields],
            "name": field.name,
        }
        return self.process_schema_field(field, record_type)

    def process_int_field(self, field: schema.IntField) -> Dict[str, Any]:
        return {
            **self.process_schema_field(field, "int"),
        }

    def process_long_field(self, field: schema.LongField) -> Dict[str, Any]:
        return {
            **self.process_schema_field(field, "long"),
        }

    def process_string_field(self, field: schema.StringField) -> Dict[str, Any]:
        return {
            **self.process_schema_field(field, "string"),
        }

    def process_bool_field(self, field: schema.BoolField) -> Dict[str, Any]:
        return {
            **self.process_schema_field(field, "boolean"),
        }

    def process_float_field(self, field: schema.FloatField) -> Dict[str, Any]:
        return {
            **self.process_schema_field(field, "float"),
        }

    def process_double_field(self, field: schema.DoubleField) -> Dict[str, Any]:
        return {
            **self.process_schema_field(field, "double"),
        }

    def process_object_field(self, field: schema.ObjectField) -> Dict[str, Any]:
        return {
            **self.process_schema_field(field, "bytes"),
            "_l5ml_metatype": "object",
            "_is_heavy_pointer": field.is_heavy_pointer,
        }

    def process_array_field(self, field: schema.ArrayField) -> Dict[str, Any]:
        array_type = field.element_field._accept_visitor(self)
        array_type["items"] = array_type["type"]
        array_type["type"] = "array"

        field_type = array_type if field.required else ["null", array_type]
        return {
            "name": field.name,
            "type": field_type,
        }
