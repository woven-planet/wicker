import copy
import json
import unittest
from typing import Any, Dict

import avro.schema  # type: ignore

from wicker import schema
from wicker.core.errors import WickerSchemaException
from wicker.schema import dataloading, dataparsing, serialization
from wicker.schema.schema import PRIMARY_KEYS_TAG
from wicker.testing.codecs import Vector, VectorCodec

TEST_SCHEMA = schema.DatasetSchema(
    fields=[
        schema.IntField("label", description="Label of the example"),
        schema.RecordField(
            "lidar_point_cloud",
            fields=[
                schema.RecordField(
                    "lidar_metadata",
                    fields=[
                        schema.StringField(
                            "lidar_model",
                            description="Model of lidar used to generate data",
                        ),
                        schema.DoubleField(
                            "lidar_calibration_error",
                            description="Some lidar calibration metric",
                            required=False,
                        ),
                    ],
                    description="Some metadata about the lidar",
                ),
            ],
            description="Lidar point cloud data",
        ),
        schema.LongField("timestamp_ns", description="Some timestamp field in ns"),
        schema.FloatField("ego_speed", description="Absolute speed of ego"),
        schema.BoolField("qc", description="A quality control field", required=False),
        schema.RecordField(
            "extra_metadata",
            description="Extra metadata",
            fields=[
                schema.IntField("meta_1", description="Metadata 1"),
                schema.IntField("meta_2", description="Metadata 2", required=False),
            ],
            required=False,
        ),
        schema.ArrayField(
            schema.StringField("array_stringfield", description="some array"),
        ),
    ],
    primary_keys=["timestamp_ns"],
)
TEST_EXAMPLE_REQUIRED: Dict[str, Any] = {
    "label": 0,
    "lidar_point_cloud": {
        "lidar_metadata": {
            "lidar_model": "harambe",
        },
    },
    "timestamp_ns": 1337,
    "ego_speed": 1337.1337,
    "array_stringfield": ["foo", "bar", "baz"],
}
# When we load the example all the keys for any unset non-required fields will are added
TEST_EXAMPLE_LOAD_REQUIRED = copy.deepcopy(TEST_EXAMPLE_REQUIRED)
TEST_EXAMPLE_LOAD_REQUIRED.update(
    {
        "lidar_point_cloud": {
            "lidar_metadata": {
                "lidar_model": "harambe",
                "lidar_calibration_error": None,
            },
        },
        "qc": None,
        "extra_metadata": None,
    }
)
# Example with everything field set
TEST_EXAMPLE_FULL = copy.deepcopy(TEST_EXAMPLE_REQUIRED)
TEST_EXAMPLE_FULL.update(
    {
        "lidar_point_cloud": {
            "lidar_metadata": {
                "lidar_model": "harambe",
                "lidar_calibration_error": 1.337,
            },
        },
        "qc": True,
        "extra_metadata": {
            "meta_1": 1,
            "meta_2": 2,
        },
    }
)

TEST_SERIALIZED_JSON_V2 = {
    "_description": "",
    "_json_version": 2,
    PRIMARY_KEYS_TAG: '["timestamp_ns"]',
    "fields": [
        {"_description": "Label of the example", "name": "label", "type": "int"},
        {
            "_description": "Lidar point cloud data",
            "name": "lidar_point_cloud",
            "type": {
                "fields": [
                    {
                        "_description": "Some metadata about the lidar",
                        "name": "lidar_metadata",
                        "type": {
                            "fields": [
                                {
                                    "_description": "Model of lidar used to generate data",
                                    "name": "lidar_model",
                                    "type": "string",
                                },
                                {
                                    "_description": "Some lidar calibration metric",
                                    "name": "lidar_calibration_error",
                                    "type": ["null", "double"],
                                },
                            ],
                            "name": "lidar_metadata",
                            "type": "record",
                        },
                    }
                ],
                "name": "lidar_point_cloud",
                "type": "record",
            },
        },
        {"_description": "Some timestamp field in ns", "name": "timestamp_ns", "type": "long"},
        {"_description": "Absolute speed of ego", "name": "ego_speed", "type": "float"},
        {"_description": "A quality control field", "name": "qc", "type": ["null", "boolean"]},
        {
            "_description": "Extra metadata",
            "name": "extra_metadata",
            "type": [
                "null",
                {
                    "fields": [
                        {"_description": "Metadata 1", "name": "meta_1", "type": "int"},
                        {"_description": "Metadata 2", "name": "meta_2", "type": ["null", "int"]},
                    ],
                    "name": "extra_metadata",
                    "type": "record",
                },
            ],
        },
        {
            "name": "array_stringfield",
            "type": {"_description": "some array", "items": "string", "name": "array_stringfield", "type": "array"},
        },
    ],
    "name": "fields",
    "type": "record",
}


class TestSchemaParseExample(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None

    def test_parse_full_example(self) -> None:
        parsed_example = dataparsing.parse_example(TEST_EXAMPLE_FULL, TEST_SCHEMA)
        self.assertEqual(parsed_example, TEST_EXAMPLE_FULL)

    def test_parse_required_fields(self) -> None:
        parsed_example = dataparsing.parse_example(TEST_EXAMPLE_REQUIRED, TEST_SCHEMA)
        self.assertEqual(parsed_example, TEST_EXAMPLE_LOAD_REQUIRED)

    def test_fail_required(self) -> None:
        example = copy.deepcopy(TEST_EXAMPLE_REQUIRED)
        del example["label"]
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn("Example missing keys", str(e.exception))

        example = copy.deepcopy(TEST_EXAMPLE_REQUIRED)
        del example["lidar_point_cloud"]
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn("Example missing keys", str(e.exception))

    def test_fail_type_int(self) -> None:
        example = copy.deepcopy(TEST_EXAMPLE_FULL)
        example["label"] = "SHOULD_BE_INT"
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn("Error at path label", str(e.exception))

    def test_fail_type_long(self) -> None:
        example = copy.deepcopy(TEST_EXAMPLE_FULL)
        example["timestamp_ns"] = "SHOULD_BE_LONG"
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn("Error at path timestamp_ns", str(e.exception))

    def test_fail_type_bool(self) -> None:
        example = copy.deepcopy(TEST_EXAMPLE_FULL)
        example["qc"] = "SHOULD_BE_BOOL"
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn("Error at path qc", str(e.exception))

    def test_fail_type_float(self) -> None:
        example = copy.deepcopy(TEST_EXAMPLE_FULL)
        example["ego_speed"] = "SHOULD_BE_FLOAT"
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn("Error at path ego_speed", str(e.exception))

    def test_fail_type_double(self) -> None:
        example: Dict[str, Any] = copy.deepcopy(TEST_EXAMPLE_FULL)
        example["lidar_point_cloud"]["lidar_metadata"]["lidar_calibration_error"] = "SHOULD_BE_DOUBLE"
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn(
            "Error at path lidar_point_cloud.lidar_metadata.lidar_calibration_error",
            str(e.exception),
        )

    def test_fail_type_string(self) -> None:
        example: Dict[str, Any] = copy.deepcopy(TEST_EXAMPLE_FULL)
        example["lidar_point_cloud"]["lidar_metadata"]["lidar_model"] = 1337
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn(
            "Error at path lidar_point_cloud.lidar_metadata.lidar_model",
            str(e.exception),
        )

    def test_fail_type_record(self) -> None:
        example = copy.deepcopy(TEST_EXAMPLE_FULL)
        example["lidar_point_cloud"] = "SHOULD_BE_DICT"
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn("Error at path lidar_point_cloud", str(e.exception))

    def test_fail_keys_record(self) -> None:
        example = copy.deepcopy(TEST_EXAMPLE_FULL)
        del example["label"]
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn("Error at path :", str(e.exception))

    def test_fail_type_array(self) -> None:
        example = copy.deepcopy(TEST_EXAMPLE_FULL)
        example["array_stringfield"] = "foo"
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn("Error at path array_stringfield:", str(e.exception))

    def test_fail_element_type_array(self) -> None:
        example = copy.deepcopy(TEST_EXAMPLE_FULL)
        example["array_stringfield"] = [1, 2, 3]
        with self.assertRaises(WickerSchemaException) as e:
            dataparsing.parse_example(example, TEST_SCHEMA)
        self.assertIn("Error at path array_stringfield.elem[0]:", str(e.exception))


class TestSchemaValidation(unittest.TestCase):
    def test_schema_no_primary_keys(self) -> None:
        with self.assertRaises(WickerSchemaException) as e:
            schema.DatasetSchema(fields=[], primary_keys=[])
        self.assertIn("The primary_keys attribute can not be empty", str(e.exception))

    def test_schema_invalid_primary_keys(self) -> None:
        with self.assertRaises(WickerSchemaException) as e:
            schema.DatasetSchema(fields=[], primary_keys=["unknown_key"])
        self.assertIn("'unknown_key' not found", str(e.exception))

    def test_schema_required_primary_keys(self) -> None:
        schema.DatasetSchema(fields=[schema.StringField("car_id")], primary_keys=["car_id"])
        with self.assertRaises(WickerSchemaException) as e:
            schema.DatasetSchema(
                fields=[schema.StringField("car_id", required=False)],
                primary_keys=["car_id"],
            )
        self.assertIn("must have the 'required' tag, but 'car_id' doesn't", str(e.exception))

    def test_schema_invalid_primary_keys_type(self) -> None:
        bad_fields = [
            schema.FloatField("float_key"),
            schema.DoubleField("double_key"),
            schema.RecordField("record_key", fields=[]),
        ]
        for f in bad_fields:
            with self.assertRaises(WickerSchemaException, msg=f"field.name={f.name}") as e:
                schema.DatasetSchema(fields=[f], primary_keys=[f.name])
            self.assertIn(f"'{f.name}' cannot be a primary key", str(e.exception), msg=f"field.name={f.name}")


class TestSchemaSerialization(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None

    def test_schema_name_error_dashes(self) -> None:
        with self.assertRaises(ValueError):
            schema.StringField(name="name-with-dashes")

    def test_schema_name_error_start_with_number(self) -> None:
        with self.assertRaises(ValueError):
            schema.StringField(name="0foo")
        with self.assertRaises(ValueError):
            schema.StringField(name="0")

    def test_schema_name_single_char(self) -> None:
        schema.StringField(name="q")

    def test_serialize_to_str(self) -> None:
        serialized = serialization.dumps(TEST_SCHEMA)
        avro.schema.parse(serialized)
        self.assertEqual(json.loads(serialized), TEST_SERIALIZED_JSON_V2)

    def test_serialize_to_str_pretty(self) -> None:
        serialized = serialization.dumps(TEST_SCHEMA, pretty=True)
        avro.schema.parse(serialized)
        self.assertEqual(serialized, json.dumps(TEST_SERIALIZED_JSON_V2, sort_keys=True, indent=4))

    def test_loads(self) -> None:
        self.assertEqual(
            TEST_SCHEMA,
            serialization.loads(json.dumps(TEST_SERIALIZED_JSON_V2)),
        )

    def test_loads_bad_type(self) -> None:
        with self.assertRaises(WickerSchemaException):
            serialized_bad_type = copy.deepcopy(TEST_SERIALIZED_JSON_V2)
            serialized_bad_type["fields"][0]["type"] = "BAD_TYPE_123"  # type: ignore
            serialization.loads(json.dumps(serialized_bad_type))

    def test_loads_bad_type_nullable(self) -> None:
        with self.assertRaises(WickerSchemaException):
            serialized_bad_type = copy.deepcopy(TEST_SERIALIZED_JSON_V2)
            serialized_bad_type["fields"][0]["type"] = ["null", "BAD_TYPE_123"]  # type: ignore
            serialization.loads(json.dumps(serialized_bad_type))


class TestSchemaLoading(unittest.TestCase):
    def test_load(self) -> None:
        loaded_example = dataloading.load_example(TEST_EXAMPLE_FULL, TEST_SCHEMA)

        # Assert that values are semantically equal
        self.assertEqual(TEST_EXAMPLE_FULL, loaded_example)

        # Assert that the IDs of the values are also equal (no additional copies of the data were created)
        self.assertEqual(id(TEST_EXAMPLE_FULL["label"]), id(loaded_example["label"]))
        self.assertEqual(
            id(TEST_EXAMPLE_FULL["lidar_point_cloud"]["lidar_metadata"]["lidar_model"]),
            id(loaded_example["lidar_point_cloud"]["lidar_metadata"]["lidar_model"]),
        )
        self.assertEqual(
            id(TEST_EXAMPLE_FULL["lidar_point_cloud"]["lidar_metadata"]["lidar_calibration_error"]),
            id(loaded_example["lidar_point_cloud"]["lidar_metadata"]["lidar_calibration_error"]),
        )
        self.assertEqual(id(TEST_EXAMPLE_FULL["timestamp_ns"]), id(loaded_example["timestamp_ns"]))
        self.assertEqual(id(TEST_EXAMPLE_FULL["ego_speed"]), id(loaded_example["ego_speed"]))
        self.assertEqual(id(TEST_EXAMPLE_FULL["qc"]), id(loaded_example["qc"]))

        # Assert that the IDs of dictionaries and lists are not equal (data is not modified in-place)
        self.assertNotEqual(id(TEST_EXAMPLE_FULL), id(loaded_example))
        self.assertNotEqual(
            id(TEST_EXAMPLE_FULL["lidar_point_cloud"]),
            id(loaded_example["lidar_point_cloud"]),
        )
        self.assertNotEqual(
            id(TEST_EXAMPLE_FULL["lidar_point_cloud"]["lidar_metadata"]),
            id(loaded_example["lidar_point_cloud"]["lidar_metadata"]),
        )
        self.assertNotEqual(id(TEST_EXAMPLE_FULL["extra_metadata"]), id(loaded_example["extra_metadata"]))
        self.assertNotEqual(id(TEST_EXAMPLE_FULL["array_stringfield"]), id(loaded_example["array_stringfield"]))

    def test_load_columns_required(self) -> None:
        subset_example = copy.deepcopy(TEST_EXAMPLE_REQUIRED)
        del subset_example["label"]
        loaded_example = dataloading.load_example(subset_example, TEST_SCHEMA)

        # Assert that values are semantically equal
        self.assertEqual(subset_example, loaded_example)

    def test_load_extra_keys_ignored(self) -> None:
        extra_keys_example = copy.deepcopy(TEST_EXAMPLE_FULL)
        extra_keys_example["extra_key_foo"] = 1
        loaded_example = dataloading.load_example(extra_keys_example, TEST_SCHEMA)

        # Assert that values are semantically equal
        self.assertEqual(TEST_EXAMPLE_FULL, loaded_example)

    def test_load_record_from_list_kv_tuples(self) -> None:
        example = copy.deepcopy(TEST_EXAMPLE_FULL)
        example["lidar_point_cloud"]["lidar_metadata"] = [
            (k, v) for k, v in example["lidar_point_cloud"]["lidar_metadata"].items()
        ]
        example["lidar_point_cloud"] = [(k, v) for k, v in example["lidar_point_cloud"].items()]
        parsed_example = dataloading.load_example(example, TEST_SCHEMA)
        self.assertEqual(parsed_example, TEST_EXAMPLE_FULL)


def test_schema(testcase: unittest.TestCase, schema_to_test: schema.DatasetSchema) -> None:
    serialized = serialization.dumps(schema_to_test, pretty=True)
    loaded = serialization.loads(serialized)
    loaded_serialized = serialization.dumps(schema_to_test, pretty=True)
    testcase.assertEqual(loaded, schema_to_test, msg=f"{serialized} vs {loaded_serialized}")
    testcase.assertTrue(json.loads(serialized))
    testcase.assertTrue(avro.schema.parse(serialized))


class TestObjectSchemas(unittest.TestCase):
    OBJECT_FIELD = schema.ObjectField("encoded_vector", VectorCodec(compression_method=12), required=False)
    SCHEMA = schema.DatasetSchema(
        fields=[OBJECT_FIELD, schema.StringField("sample_id")],
        primary_keys=["sample_id"],
    )
    EXAMPLE = {"sample_id": "sample000", "encoded_vector": Vector([1, 2, 3, 4])}
    EXAMPLE_BAD_TYPE = {"sample_id": "sample000", "encoded_vector": [1, 2, 3, 4]}
    EXAMPLE_NONE = {"sample_id": "sample000"}

    def test_serialization(self) -> None:
        serialized = serialization.dumps(TestObjectSchemas.SCHEMA, pretty=True)
        loaded = serialization.loads(serialized)
        loaded_serialized = serialization.dumps(TestObjectSchemas.SCHEMA, pretty=True)
        self.assertEqual(loaded, TestObjectSchemas.SCHEMA, msg=f"{serialized} vs {loaded_serialized}")
        self.assertTrue(json.loads(serialized))
        self.assertTrue(avro.schema.parse(serialized))

    def test_deserialization_unknown_codec(self) -> None:
        # Test the case where we want to deserialize a schema and we don't have the necessary codec.
        serialized = serialization.dumps(TestObjectSchemas.SCHEMA, pretty=True)
        serialized = serialized.replace("VectorCodec", "UnknownCodec")
        # By default we should fail
        with self.assertRaises(WickerSchemaException):
            loaded = serialization.loads(serialized)
        # But with the treat_objects_as_bytes=True we should be able to do it.
        loaded = serialization.loads(serialized, treat_objects_as_bytes=True)
        serialized2 = serialization.dumps(loaded, pretty=True)
        # Make sure that if we reserialize the schema that got loaded with treat_objects_as_bytes=True, we get the
        # same thing as the original schema.
        self.assertEqual(serialized, serialized2)

    def test_good_example1(self) -> None:
        # Check parsing and loading
        parsed_example = dataparsing.parse_example(TestObjectSchemas.EXAMPLE, TestObjectSchemas.SCHEMA)
        self.assertTrue(isinstance(parsed_example["encoded_vector"], bytes))
        loaded_schema = serialization.loads(serialization.dumps(TestObjectSchemas.SCHEMA, pretty=True))
        loaded_example = dataloading.load_example(parsed_example, loaded_schema)
        self.assertEqual(loaded_example, TestObjectSchemas.EXAMPLE)
        assert isinstance(loaded_schema.schema_record.fields[0], schema.ObjectField)  # Make mypy happy
        assert isinstance(loaded_schema.schema_record.fields[0].codec, VectorCodec)
        self.assertEqual(loaded_schema.schema_record.fields[0].codec.compression_method, 12)

    def test_example_none(self) -> None:
        parsed_example = dataparsing.parse_example(TestObjectSchemas.EXAMPLE_NONE, TestObjectSchemas.SCHEMA)
        loaded_example = dataloading.load_example(parsed_example, TestObjectSchemas.SCHEMA)
        self.assertEqual(loaded_example, TestObjectSchemas.EXAMPLE_NONE)

    def test_bad_validation(self) -> None:
        with self.assertRaises(WickerSchemaException):
            dataparsing.parse_example(TestObjectSchemas.EXAMPLE_BAD_TYPE, TestObjectSchemas.SCHEMA)

    def test_loads_bad_l5ml_metatype(self) -> None:
        with self.assertRaises(WickerSchemaException) as err:
            serialized_bad_type = json.loads(serialization.dumps(TestObjectSchemas.SCHEMA))
            serialized_bad_type["fields"][0]["_l5ml_metatype"] = "BAD_TYPE_123"  # type: ignore
            serialization.loads(json.dumps(serialized_bad_type))
        self.assertIn("Unhandled _l5ml_metatype for avro bytes type: BAD_TYPE_123", str(err.exception))
