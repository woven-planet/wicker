import io
import unittest

import numpy as np

from wicker.core.errors import WickerSchemaException
from wicker.schema.schema import WickerNumpyCodec

EYE_ARR = np.eye(4)
eye_arr_bio = io.BytesIO()
np.save(eye_arr_bio, EYE_ARR)
EYE_ARR_BYTES = eye_arr_bio.getvalue()


class TestNumpyCodec(unittest.TestCase):
    def test_codec_none_shape(self) -> None:
        codec = WickerNumpyCodec(shape=None, dtype="float64")
        self.assertEqual(codec.get_codec_name(), "wicker_numpy")
        self.assertEqual(codec.object_type(), np.ndarray)
        self.assertEqual(
            codec.save_codec_to_dict(),
            {
                "dtype": "float64",
                "shape": None,
            },
        )
        self.assertEqual(WickerNumpyCodec.load_codec_from_dict(codec.save_codec_to_dict()), codec)
        self.assertEqual(codec.validate_and_encode_object(EYE_ARR), EYE_ARR_BYTES)
        np.testing.assert_equal(codec.decode_object(EYE_ARR_BYTES), EYE_ARR)

    def test_codec_unbounded_dim_shape(self) -> None:
        codec = WickerNumpyCodec(shape=(-1, -1), dtype="float64")
        self.assertEqual(codec.get_codec_name(), "wicker_numpy")
        self.assertEqual(codec.object_type(), np.ndarray)
        self.assertEqual(
            codec.save_codec_to_dict(),
            {
                "dtype": "float64",
                "shape": [-1, -1],
            },
        )
        self.assertEqual(WickerNumpyCodec.load_codec_from_dict(codec.save_codec_to_dict()), codec)
        self.assertEqual(codec.validate_and_encode_object(EYE_ARR), EYE_ARR_BYTES)
        np.testing.assert_equal(codec.decode_object(EYE_ARR_BYTES), EYE_ARR)

        # Should raise when provided with bad shapes with too few/many dimensions
        with self.assertRaises(WickerSchemaException):
            codec.validate_and_encode_object(np.ones((10,)))
        with self.assertRaises(WickerSchemaException):
            codec.validate_and_encode_object(np.ones((10, 10, 10)))

    def test_codec_fixed_shape(self) -> None:
        codec = WickerNumpyCodec(shape=(4, 4), dtype="float64")
        self.assertEqual(codec.get_codec_name(), "wicker_numpy")
        self.assertEqual(codec.object_type(), np.ndarray)
        self.assertEqual(
            codec.save_codec_to_dict(),
            {
                "dtype": "float64",
                "shape": [4, 4],
            },
        )
        self.assertEqual(WickerNumpyCodec.load_codec_from_dict(codec.save_codec_to_dict()), codec)
        self.assertEqual(codec.validate_and_encode_object(EYE_ARR), EYE_ARR_BYTES)
        np.testing.assert_equal(codec.decode_object(EYE_ARR_BYTES), EYE_ARR)

        # Should raise when provided with bad shapes with too few/many/wrong dimensions
        with self.assertRaises(WickerSchemaException):
            codec.validate_and_encode_object(np.ones((10,)))
        with self.assertRaises(WickerSchemaException):
            codec.validate_and_encode_object(np.ones((10, 10, 10)))
        with self.assertRaises(WickerSchemaException):
            codec.validate_and_encode_object(np.ones((5, 4)))

    def test_codec_bad_dtype(self) -> None:
        with self.assertRaises(WickerSchemaException):
            WickerNumpyCodec(shape=(4, 4), dtype="SOME_BAD_DTYPE")
