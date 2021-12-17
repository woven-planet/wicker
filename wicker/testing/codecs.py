import json
from typing import Any, Dict, List, Type

from wicker.schema import codecs


class Vector:
    def __init__(self, data: List[int]):
        self.data = data

    def __eq__(self, other: Any) -> bool:
        return super().__eq__(other) and self.data == other.data


class VectorCodec(codecs.Codec):
    def __init__(self, compression_method: int) -> None:
        self.compression_method = compression_method

    @staticmethod
    def _codec_name() -> str:
        return "VectorCodec"

    def save_codec_to_dict(self) -> Dict[str, Any]:
        return {"compression_method": self.compression_method}

    @staticmethod
    def load_codec_from_dict(data: Dict[str, Any]) -> codecs.Codec:
        return VectorCodec(compression_method=data["compression_method"])

    def validate_and_encode_object(self, obj: Vector) -> bytes:
        # Inefficient but simple encoding method for testing.
        return json.dumps(obj.data).encode("utf-8")

    def decode_object(self, data: bytes) -> Vector:
        return Vector(json.loads(data.decode()))

    def object_type(self) -> Type[Any]:
        return Vector
