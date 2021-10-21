from __future__ import annotations

import abc
from typing import Any, Dict, Type


class Codec(abc.ABC):
    """Base class for all object decoder/encoders.

    Defining Codec classes allows user to use arbitrary types in their Wicker Examples. Subclasses of Codec provide
    functionality to serialize/deserialize data to/from underlying storage, and can be used to abstract functionality
    such as compression, validation of object fields and provide adaptor functionality to other libraries (e.g. NumPy,
    PIL, torch).

    Wicker provides some Codec implementations by default for NumPy, but users can choose to define their own Codecs
    and use them during data-writing and data-reading.
    """

    # Maps from codec_name to codec class.
    codec_registry: Dict[str, Type[Codec]] = {}

    def __init_subclass__(cls: Type[Codec], **kwargs: Any) -> None:
        """Automatically register subclasses of Codec."""
        if cls._codec_name() in Codec.codec_registry:
            raise KeyError(f"Codec '{cls._codec_name()}' was already defined.")
        if cls._codec_name():
            Codec.codec_registry[cls._codec_name()] = cls

    @staticmethod
    @abc.abstractmethod
    def _codec_name() -> str:
        """Needs to be overriden to return a globally unique name for the codec."""
        pass

    def get_codec_name(self) -> str:
        """Accessor for _codec_name. In general, derived classes should not touch this, and should just implement
        _codec_name. This accessor is used internally to support generic schema serialization/deserialization.
        """
        return self._codec_name()

    def save_codec_to_dict(self) -> Dict[str, Any]:
        """If you want to save some parameters of this codec with the dataset
        schema, return the fields here. The returned dictionary should be JSON compatible.
        Note that this is a dataset-level value, not a per example value."""
        return {}

    @staticmethod
    @abc.abstractmethod
    def load_codec_from_dict(data: Dict[str, Any]) -> Codec:
        """Create a new instance of this codec with the given parameters."""
        pass

    @abc.abstractmethod
    def validate_and_encode_object(self, obj: Any) -> bytes:
        """Encode the given object into bytes. The function is also responsible for validating the data.
        :param obj: Object to encode
        :return: The encoded bytes for the given object."""
        pass

    @abc.abstractmethod
    def decode_object(self, data: bytes) -> Any:
        """Decode an object from the given bytes. This is the opposite of validate_and_encode_object.
        We expect obj == decode_object(validate_and_encode_object(obj))
        :param data: bytes to decode.
        :return: Decoded object."""
        pass

    def object_type(self) -> Type[Any]:
        """Return the expected type of the objects handled by this codec.
        This method can be overriden to match more specific classes."""
        return object

    def __eq__(self, other: Any) -> bool:
        return (
            super().__eq__(other)
            and self.get_codec_name() == other.get_codec_name()
            and self.save_codec_to_dict() == other.save_codec_to_dict()
        )
