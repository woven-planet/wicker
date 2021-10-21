from typing import Any, Dict, Optional, Tuple, Type, TypeVar

from wicker.core.errors import WickerSchemaException

T = TypeVar("T")


# Unfortunately, mypy has weak support for recursive typing so we resort to Any here, but Any
# should really be an alias for: Union[AvroRecord, str, bool, int, float, bytes]
AvroRecord = Dict[str, Any]


def validate_field_type(val: Any, type_: Type[T], required: bool, current_path: Tuple[str, ...]) -> Optional[T]:
    """Validates the type of a field

    :param val: value to validate
    :type val: Any
    :param type_: type to validate against
    :type type_: Type[T]
    :param required: whether or not the value is required (to be non-None)
    :type required: bool
    :param current_path: current parsing path
    :type current_path: Tuple[str, ...]
    :raises WickerSchemaException: when parsing error occurs
    :return: val, but validated to be of type T
    :rtype: T
    """
    if val is None:
        if not required:
            return val
        raise WickerSchemaException(
            f"Error at path {'.'.join(current_path)}: Example provided a None value for required field"
        )
    elif not isinstance(val, type_):
        raise WickerSchemaException(
            f"Error at path {'.'.join(current_path)}: Example provided a {type(val)} value, expected {type_}"
        )
    return val


def validate_dict(val: Any, required: bool, current_path: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """Validates a dictionary

    :param val: incoming value to validate as a dictionary
    :type val: Any
    :param required: whether or not the value is required (to be non-None)
    :type required: bool
    :param current_path: current parsing path
    :type current_path: Tuple[str, ...]
    :return: parsed dictionary
    :rtype: dict
    """
    if val is None:
        if not required:
            return val
        raise WickerSchemaException(
            f"Error at path {'.'.join(current_path)}: Example provided a None value for required field"
        )
    # PyArrow returns record fields as lists of (k, v) tuples
    elif isinstance(val, list):
        try:
            parsed_val = dict(val)
        except ValueError:
            raise WickerSchemaException(f"Error at path {'.'.join(current_path)}: Unable to convert list to dict")
    elif isinstance(val, dict):
        parsed_val = val
    else:
        raise WickerSchemaException(f"Error at path {'.'.join(current_path)}: Unable to convert object to dict")

    return validate_field_type(parsed_val, dict, required, current_path)
