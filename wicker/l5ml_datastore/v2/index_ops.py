"""Collection of functions to manipulate datasets.
"""
import copy
from typing import Optional, Tuple, TypeVar

import pyarrow  # type: ignore
import pyarrow.compute  # type: ignore

from wicker.core.datasets import AbstractDataset
from wicker.core.definitions import Example
from wicker.core.errors import WickerDatastoreException, WickerSchemaException
from wicker.l5ml_datastore import cpp_extensions
from wicker.schema.schema import ArrayField, DatasetSchema

CellSpecification = cpp_extensions.CellSpecification
ColumnHistorySpecification = cpp_extensions.ColumnHistorySpecification
WindowSpecification = cpp_extensions.WindowSpecification

T = TypeVar("T")


def _assert_not_none(x: Optional[T]) -> T:
    assert x is not None
    return x


def merge_schemas(
    left: DatasetSchema, right: DatasetSchema, left_required: bool = True, right_required: bool = True
) -> DatasetSchema:
    """Create a new schema by combining left and right schemas. Raises WickerSchemaException if the two
    schemas are incompatible.
    The left_required and right_required flags control which fields are required in the merged schema.
    :param left: One of the schemas to merge.
    :param right: The other schema to merge.
    :param left_required: If a field comes from the left schema, it is required in the final schema iff it is required
      in the left schema and left_required is set.
    :param right_required: If a field comes from the right schema, it is required in the final schema iff it is required
      in the right schema and right_required is set.
    :return: A new schema with fields from both left and right.
    """
    if left.primary_keys != right.primary_keys:
        raise WickerSchemaException(
            "The schemas have different primary keys: Got " f"{left.primary_keys} and  {right.primary_keys}"
        )
    # Make deep copies of the schemas so that we can change the "required" status.
    left = copy.deepcopy(left)
    right = copy.deepcopy(right)

    new_fields = left._columns
    if not left_required:
        for name, field in new_fields.items():
            field.required = name in left.primary_keys
    for name, field in right._columns.items():
        if name in new_fields:
            if not new_fields[name].is_compatible(field):
                raise WickerSchemaException(
                    f"Can not merge schemas. Field '{name}' seen with two different types. "
                    f"{type(new_fields[name])}  {type(field)}"
                )
            if right_required and field.required:
                new_fields[name].required = True
        else:
            field.required = field.required and right_required
            new_fields[name] = field
    return DatasetSchema(fields=list(new_fields.values()), primary_keys=left.primary_keys)


def join_datasets(left: AbstractDataset, right: AbstractDataset, how: str = "outer") -> AbstractDataset:
    """Join two datasets (that have the same primary keys) on their primary keys.
    :param left: First dataset to merge.
    :param right: Second dataset to merge.
    :param how: Type of merge to apply. The possible values follow the options for pandas.Dataframe.merge:
      "left": use only keys from left frame, similar to a SQL left outer join
      "right": use only keys from right frame, similar to a SQL right outer join
      "outer": use union of keys from both frames, similar to a SQL full outer join
      "inner": use intersection of keys from both frames, similar to a SQL inner join
    :return: A new dataset resulting from the merge of "left" and "right".
    """
    # Note, this function does a very simple merge based on equality of keys.
    # Maybe we should have a function for resolving approximate timestamps.
    # We could truncate the precision of timestamps, but that's not a very good solution.
    # Better to use the left side's timestamp as reference and merge right side rows that are within an
    # error threshold.
    if not left.is_compatible(right):
        raise WickerDatastoreException(
            "Dataset a and b are not compatible. Datasets must be of the same type "
            "and have the same backing infrastructure."
        )
    left_required = False
    right_required = False
    if how == "inner":
        left_required = True
        right_required = True
    elif how == "left":
        left_required = True
    elif how == "right":
        right_required = True
    elif how == "outer":
        pass
    else:
        raise WickerDatastoreException(f"Unsupported join type {how}.")
    dst_schema = merge_schemas(left.schema, right.schema, left_required, right_required)

    dst_df = left.arrow_table.to_pandas().merge(right.arrow_table.to_pandas(), how=how, on=left.schema.primary_keys)
    dst_df.sort_index(inplace=True)

    return left.create_compatible_anonymous_dataset(pyarrow.Table.from_pandas(dst_df), dst_schema)


class ColumnWindowBuilder:
    """Collection of helper functions to build window specifications for apply_temporal_window_specifiction."""

    @staticmethod
    def uniform_history(
        column_name: str, num_samples: int, delta: int, margin: int, is_required: bool = True
    ) -> ColumnHistorySpecification:
        """uniform_history creates a request for regularly spaced objects before the current row.
        For example:
        uniform_history("image", 4, 100*MS, 10*MS, is_required=True)
        requests that for each example we fetch 4 objects from the "image" column that are at
        0ms, -100ms, -200ms, -300ms relative to the example's row.
        :param column_name: Name of the column of the dataset to retrieve objects from.
        :param num_samples: Total number of objects to retrieve for each output row. Object #i needs to have its range
          key in the interval [row_range_key - i * delta - margin, row_range_key - i * delta + margin]
        :param delta: Distance between the range key of the samples we want.
        :param margin: Allowed error between the desired distance and the actual distance in the dataset.
        :param is_required: If set, only rows where we find num_samples objects matching the provided delta and margin
          will be returned by the apply_temporal_window_specifiction function. If unset, missing data will be set to
        None.
        :return: A ColumnHistorySpecification that can be added to a WindowSpecification object and passed to
        apply_temporal_window_specifiction to filter a dataset.
        """
        spec = [
            CellSpecification(
                min_offset=(-i * delta - margin),
                max_offset=(-i * delta + margin),
                is_required=is_required,
            )
            for i in range(num_samples)
        ]
        spec.reverse()
        return ColumnHistorySpecification(column_name, spec=spec).set_results_as_list(True)

    @staticmethod
    def single_value(column_name: str, margin: int = 0, is_required: bool = True) -> ColumnHistorySpecification:
        """single_value is a helper function to build specifications for the apply_temporal_window_specifiction
        function.
        single_value creates a request for a single object at the range key as the current row.
        In most cases the range_key will be the second primary key, and will represent a timestamp.
        :param column_name: Name of the column of the dataset to retrieve objects from.
        :param margin: Allowed error between the range key of the data to fetch and the range key of the row in the
          dataset.
        :param is_required: If set, only rows where we find an object matching the provided margin will
          be returned by the apply_temporal_window_specifiction function. If unset, missing data will be set to None.
        :return: A ColumnHistorySpecification that can be added to a WindowSpecification object and passed to
          apply_temporal_window_specifiction to filter a dataset.
        """
        cell_spec = CellSpecification(min_offset=-margin, max_offset=margin, is_required=is_required)
        return ColumnHistorySpecification(column_name, spec=[cell_spec])


def sample_by_range_key(dataset: AbstractDataset, min_interval: int) -> AbstractDataset:
    """Sample a dataset and return a new dataset where two consecutive samples have at least a difference of
    "min_interval" in their range keys.
    :param dataset: Input dataset to resample.
    :param min_interval: Minimum range key distance between two examples in the output dataset.
    :return: Resampled dataset.
    """
    hash_key, range_key = _get_range_and_hash_keys(dataset)
    dst_table = cpp_extensions.sample_by_range_key(dataset.arrow_table, hash_key, range_key, min_interval)
    return dataset.create_compatible_anonymous_dataset(dst_table, dataset.schema)


def _get_range_and_hash_keys(dataset: AbstractDataset) -> Tuple[str, str]:
    """Find the name of the hash key and range_key in the dataset.
    In the current implementation this will be the name of the first and second primary keys. These keys are
    expected to be of type string and int64 respectively.
    """
    if len(dataset.schema.primary_keys) < 2:
        raise WickerDatastoreException("This function expects 2 primary keys of type string and int64 respectively.")
    hash_key = dataset.schema.primary_keys[0]
    range_key = dataset.schema.primary_keys[1]
    hash_key_type = dataset.arrow_table.field(hash_key).type
    range_key_type = dataset.arrow_table.field(range_key).type
    if hash_key_type != pyarrow.string():
        raise WickerDatastoreException(
            "The first primary key (hash_key) is expected to be of string type. " f"Found {hash_key_type}"
        )
    if range_key_type != pyarrow.int64():
        raise WickerDatastoreException(
            "The second primary key (range_key) is expected to be of int64 type. " f"Found {range_key_type}"
        )
    return hash_key, range_key


def apply_temporal_window_specification(dataset: AbstractDataset, spec: WindowSpecification) -> AbstractDataset:
    """Apply a windowing function to a dataset.
    :param dataset: Dataset to apply windowing operation to.
    :param spec: Specification of the operation to Apply.
    The specification can be built using the WindowSpecification, ColumnHistorySpecification and CellSpecification
    classes, or using helper methods of ColumnWindowBuilder like single_value and uniform_history.
    Usage example:
    Suppose the dataset contains the following data:

        car_id  timestamp_ms  image      detections
    1   car1      1230000      <img1>    <detection1>
    2   car1      1230100      <img2>    <detection2>
    3   car1      1230200      <img3>    <detection3>
    4   car2      1230000      <img4>    <detection4>
    5   car2      1230100      <img5>    <detection5>
    6   car2      1230200      <img6>    <detection6>
    7   car2      1230300      <img7>    <detection7>

    Let's define
    spec = WindowSpecification(
        column_specs=[
            single_value("detections"),
            uniform_history("image", 3, 100, 20, is_required=True)
        ]
    )

    The WindowSpecification indicates that we want to extract data from 2 columns:
    "detections" and "image".
    For the "detections" column we only want to get one value, the value at the current row.
    For the "image" column, we want to get an array of 3 values: one image every 100ms with an error of +/-20ms:
    which translates to one image in the -220..-180ms range, one image in the -120..-80ms range, and one image in the
    -20..+20ms range, in that order.

    If we apply_temporal_window_specification(dataset, spec), we get the following output:
        car_id  timestamp_ms     image                detections
    1   car1      1230200     [<img1>,<img2>,<img3>] <detection3>
    2   car2      1230200     [<img4>,<img5>,<img6>] <detection6>
    3   car2      1230300     [<img5>,<img6>,<img7>] <detection7>
    """
    src_schema = dataset.schema
    src_table = dataset.arrow_table
    hash_key, range_key = _get_range_and_hash_keys(dataset)

    fields = [_assert_not_none(src_schema.get_column(k)) for k in src_schema.primary_keys]
    for col_spec in spec.column_specs:
        assert col_spec.column_name not in src_schema.primary_keys
        field = _assert_not_none(src_schema.get_column(col_spec.column_name))
        if col_spec.results_as_list:
            fields.append(ArrayField(field))
        else:
            fields.append(field)
    dst_schema = DatasetSchema(fields, primary_keys=src_schema.primary_keys)
    dst_table = cpp_extensions.apply_temporal_window_specification(src_table, spec, hash_key, range_key)
    return dataset.create_compatible_anonymous_dataset(dst_table, dst_schema)
