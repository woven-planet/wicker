#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <arrow/python/pyarrow.h>
#pragma GCC diagnostic pop
#include <arrow/array/array_base.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/chunked_array.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include "arrow_util.h"
#include "temporal_windowing.h"
#include "util.h"

namespace py = pybind11;

namespace l5ml_datastore {

py::object applyTemporalWindowSpecification(
    py::object pyarrow_table,
    const WindowSpecification& spec,
    const std::string& hash_key,
    const std::string& range_key) {
  arrow::py::import_pyarrow();

  // Convert PyObject* to native C++ object
  arrow::Result<std::shared_ptr<arrow::Table>> unwrapped_table_result =
      arrow::py::unwrap_table(pyarrow_table.ptr());
  if (!unwrapped_table_result.ok()) {
    FATAL("Error decoding the table.\n");
    return py::object();
  }
  std::shared_ptr<arrow::Table> table = *unwrapped_table_result;

  // Create the new schema. Some fields are copied over, some are dropped, some are converted to
  // lists of the same type, depending on the WindowSpecification.
  std::vector<std::shared_ptr<arrow::Field>> new_fields;
  new_fields.push_back(table->schema()->GetFieldByName(hash_key));
  new_fields.push_back(table->schema()->GetFieldByName(range_key));

  for (const ColumnHistorySpecification& col_spec : spec.column_specs) {
    // Most field types are pass-through, except for the ones that return lists.
    if (col_spec.results_as_list) {
      // The windowing function will capture several values for this column, so we need to change
      // the type to a list of the type of the original field
      std::shared_ptr<arrow::Field> list_field(new arrow::Field(
          col_spec.column_name,
          std::make_shared<arrow::FixedSizeListType>(
              table->schema()->GetFieldByName(col_spec.column_name)->type(),
              col_spec.spec.size())));
      new_fields.push_back(list_field);
    } else {
      // We capture one element here, so we keep the original type.
      new_fields.push_back(table->schema()->GetFieldByName(col_spec.column_name));
      CHECK(col_spec.spec.size() == 1);
    }
  }
  std::shared_ptr<arrow::Schema> new_schema(
      new arrow::Schema(new_fields, table->schema()->metadata()));

  // Create iterators for the columns of the original data that we need to extract data from.
  // The iterators point to data owned by the carrays, and we need to keep those shared_ptr for as
  // long as we need the iterators.
  std::vector<std::shared_ptr<arrow::ChunkedArray>> carrays;
  std::vector<ChunkedArrayIterator> data_column_iterators;
  for (const ColumnHistorySpecification& col_spec : spec.column_specs) {
    std::shared_ptr<arrow::ChunkedArray> carray = table->GetColumnByName(col_spec.column_name);
    CHECK_MSG(carray, "Can not retrieve column " + col_spec.column_name);
    carrays.push_back(carray);
    data_column_iterators.push_back(ChunkedArrayIterator(carray.get()));
  }
  CHECK(data_column_iterators.size() == spec.column_specs.size());

  // Extract all the car_id/timestamps for all the rows, it makes it easier to run the selection
  // loops below.
  // TODO(flefevere): For now this is somewhat specific (which simplifies the code a lot).
  // But we should be able to make it work with other keys/types if that becomes needed.
  std::shared_ptr<arrow::ChunkedArray> hash_values_carray = table->GetColumnByName(hash_key);
  std::shared_ptr<arrow::ChunkedArray> range_values_carray = table->GetColumnByName(range_key);
  std::vector<arrow::util::string_view> hash_values =
      chunkedArrayToStringViewVector(*hash_values_carray);
  std::vector<int64_t> range_values =
      chunkedArrayToNumericalVector<arrow::Int64Type>(*range_values_carray);

  // Create array builders so that we can output the results.
  arrow::StringBuilder hash_values_builder;
  arrow::NumericBuilder<arrow::Int64Type> range_values_builder;

  // Builders for the data columns.
  std::vector<std::shared_ptr<arrow::ArrayBuilder>> data_column_builders;
  // Skip the hash/range keys when making the builder.
  for (size_t i = 2; i < new_fields.size(); ++i) {
    std::unique_ptr<arrow::ArrayBuilder> tmp;
    arrow::MakeBuilder(arrow::default_memory_pool(), new_fields[i]->type(), &tmp);
    data_column_builders.push_back(std::shared_ptr<arrow::ArrayBuilder>(tmp.release()));
  }
  CHECK(data_column_builders.size() == data_column_iterators.size());

  // Select the data as specified in "spec"
  for (ssize_t i = 0; i < range_values_carray->length(); ++i) {
    const arrow::util::string_view hash_value = hash_values[i];
    const int64_t range_value = range_values[i];
    bool window_valid = true;
    std::vector<std::vector<ChunkedArrayIterator>> row_data(spec.column_specs.size());
    for (size_t c = 0; c < spec.column_specs.size(); ++c) {
      row_data[c].clear();
      const ColumnHistorySpecification& col_spec = spec.column_specs[c];
      // For each column, find all the elements we need. We start from the current index and go
      // backwards.
      for (const CellSpecification& cell_spec : col_spec.spec) {
        // The iterator we use to go backwards.
        ChunkedArrayIterator window_cell_iterator = data_column_iterators[c];
        // This iterator will be set to the cell that matches our query, if such a cell exists.
        ChunkedArrayIterator found_cell_iterator;
        while (window_cell_iterator.index() >= 0) {
          const arrow::util::string_view& cell_hash_value =
              hash_values[window_cell_iterator.index()];
          const int64_t cell_range_value = range_values[window_cell_iterator.index()];
          const int64_t delta = cell_range_value - range_value;
          if (cell_hash_value != hash_value || delta < cell_spec.min_offset) break;
          if (delta <= cell_spec.max_offset) {
            // TODO: consider resolving conflicts in a better way when multiple options are possible
            // Here, we pick the first matching element that matches.
            found_cell_iterator = window_cell_iterator;
            break;
          }
          --window_cell_iterator;
        }
        if (cell_spec.is_required &&
            (found_cell_iterator.empty() || found_cell_iterator.isNull())) {
          window_valid = false;
          break;
        }
        row_data[c].push_back(found_cell_iterator);
      }
      if (!window_valid) break;
    }
    if (window_valid) {
      hash_values_builder.Append(std::string(hash_value));
      range_values_builder.Append(range_value);
      for (size_t c = 0; c < spec.column_specs.size(); ++c) {
        const ColumnHistorySpecification& col_spec = spec.column_specs[c];
        if (col_spec.results_as_list) {
          copyListDataToBuilder(row_data[c], data_column_builders[c].get());
        } else {
          // Single value. Copy the source data from the given iterator.
          CHECK(row_data[c].size() == 1);
          copyDataToBuilder(row_data[c][0], data_column_builders[c].get());
        }
      }
    }
    for (size_t c = 0; c < data_column_iterators.size(); ++c) {
      ++data_column_iterators[c];
    }
  }
  // Finalize the arrays and create the final arrow::Table
  // all_builders includes the index columns and the data columns.
  std::vector<arrow::ArrayBuilder*> all_builders;
  all_builders.push_back(&hash_values_builder);
  all_builders.push_back(&range_values_builder);
  for (size_t i = 0; i < data_column_builders.size(); ++i) {
    all_builders.push_back(data_column_builders[i].get());
  }
  CHECK(all_builders.size() == new_fields.size());
  return createTableFromBuilders(all_builders, new_schema);
}

}  // namespace l5ml_datastore
