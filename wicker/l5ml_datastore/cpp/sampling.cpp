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
#include "sampling.h"
#include "util.h"

namespace py = pybind11;

namespace l5ml_datastore {

py::object sampleByRangeKey(
    py::object pyarrow_table,
    const std::string& hash_key,
    const std::string& range_key,
    const int64_t min_interval) {
  arrow::py::import_pyarrow();

  // Convert PyObject* to native C++ object
  arrow::Result<std::shared_ptr<arrow::Table>> unwrapped_table_result =
      arrow::py::unwrap_table(pyarrow_table.ptr());
  if (!unwrapped_table_result.ok()) {
    FATAL("Error decoding the table.\n");
    return py::object();
  }
  std::shared_ptr<arrow::Table> table = *unwrapped_table_result;

  // Create iterators for the original data.
  std::vector<std::shared_ptr<arrow::ChunkedArray>> carrays = table->columns();
  std::vector<ChunkedArrayIterator> data_column_iterators;
  for (const std::shared_ptr<arrow::ChunkedArray>& carray : carrays) {
    data_column_iterators.push_back(ChunkedArrayIterator(carray.get()));
  }

  // TODO(flefevere): For now this is somewhat specific (which simplifies the code a lot).
  // But we should be able to make it work with other keys/types if that becomes needed.
  std::shared_ptr<arrow::ChunkedArray> hash_values_carray = table->GetColumnByName(hash_key);
  std::shared_ptr<arrow::ChunkedArray> range_values_carray = table->GetColumnByName(range_key);
  std::vector<arrow::util::string_view> hash_values =
      chunkedArrayToStringViewVector(*hash_values_carray);
  std::vector<int64_t> range_values =
      chunkedArrayToNumericalVector<arrow::Int64Type>(*range_values_carray);

  // Create array builders so that we can output the results.
  std::vector<std::shared_ptr<arrow::ArrayBuilder>> builders;
  for (const std::shared_ptr<arrow::ChunkedArray>& carray : carrays) {
    std::unique_ptr<arrow::ArrayBuilder> tmp;
    arrow::MakeBuilder(arrow::default_memory_pool(), carray->type(), &tmp);
    builders.push_back(std::shared_ptr<arrow::ArrayBuilder>(tmp.release()));
  }
  CHECK(builders.size() == table->columns().size());

  // Select the data.
  int64_t last_range_value = 0;
  arrow::util::string_view last_hash_value;
  for (ssize_t i = 0; i < range_values_carray->length(); ++i) {
    const arrow::util::string_view hash_value = hash_values[i];
    const int64_t range_value = range_values[i];
    if (i == 0 || hash_value != last_hash_value || last_range_value + min_interval <= range_value) {
      last_range_value = range_value;
      for (size_t c = 0; c < data_column_iterators.size(); ++c) {
        copyDataToBuilder(data_column_iterators[c], builders[c].get());
      }
    }
    last_hash_value = hash_value;
    for (size_t c = 0; c < data_column_iterators.size(); ++c) {
      ++data_column_iterators[c];
    }
  }

  return createTableFromBuilders(builders, table->schema());
}

}  // namespace l5ml_datastore
