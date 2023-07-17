#include "arrow_util.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <arrow/python/pyarrow.h>
#pragma GCC diagnostic pop
#include <arrow/table.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace l5ml_datastore {

std::vector<arrow::util::string_view> chunkedArrayToStringViewVector(
    const arrow::ChunkedArray& carray) {
  std::vector<arrow::util::string_view> output;
  output.reserve(carray.length());
  for (const std::shared_ptr<arrow::Array>& chunk : carray.chunks()) {
    const arrow::BinaryArray* data = dynamic_cast<const arrow::BinaryArray*>(chunk.get());
    CHECK(data);
    for (ssize_t i = 0; i < data->length(); ++i) {
      output.push_back(data->GetView(i));
    }
  }
  return output;
}

void copyDataToBuilder(
    const arrow::Array* chunk, ssize_t index_in_chunk, arrow::ArrayBuilder* builder);

// Copy a numeric value (integers or various sizes, floats, double) from the source, defined by an
// array ("chunk") and the position in the array ("index_in_chunk") to "builder".
template <class ArrowType>
void copyNumericValue(
    const arrow::Array* chunk, ssize_t index_in_chunk, arrow::ArrayBuilder* builder) {
  dynamic_cast<arrow::NumericBuilder<ArrowType>*>(builder)->Append(
      dynamic_cast<const arrow::NumericArray<ArrowType>*>(chunk)->Value(index_in_chunk));
}

template <class ArrayType, class BuilderType>
void copyListValue(
    const arrow::Array* chunk, ssize_t index_in_chunk, arrow::ArrayBuilder* builder) {
  const ArrayType* list_array = dynamic_cast<const ArrayType*>(chunk);
  BuilderType* list_builder = dynamic_cast<BuilderType*>(builder);
  CHECK(list_builder);
  arrow::ArrayBuilder* value_builder = list_builder->value_builder();
  CHECK(value_builder);
  list_builder->Append();
  const ssize_t offset = list_array->value_offset(index_in_chunk);
  for (ssize_t i = 0; i < list_array->value_length(index_in_chunk); ++i) {
    copyDataToBuilder(list_array->values().get(), offset + i, value_builder);
  }
}

void copyDataToBuilder(
    const arrow::Array* chunk, ssize_t index_in_chunk, arrow::ArrayBuilder* builder) {
  if (chunk->IsNull(index_in_chunk)) {
    builder->AppendNull();
    return;
  }
  switch (builder->type()->id()) {
    case arrow::Type::BOOL: {
      dynamic_cast<arrow::BooleanBuilder*>(builder)->Append(
          dynamic_cast<const arrow::BooleanArray*>(chunk)->Value(index_in_chunk));
      return;
    }
    case arrow::Type::UINT8:
      return copyNumericValue<arrow::UInt8Type>(chunk, index_in_chunk, builder);
    case arrow::Type::INT8:
      return copyNumericValue<arrow::Int8Type>(chunk, index_in_chunk, builder);
    case arrow::Type::UINT16:
      return copyNumericValue<arrow::UInt16Type>(chunk, index_in_chunk, builder);
    case arrow::Type::INT16:
      return copyNumericValue<arrow::Int16Type>(chunk, index_in_chunk, builder);
    case arrow::Type::UINT32:
      return copyNumericValue<arrow::UInt32Type>(chunk, index_in_chunk, builder);
    case arrow::Type::INT32:
      return copyNumericValue<arrow::Int32Type>(chunk, index_in_chunk, builder);
    case arrow::Type::UINT64:
      return copyNumericValue<arrow::UInt64Type>(chunk, index_in_chunk, builder);
    case arrow::Type::INT64:
      return copyNumericValue<arrow::Int64Type>(chunk, index_in_chunk, builder);
    case arrow::Type::FLOAT:
      return copyNumericValue<arrow::FloatType>(chunk, index_in_chunk, builder);
    case arrow::Type::DOUBLE:
      return copyNumericValue<arrow::DoubleType>(chunk, index_in_chunk, builder);
    case arrow::Type::STRING:
    case arrow::Type::BINARY: {
      arrow::util::string_view data =
          dynamic_cast<const arrow::BinaryArray*>(chunk)->GetView(index_in_chunk);
      dynamic_cast<arrow::BinaryBuilder*>(builder)->Append(data);
      return;
    }
    case arrow::Type::LIST: {
      return copyListValue<arrow::ListArray, arrow::ListBuilder>(chunk, index_in_chunk, builder);
    }
    case arrow::Type::FIXED_SIZE_LIST: {
      return copyListValue<arrow::FixedSizeListArray, arrow::FixedSizeListBuilder>(
          chunk, index_in_chunk, builder);
    }
    case arrow::Type::STRUCT: {
      const arrow::StructArray* struct_array = dynamic_cast<const arrow::StructArray*>(chunk);
      arrow::StructBuilder* struct_builder = dynamic_cast<arrow::StructBuilder*>(builder);
      CHECK(struct_builder);
      struct_builder->Append();
      for (int i = 0; i < struct_builder->num_fields(); ++i) {
        copyDataToBuilder(
            struct_array->field(i).get(), index_in_chunk, struct_builder->field_builder(i));
      }
      return;
    }
    default:
      break;
  }
  FATAL("Type " + builder->type()->ToString() + " not handled yet.");
}

void copyDataToBuilder(const ChunkedArrayIterator& it, arrow::ArrayBuilder* builder) {
  if (it.empty()) {
    builder->AppendNull();
    return;
  }
  copyDataToBuilder(it.chunk(), it.index_in_chunk(), builder);
}

void copyListDataToBuilder(
    const std::vector<ChunkedArrayIterator>& its, arrow::ArrayBuilder* builder) {
  arrow::FixedSizeListBuilder* list_builder = dynamic_cast<arrow::FixedSizeListBuilder*>(builder);
  CHECK(list_builder);
  arrow::ArrayBuilder* value_builder = list_builder->value_builder();
  CHECK(value_builder);
  list_builder->Append();
  for (size_t i = 0; i < its.size(); ++i) {
    copyDataToBuilder(its[i], value_builder);
  }
}

py::object createTableFromBuilders(
    std::vector<std::shared_ptr<arrow::ArrayBuilder>>& builders,
    const std::shared_ptr<arrow::Schema>& schema) {
  std::vector<arrow::ArrayBuilder*> all_builders(builders.size());
  for (size_t i = 0; i < builders.size(); ++i) {
    all_builders[i] = builders[i].get();
  }
  return createTableFromBuilders(all_builders, schema);
}

py::object createTableFromBuilders(
    std::vector<arrow::ArrayBuilder*>& builders, const std::shared_ptr<arrow::Schema>& schema) {
  // Finalize the arrays and create the final arrow::Table
  std::vector<std::shared_ptr<arrow::ChunkedArray>> chunked_columns;

  for (size_t i = 0; i < builders.size(); ++i) {
    std::shared_ptr<arrow::Array> array;
    builders[i]->Finish(&array);
    chunked_columns.push_back(std::make_shared<arrow::ChunkedArray>(array));
  }

  std::shared_ptr<arrow::Table> new_table = arrow::Table::Make(schema, chunked_columns);
  PyObject* table_obj = arrow::py::wrap_table(new_table);
  return py::reinterpret_steal<py::object>(table_obj);
}

}  // namespace l5ml_datastore
