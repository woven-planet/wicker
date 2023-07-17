#pragma once

#include <arrow/array/array_base.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/chunked_array.h>
#include <arrow/type.h>
#include <stdint.h>
#include "util.h"

// Some functions to work with Arrow arrays.

namespace l5ml_datastore {

// Iterator simplifying interactions with Arrow's chunked arrays. The iterator supports increment
// decrement operations.
class ChunkedArrayIterator {
public:
  ChunkedArrayIterator()
      : chunked_array_(nullptr), index_(0), chunk_index_(0), index_in_chunk_(0), chunk_length_(0) {}
  ChunkedArrayIterator(const arrow::ChunkedArray* chunked_array)
      : chunked_array_(chunked_array),
        index_(0),
        chunk_index_(0),
        index_in_chunk_(0),
        chunk_length_(0) {}
  ChunkedArrayIterator& operator++() {
    index_++;
    index_in_chunk_++;
    if (index_in_chunk_ >= chunk_length_) {
      if (chunk_index_ + 1 < static_cast<ssize_t>(chunked_array_->chunks().size())) {
        index_in_chunk_ -= chunk_length_;
        ++chunk_index_;
        chunk_length_ = chunked_array_->chunks()[chunk_index_]->length();
      }
    }
    return *this;
  }
  ChunkedArrayIterator& operator--() {
    index_--;
    index_in_chunk_--;
    if (index_in_chunk_ < 0) {
      if (chunk_index_ > 0) {
        index_in_chunk_ += chunk_length_;
        --chunk_index_;
        chunk_length_ = chunked_array_->chunks()[chunk_index_]->length();
      }
    }
    return *this;
  }
  // Return true if the iterator is not initialized with an array.
  bool empty() const { return !chunked_array_; }

  // Return the index of the current element in the chunked array.
  ssize_t index() const { return index_; }

  // The data in the chunked array is stored in multiple arrays. This function returns the array
  // that matches the current index.
  // This should only be called when the iterator is valid.
  const arrow::Array* chunk() const { return chunked_array_->chunks()[chunk_index_].get(); }

  // index_in_chunk returns the index of the current element in the current chunk.
  ssize_t index_in_chunk() const { return index_in_chunk_; }

  // Return true if the value at the current index is null.
  bool isNull() const { return chunk()->IsNull(index_in_chunk_); }

private:
  const arrow::ChunkedArray* chunked_array_;
  ssize_t index_;
  ssize_t chunk_index_, index_in_chunk_;
  ssize_t chunk_length_;
};

// Copy a chunked array of numerical values (integers of any length/float/double) to a std::vector.
template <class T>
std::vector<typename T::c_type> chunkedArrayToNumericalVector(const arrow::ChunkedArray& carray) {
  std::vector<typename T::c_type> output;
  output.reserve(carray.length());
  for (const std::shared_ptr<arrow::Array>& chunk : carray.chunks()) {
    const arrow::NumericArray<T>* data = dynamic_cast<const arrow::NumericArray<T>*>(chunk.get());
    CHECK(data);
    std::copy(data->raw_values(), data->raw_values() + data->length(), std::back_inserter(output));
  }
  return output;
}

// Copy a chunked array of strings (or binary data) to an std::vector.
std::vector<arrow::util::string_view> chunkedArrayToStringViewVector(
    const arrow::ChunkedArray& carray);

// Copy the value referenced by the iterator "src" to the builder "builder".
// This function assumes that both it and builder use the same underlying type.
void copyDataToBuilder(const ChunkedArrayIterator& src, arrow::ArrayBuilder* builder);

// Copy the list of values referenced by the iterators "srcs" to the builder "builder".
// This function assumes that builder is a FixedSizedListBuilder, that its.size() matches the
// size of the fixed list, and that the type of elements in "srcs" and "builder" is the same.
void copyListDataToBuilder(
    const std::vector<ChunkedArrayIterator>& srcs, arrow::ArrayBuilder* builder);

// Create a pyarrow table from a vector of ArrayBuilder and a schema.
pybind11::object createTableFromBuilders(
    std::vector<std::shared_ptr<arrow::ArrayBuilder>>& builders,
    const std::shared_ptr<arrow::Schema>& schema);

pybind11::object createTableFromBuilders(
    std::vector<arrow::ArrayBuilder*>& builders, const std::shared_ptr<arrow::Schema>& schema);

}  // namespace l5ml_datastore
