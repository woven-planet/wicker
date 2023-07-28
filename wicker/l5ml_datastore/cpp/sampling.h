#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <string>

namespace l5ml_datastore {

static constexpr const char* sample_by_range_key_doc = R"pydoc(
Take samples from the given pyarrow_table and return a new table with the same schema,
so that in the new table, there is a minimum difference of min_interval between the range keys of
two consecutive samples that share the same hash_key.
For example with a dataset like this:
car_id  timestamp_ns  data
a        10000        <data1>
a        10050        <data2>
a        10100        <data3>
a        10150        <data4>
b        10150        <data5>
Applying sampleByRangeKey(dataset, "car_id", "timestamp_ns", 100) will return
car_id  timestamp_ns  data
a        10000        <data1>
a        10100        <data3>
b        10150        <data5>
)pydoc";
pybind11::object sampleByRangeKey(
    pybind11::object pyarrow_table,
    const std::string& hash_key,
    const std::string& range_key,
    const int64_t min_interval);

}  // namespace l5ml_datastore
