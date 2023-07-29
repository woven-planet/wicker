#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace l5ml_datastore {

static constexpr const char* temporal_windowing_doc = R"pydoc(
CellSpecification, ColumnHistorySpecification and WindowSpecification are structures used in
applyTemporalWindowSpecification to apply windowing functions to an Arrow table.

The current implementation expects the table to have two primary keys: a "hash key" (of type
string) and a "range key" (of type int64).
The applyTemporalWindowSpecification will look at all the rows in the table, find the rows
that match the specification given in the WindowSpecification "spec" and return a new table with
the matching rows. The WindowSpecification allows gathering data from multiple rows based on
timestamp relative to the target row.
For example suppose we have 100 rows containing pointcloud data every 100ms, the windowing
function allows to build 90 rows where each row contains the vector of the 10 point clouds
immediately preceding the row's timestamp. This is a very simple case, a lot more options are
possible.
The WindowSpecification is a vector of ColumnHistorySpecification, each describing what needs to
happen for each column that is returned.
The ColumnHistorySpecification contains a list of CellSpecification.
Each CellSpecification will fetch one return value for that column.

An example is probably the best way to explain what's happening. Let's consider the following
case (using python syntax here to express the classes):

window = WindowSpecification(
  column_specs=[
    ColumnHistorySpecification(
      column_name="image",
      spec=[
        CellSpecification(min_offset=-220, max_offset=-180, is_required=True),
        CellSpecification(min_offset=-120, max_offset=-80, is_required=True),
        CellSpecification(min_offset=-20, max_offset=20, is_required=True),
      ],
      results_as_list=True,
    ),
    ColumnHistorySpecification(
      column_name="detections",
      spec=[
        CellSpecification(min_offset=0, max_offset=0, is_required=True),
      ],
      results_as_list=False,
    ),
  ]
)

And consider the following table:
    car_id  timestamp_ms  image      detections
1   car1      1230000      <img1>    <detection1>
2   car1      1230100      <img2>    <detection2>
3   car1      1230200      <img3>    <detection3>
4   car2      1230000      <img4>    <detection4>
5   car2      1230100      <img5>    <detection5>
6   car2      1230200      <img6>    <detection6>
7   car2      1230300      <img7>    <detection7>

If we apply the windowing function
result = applyTemporalWindowSpecification(table, window, "car_id", "timestamp_ms")

We get the following result:
car_id  timestamp_ms  image                   detections
car1      1230200     [<img1>,<img2>,<img3>] <detection3>
car2      1230200     [<img4>,<img5>,<img6>] <detection6>
car2      1230300     [<img5>,<img6>,<img7>] <detection7>

The WindowSpecification indicated that we wanted to extract data from 2 columns:
"image" and "detections".
For the "image" column, we want to get an array of 3 values:
- a value that is between -220ms and -180ms from the current row being considered,
- a value that is between -120ms and -80ms from the current row being considered
- and a value between -20ms and 20ms from the current row being considered
For the "detections" column we only want to get one value, the value at the current row.
If we look at the original table, rows 1 and 2 don't match, there is not enough history.
Row 3 works, and we can get the images from rows 1,2,3 to fulfill the request.
Rows 4,5 don't work, not enough history (we disregard any row that does not have the same hash
key, here "car_id").
Rows 6 and 7 have enough history and the timestamps match the given ranges.
)pydoc";

static constexpr const char* cell_specification_doc = R"pydoc(
  Match a cell relative to the current row being considered in the windowing operation such that
    current_row.range_key + min_offset <= cell.range_key <= current_row.range_key + max_offset
  If is_required is False then, if no match is found, a None value is returned.
  If is_required is True and no match is found, the current row is skipped.
)pydoc";
struct CellSpecification {
  int64_t min_offset;
  int64_t max_offset;
  bool is_required = false;
};

static constexpr const char* column_history_specification_doc = R"pydoc(
  Match several cells relative to the current row  being considered in the windowing operation
  and aggregates them into a vector.
)pydoc";
struct ColumnHistorySpecification {
  // Column to match and extract data from.
  std::string column_name;
  // Specification of all the cells to match and gather in the current column.
  std::vector<CellSpecification> spec;
  // If spec.size()==1, the data can be returned as a vector of size 1 or an item of the original
  // type depending on the value of this flag.
  bool results_as_list = false;
  ColumnHistorySpecification& setResultsAsList(bool value) {
    results_as_list = value;
    return *this;
  }
};

struct WindowSpecification {
  // List of all the columns that we need to match/gather data from and what needs to be done with
  // each column.
  std::vector<ColumnHistorySpecification> column_specs;
};

static constexpr const char* apply_temporal_window_specification_doc = R"pydoc(
Apply the windowing specification to the given pyarrow Table and return a new pyarrow Table with
the results.
hash_key must be the name of a primary column of type string.
range_key must be the name of a primary column of type int64.
)pydoc";
pybind11::object applyTemporalWindowSpecification(
    pybind11::object pyarrow_table,
    const WindowSpecification& spec,
    const std::string& hash_key,
    const std::string& range_key);

}  // namespace l5ml_datastore
