#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sampling.h"
#include "temporal_windowing.h"
#include "hello_world.h"

namespace py = pybind11;

namespace l5ml_datastore {

// If you modify the APIs here, please update the cpp_extensions.pyi file to reflect the changes.
PYBIND11_MODULE(cpp_extensions, m) {
  m.doc() = R"pbdoc(
        This module implements various functions to manipulate pyarrow Tables.
    )pbdoc" +
            std::string(temporal_windowing_doc);

  py::class_<CellSpecification>(m, "CellSpecification", py::module_local())
      .def(
          py::init<int64_t, int64_t, bool>(),
          py::arg("min_offset"),
          py::arg("max_offset"),
          py::arg("is_required"),
          py::doc(cell_specification_doc))
      .def_readonly("min_offset", &CellSpecification::min_offset)
      .def_readonly("max_offset", &CellSpecification::max_offset)
      .def_readonly("is_required", &CellSpecification::is_required);

  py::class_<ColumnHistorySpecification>(m, "ColumnHistorySpecification", py::module_local())
      .def(
          py::init<std::string, std::vector<CellSpecification>, bool>(),
          py::arg("column_name"),
          py::arg("spec"),
          py::arg("results_as_list") = false,
          py::doc(column_history_specification_doc))
      .def_readonly("column_name", &ColumnHistorySpecification::column_name)
      .def_readonly("spec", &ColumnHistorySpecification::spec)
      .def_readonly("results_as_list", &ColumnHistorySpecification::results_as_list)
      .def("set_results_as_list", &ColumnHistorySpecification::setResultsAsList);

  py::class_<WindowSpecification>(m, "WindowSpecification", py::module_local())
      .def(py::init<std::vector<ColumnHistorySpecification>>(), py::arg("column_specs"))
      .def_readonly("column_specs", &WindowSpecification::column_specs);

  m.def(
      "apply_temporal_window_specification",
      &applyTemporalWindowSpecification,
      py::doc(apply_temporal_window_specification_doc));

  m.def("sample_by_range_key", &sampleByRangeKey, py::doc(sample_by_range_key_doc));

  m.def("some_function", &some_function, py::doc(some_function_key_doc));

}

}  // namespace l5ml_datastore
