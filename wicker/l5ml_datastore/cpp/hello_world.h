#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <string>

namespace l5ml_datastore {

static constexpr const char* some_function_key_doc = R"pydoc(
Hello
)pydoc";
pybind11::object some_function();

}  // namespace l5ml_datastore
