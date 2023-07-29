#pragma once

#include <pybind11/pybind11.h>

// General utility functions/macros.
// These are pretty basic, but we have a very small code base :)

// Helper macros to transform the line numbers below into strings.
#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

// Verify that __v__ is true. If not, raise an exception that can be caught by python, with the
// file/line number of the failing statement.
#define CHECK(__v__)                                                                       \
  if (!(__v__)) {                                                                          \
    throw pybind11::value_error(__FILE__ ":" STRINGIFY(__LINE__) ":CHECK failed " #__v__); \
  }

// Same as CHECK, but also outputs an extra message.
#define CHECK_MSG(__v__, __msg__)                                                              \
  if (!(__v__)) {                                                                              \
    throw pybind11::value_error(                                                               \
        __FILE__ ":" STRINGIFY(__LINE__) ":CHECK failed " #__v__ ": " + std::string(__msg__)); \
  }

// Raise an exception that can be caught by python, with the file/line number of the FATAL statement
// location and an error message.
#define FATAL(__msg__) \
  throw pybind11::value_error(std::string(__FILE__ ":" STRINGIFY(__LINE__) ":") + (__msg__))
