#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;

namespace l5ml_datastore {


void some_function()
{
    std::cout << "Hello, World is !" << std::endl;
}

}