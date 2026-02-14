#include "metal_device.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

static py::dict run_kernel_wrapper(MetalDevice& self,
                                   const std::string& source,
                                   const std::string& kernel_name,
                                   int grid_size,
                                   py::list buffer_configs)
{
    // Convert py::list of py::dict -> vector<BufferConfig>
    std::vector<BufferConfig> configs;
    for (auto item : buffer_configs) {
        py::dict cfg = item.cast<py::dict>();
        BufferConfig bc;
        bc.name = cfg["name"].cast<std::string>();
        bc.type = cfg["type"].cast<std::string>();

        if (cfg.contains("data")) {
            py::list data = cfg["data"].cast<py::list>();
            for (auto v : data)
                bc.data.push_back(v.cast<double>());
        } else if (cfg.contains("size")) {
            bc.size = cfg["size"].cast<int>();
            bc.is_sized = true;
        } else if (cfg.contains("value")) {
            bc.value = cfg["value"].cast<double>();
            bc.is_value = true;
        }

        configs.push_back(std::move(bc));
    }

    // Call the C++ implementation
    auto results = self.run_kernel(source, kernel_name, grid_size, configs);

    // Convert map<string, vector<double>> -> py::dict of py::list
    py::dict out;
    for (auto& [name, values] : results) {
        py::list lst;
        for (double v : values)
            lst.append(v);
        out[py::cast(name)] = lst;
    }
    return out;
}

PYBIND11_MODULE(metal_backend, m) {
    m.doc() = "Metal GPU runtime for running .metal kernels";
    py::class_<MetalDevice>(m, "MetalDevice")
        .def(py::init<>())
        .def("run_kernel", &run_kernel_wrapper,
             py::arg("source"), py::arg("kernel_name"),
             py::arg("grid_size"), py::arg("buffer_configs"));
}
