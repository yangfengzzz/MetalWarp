#include "metal_device.h"
#include "metal_renderer.h"

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

static void run_kernel_with_buffers_wrapper(MetalDevice& self,
                                            const std::string& source,
                                            const std::string& kernel_name,
                                            int grid_size,
                                            py::list buffer_ids)
{
    std::vector<int> ids;
    ids.reserve(py::len(buffer_ids));
    for (auto item : buffer_ids) {
        ids.push_back(item.cast<int>());
    }
    self.run_kernel_with_buffers(source, kernel_name, grid_size, ids);
}

PYBIND11_MODULE(metal_backend, m) {
    m.doc() = "Metal GPU runtime for running .metal kernels";
    py::class_<MetalDevice>(m, "MetalDevice")
        .def(py::init<>())
        .def("run_kernel", &run_kernel_wrapper,
             py::arg("source"), py::arg("kernel_name"),
             py::arg("grid_size"), py::arg("buffer_configs"))
        .def("create_buffer", &MetalDevice::create_buffer,
             py::arg("type"), py::arg("size"))
        .def("create_buffer_with_data", &MetalDevice::create_buffer_with_data,
             py::arg("type"), py::arg("data"))
        .def("create_scalar_buffer", &MetalDevice::create_scalar_buffer,
             py::arg("type"), py::arg("value"))
        .def("upload_buffer", &MetalDevice::upload_buffer,
             py::arg("buffer_id"), py::arg("data"))
        .def("set_scalar_buffer", &MetalDevice::set_scalar_buffer,
             py::arg("buffer_id"), py::arg("value"))
        .def("download_buffer", &MetalDevice::download_buffer,
             py::arg("buffer_id"))
        .def("run_kernel_with_buffers", &run_kernel_with_buffers_wrapper,
             py::arg("source"), py::arg("kernel_name"),
             py::arg("grid_size"), py::arg("buffer_ids"));

    py::class_<MetalRenderer>(m, "MetalRenderer")
        .def(py::init<int, int>(), py::arg("width") = 800, py::arg("height") = 800)
        .def(py::init<MetalDevice&, int, int>(),
             py::arg("device"), py::arg("width") = 800, py::arg("height") = 800)
        .def("render_frame", [](MetalRenderer& self,
                                py::list px, py::list py_list,
                                py::list vx, py::list vy) {
            std::vector<float> pos_x, pos_y, vel_x, vel_y;
            for (auto v : px)      pos_x.push_back(v.cast<float>());
            for (auto v : py_list) pos_y.push_back(v.cast<float>());
            for (auto v : vx)      vel_x.push_back(v.cast<float>());
            for (auto v : vy)      vel_y.push_back(v.cast<float>());
            self.render_frame(pos_x, pos_y, vel_x, vel_y);
        })
        .def("render_frame_from_buffers", &MetalRenderer::render_frame_from_buffers,
             py::arg("device"),
             py::arg("pos_x_buffer_id"), py::arg("pos_y_buffer_id"),
             py::arg("vel_x_buffer_id"), py::arg("vel_y_buffer_id"))
        .def("poll_events", &MetalRenderer::poll_events)
        .def("is_open", &MetalRenderer::is_open);
}
