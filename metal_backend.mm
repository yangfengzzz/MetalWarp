// Metal GPU runtime — pybind11 extension for running .metal kernels from Python.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <cstdint>

namespace py = pybind11;

class MetalDevice {
    MTL::Device* device;
    MTL::CommandQueue* queue;

public:
    MetalDevice() {
        device = MTL::CreateSystemDefaultDevice();
        if (!device)
            throw std::runtime_error("No Metal device found");
        queue = device->newCommandQueue();
    }

    ~MetalDevice() {
        queue->release();
        device->release();
    }

    py::dict run_kernel(const std::string& source,
                        const std::string& kernel_name,
                        int grid_size,
                        py::list buffer_configs)
    {
        // ── compile source → library ────────────────────────────────────
        NS::Error* error = nullptr;
        auto src = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
        auto opts = MTL::CompileOptions::alloc()->init();
        MTL::Library* library = device->newLibrary(src, opts, &error);
        opts->release();
        if (!library) {
            std::string msg = "Metal compile error";
            if (error)
                msg = error->localizedDescription()->utf8String();
            throw std::runtime_error(msg);
        }

        // ── get function → pipeline ─────────────────────────────────────
        auto fn_name = NS::String::string(kernel_name.c_str(), NS::UTF8StringEncoding);
        MTL::Function* func = library->newFunction(fn_name);
        if (!func) {
            library->release();
            throw std::runtime_error("Kernel not found: " + kernel_name);
        }

        MTL::ComputePipelineState* pipeline = device->newComputePipelineState(func, &error);
        func->release();
        if (!pipeline) {
            library->release();
            throw std::runtime_error("Failed to create pipeline");
        }

        // ── allocate buffers ────────────────────────────────────────────
        struct BufInfo {
            MTL::Buffer* buf;
            std::string name;
            std::string type;
            int count;       // number of elements (0 for scalar)
            bool is_scalar;
        };
        std::vector<BufInfo> bufs;

        for (auto item : buffer_configs) {
            py::dict cfg = item.cast<py::dict>();
            std::string name = cfg["name"].cast<std::string>();
            std::string type = cfg["type"].cast<std::string>();

            MTL::Buffer* mtl_buf = nullptr;
            int count = 0;
            bool is_scalar = false;

            if (cfg.contains("data")) {
                // Array with initial data
                py::list data = cfg["data"].cast<py::list>();
                count = (int)data.size();

                if (type == "float") {
                    std::vector<float> vec;
                    for (auto v : data) vec.push_back(v.cast<float>());
                    mtl_buf = device->newBuffer(vec.data(), vec.size() * sizeof(float),
                                                MTL::ResourceStorageModeShared);
                } else {
                    // int or uint
                    std::vector<int32_t> vec;
                    for (auto v : data) vec.push_back(v.cast<int32_t>());
                    mtl_buf = device->newBuffer(vec.data(), vec.size() * sizeof(int32_t),
                                                MTL::ResourceStorageModeShared);
                }
            } else if (cfg.contains("size")) {
                // Zero-initialized array
                count = cfg["size"].cast<int>();
                size_t byte_size = count * (type == "float" ? sizeof(float) : sizeof(int32_t));
                mtl_buf = device->newBuffer(byte_size, MTL::ResourceStorageModeShared);
            } else if (cfg.contains("value")) {
                // Scalar value
                is_scalar = true;
                if (type == "float") {
                    float val = cfg["value"].cast<float>();
                    mtl_buf = device->newBuffer(&val, sizeof(float),
                                                MTL::ResourceStorageModeShared);
                } else if (type == "uint") {
                    uint32_t val = cfg["value"].cast<uint32_t>();
                    mtl_buf = device->newBuffer(&val, sizeof(uint32_t),
                                                MTL::ResourceStorageModeShared);
                } else {
                    int32_t val = cfg["value"].cast<int32_t>();
                    mtl_buf = device->newBuffer(&val, sizeof(int32_t),
                                                MTL::ResourceStorageModeShared);
                }
            }

            bufs.push_back({mtl_buf, name, type, count, is_scalar});
        }

        // ── encode and dispatch ─────────────────────────────────────────
        MTL::CommandBuffer* cmd = queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pipeline);

        for (int i = 0; i < (int)bufs.size(); i++) {
            enc->setBuffer(bufs[i].buf, 0, i);
        }

        MTL::Size grid = MTL::Size(grid_size, 1, 1);
        NS::UInteger w = pipeline->maxTotalThreadsPerThreadgroup();
        if (w > (NS::UInteger)grid_size) w = grid_size;
        MTL::Size threadgroup = MTL::Size(w, 1, 1);

        enc->dispatchThreads(grid, threadgroup);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();

        // ── read back results ───────────────────────────────────────────
        py::dict results;
        for (auto& info : bufs) {
            if (info.is_scalar)
                continue;  // don't return scalar params

            py::list values;
            if (info.type == "float") {
                auto* ptr = (float*)info.buf->contents();
                for (int i = 0; i < info.count; i++)
                    values.append(ptr[i]);
            } else {
                auto* ptr = (int32_t*)info.buf->contents();
                for (int i = 0; i < info.count; i++)
                    values.append(ptr[i]);
            }
            results[py::cast(info.name)] = values;
        }

        // ── cleanup ─────────────────────────────────────────────────────
        for (auto& info : bufs)
            info.buf->release();
        pipeline->release();
        library->release();

        return results;
    }
};

PYBIND11_MODULE(metal_backend, m) {
    m.doc() = "Metal GPU runtime for running .metal kernels";
    py::class_<MetalDevice>(m, "MetalDevice")
        .def(py::init<>())
        .def("run_kernel", &MetalDevice::run_kernel,
             py::arg("source"), py::arg("kernel_name"),
             py::arg("grid_size"), py::arg("buffer_configs"));
}
