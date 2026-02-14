#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal_device.h"

#include <cstdint>
#include <stdexcept>

MetalDevice::MetalDevice() {
    auto* device = MTL::CreateSystemDefaultDevice();
    if (!device)
        throw std::runtime_error("No Metal device found");
    _device = device;
    _queue = device->newCommandQueue();
}

MetalDevice::~MetalDevice() {
    static_cast<MTL::CommandQueue*>(_queue)->release();
    static_cast<MTL::Device*>(_device)->release();
}

std::map<std::string, std::vector<double>> MetalDevice::run_kernel(
    const std::string& source,
    const std::string& kernel_name,
    int grid_size,
    const std::vector<BufferConfig>& buffer_configs)
{
    auto* device = static_cast<MTL::Device*>(_device);
    auto* queue = static_cast<MTL::CommandQueue*>(_queue);

    // ── compile source -> library ────────────────────────────────────
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

    // ── get function -> pipeline ─────────────────────────────────────
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

    for (auto& cfg : buffer_configs) {
        MTL::Buffer* mtl_buf = nullptr;
        int count = 0;
        bool is_scalar = false;

        if (!cfg.data.empty()) {
            // Array with initial data
            count = (int)cfg.data.size();
            if (cfg.type == "float") {
                std::vector<float> vec;
                for (auto v : cfg.data) vec.push_back((float)v);
                mtl_buf = device->newBuffer(vec.data(), vec.size() * sizeof(float),
                                            MTL::ResourceStorageModeShared);
            } else {
                std::vector<int32_t> vec;
                for (auto v : cfg.data) vec.push_back((int32_t)v);
                mtl_buf = device->newBuffer(vec.data(), vec.size() * sizeof(int32_t),
                                            MTL::ResourceStorageModeShared);
            }
        } else if (cfg.is_sized) {
            // Zero-initialized array
            count = cfg.size;
            size_t byte_size = count * (cfg.type == "float" ? sizeof(float) : sizeof(int32_t));
            mtl_buf = device->newBuffer(byte_size, MTL::ResourceStorageModeShared);
        } else if (cfg.is_value) {
            // Scalar value
            is_scalar = true;
            if (cfg.type == "float") {
                float val = (float)cfg.value;
                mtl_buf = device->newBuffer(&val, sizeof(float),
                                            MTL::ResourceStorageModeShared);
            } else if (cfg.type == "uint") {
                uint32_t val = (uint32_t)cfg.value;
                mtl_buf = device->newBuffer(&val, sizeof(uint32_t),
                                            MTL::ResourceStorageModeShared);
            } else {
                int32_t val = (int32_t)cfg.value;
                mtl_buf = device->newBuffer(&val, sizeof(int32_t),
                                            MTL::ResourceStorageModeShared);
            }
        }

        bufs.push_back({mtl_buf, cfg.name, cfg.type, count, is_scalar});
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
    std::map<std::string, std::vector<double>> results;
    for (auto& info : bufs) {
        if (info.is_scalar)
            continue;

        std::vector<double> values;
        if (info.type == "float") {
            auto* ptr = (float*)info.buf->contents();
            for (int i = 0; i < info.count; i++)
                values.push_back((double)ptr[i]);
        } else {
            auto* ptr = (int32_t*)info.buf->contents();
            for (int i = 0; i < info.count; i++)
                values.push_back((double)ptr[i]);
        }
        results[info.name] = std::move(values);
    }

    // ── cleanup ─────────────────────────────────────────────────────
    for (auto& info : bufs)
        info.buf->release();
    pipeline->release();
    library->release();

    return results;
}
