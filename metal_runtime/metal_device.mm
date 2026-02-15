#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal_device.h"

#include <cstdint>
#include <stdexcept>

namespace {

struct PipelineBundle {
    MTL::Library* library = nullptr;
    MTL::ComputePipelineState* pipeline = nullptr;
};

PipelineBundle create_pipeline(MTL::Device* device,
                               const std::string& source,
                               const std::string& kernel_name) {
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
        std::string msg = "Failed to create pipeline";
        if (error)
            msg = error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }

    return {library, pipeline};
}

void encode_and_dispatch(MTL::CommandQueue* queue,
                         MTL::ComputePipelineState* pipeline,
                         int grid_size,
                         const std::vector<MTL::Buffer*>& buffers) {
    MTL::CommandBuffer* cmd = queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pipeline);

    for (int i = 0; i < static_cast<int>(buffers.size()); i++) {
        enc->setBuffer(buffers[i], 0, i);
    }

    MTL::Size grid = MTL::Size(grid_size, 1, 1);
    NS::UInteger w = pipeline->maxTotalThreadsPerThreadgroup();
    if (w > static_cast<NS::UInteger>(grid_size))
        w = static_cast<NS::UInteger>(grid_size);
    MTL::Size threadgroup = MTL::Size(w, 1, 1);

    enc->dispatchThreads(grid, threadgroup);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}

MTL::Buffer* create_typed_array_buffer(MTL::Device* device,
                                       const std::string& type,
                                       int count,
                                       const std::vector<double>* data) {
    if (count < 0)
        throw std::runtime_error("Buffer size must be >= 0");

    if (type == "float") {
        if (data) {
            std::vector<float> vec;
            vec.reserve(data->size());
            for (double v : *data)
                vec.push_back(static_cast<float>(v));
            return device->newBuffer(vec.data(), vec.size() * sizeof(float),
                                     MTL::ResourceStorageModeShared);
        }
        return device->newBuffer(count * sizeof(float), MTL::ResourceStorageModeShared);
    }

    if (data) {
        std::vector<int32_t> vec;
        vec.reserve(data->size());
        for (double v : *data)
            vec.push_back(static_cast<int32_t>(v));
        return device->newBuffer(vec.data(), vec.size() * sizeof(int32_t),
                                 MTL::ResourceStorageModeShared);
    }
    return device->newBuffer(count * sizeof(int32_t), MTL::ResourceStorageModeShared);
}

MTL::Buffer* create_scalar_buffer(MTL::Device* device,
                                  const std::string& type,
                                  double value) {
    if (type == "float") {
        float v = static_cast<float>(value);
        return device->newBuffer(&v, sizeof(float), MTL::ResourceStorageModeShared);
    }
    if (type == "uint") {
        uint32_t v = static_cast<uint32_t>(value);
        return device->newBuffer(&v, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    }
    int32_t v = static_cast<int32_t>(value);
    return device->newBuffer(&v, sizeof(int32_t), MTL::ResourceStorageModeShared);
}

} // namespace

MetalDevice::MetalDevice() {
    auto* device = MTL::CreateSystemDefaultDevice();
    if (!device)
        throw std::runtime_error("No Metal device found");
    _device = device;
    _queue = device->newCommandQueue();
}

MetalDevice::~MetalDevice() {
    for (auto& [id, info] : _gpu_buffers) {
        (void)id;
        static_cast<MTL::Buffer*>(info.mtl_buffer)->release();
    }
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

    PipelineBundle bundle = create_pipeline(device, source, kernel_name);

    struct BufInfo {
        MTL::Buffer* buf;
        std::string name;
        std::string type;
        int count;
        bool is_scalar;
    };
    std::vector<BufInfo> bufs;
    bufs.reserve(buffer_configs.size());

    for (const auto& cfg : buffer_configs) {
        MTL::Buffer* mtl_buf = nullptr;
        int count = 0;
        bool is_scalar = false;

        if (!cfg.data.empty()) {
            count = static_cast<int>(cfg.data.size());
            mtl_buf = create_typed_array_buffer(device, cfg.type, count, &cfg.data);
        } else if (cfg.is_sized) {
            count = cfg.size;
            mtl_buf = create_typed_array_buffer(device, cfg.type, count, nullptr);
        } else if (cfg.is_value) {
            is_scalar = true;
            count = 1;
            mtl_buf = ::create_scalar_buffer(device, cfg.type, cfg.value);
        } else {
            bundle.pipeline->release();
            bundle.library->release();
            throw std::runtime_error("Invalid buffer config for: " + cfg.name);
        }

        bufs.push_back({mtl_buf, cfg.name, cfg.type, count, is_scalar});
    }

    std::vector<MTL::Buffer*> mtl_buffers;
    mtl_buffers.reserve(bufs.size());
    for (auto& info : bufs)
        mtl_buffers.push_back(info.buf);

    encode_and_dispatch(queue, bundle.pipeline, grid_size, mtl_buffers);

    std::map<std::string, std::vector<double>> results;
    for (auto& info : bufs) {
        if (info.is_scalar)
            continue;

        std::vector<double> values;
        values.reserve(info.count);
        if (info.type == "float") {
            auto* ptr = static_cast<float*>(info.buf->contents());
            for (int i = 0; i < info.count; i++)
                values.push_back(static_cast<double>(ptr[i]));
        } else {
            auto* ptr = static_cast<int32_t*>(info.buf->contents());
            for (int i = 0; i < info.count; i++)
                values.push_back(static_cast<double>(ptr[i]));
        }
        results[info.name] = std::move(values);
    }

    for (auto& info : bufs)
        info.buf->release();
    bundle.pipeline->release();
    bundle.library->release();

    return results;
}

int MetalDevice::create_buffer(const std::string& type, int size) {
    auto* device = static_cast<MTL::Device*>(_device);
    MTL::Buffer* buf = create_typed_array_buffer(device, type, size, nullptr);
    int id = _next_buffer_id++;
    _gpu_buffers[id] = GpuBuffer{buf, type, size, false};
    return id;
}

int MetalDevice::create_buffer_with_data(const std::string& type, const std::vector<double>& data) {
    auto* device = static_cast<MTL::Device*>(_device);
    MTL::Buffer* buf = create_typed_array_buffer(device, type, static_cast<int>(data.size()), &data);
    int id = _next_buffer_id++;
    _gpu_buffers[id] = GpuBuffer{buf, type, static_cast<int>(data.size()), false};
    return id;
}

int MetalDevice::create_scalar_buffer(const std::string& type, double value) {
    auto* device = static_cast<MTL::Device*>(_device);
    MTL::Buffer* buf = ::create_scalar_buffer(device, type, value);
    int id = _next_buffer_id++;
    _gpu_buffers[id] = GpuBuffer{buf, type, 1, true};
    return id;
}

void MetalDevice::upload_buffer(int buffer_id, const std::vector<double>& data) {
    auto it = _gpu_buffers.find(buffer_id);
    if (it == _gpu_buffers.end())
        throw std::runtime_error("Unknown buffer id: " + std::to_string(buffer_id));

    auto& info = it->second;
    if (info.is_scalar)
        throw std::runtime_error("upload_buffer requires a non-scalar buffer");
    if (static_cast<int>(data.size()) != info.count)
        throw std::runtime_error("upload_buffer size mismatch");

    if (info.type == "float") {
        auto* dst = static_cast<float*>(static_cast<MTL::Buffer*>(info.mtl_buffer)->contents());
        for (int i = 0; i < info.count; i++)
            dst[i] = static_cast<float>(data[i]);
    } else {
        auto* dst = static_cast<int32_t*>(static_cast<MTL::Buffer*>(info.mtl_buffer)->contents());
        for (int i = 0; i < info.count; i++)
            dst[i] = static_cast<int32_t>(data[i]);
    }
}

void MetalDevice::set_scalar_buffer(int buffer_id, double value) {
    auto it = _gpu_buffers.find(buffer_id);
    if (it == _gpu_buffers.end())
        throw std::runtime_error("Unknown buffer id: " + std::to_string(buffer_id));

    auto& info = it->second;
    if (!info.is_scalar)
        throw std::runtime_error("set_scalar_buffer requires a scalar buffer");

    auto* buf = static_cast<MTL::Buffer*>(info.mtl_buffer);
    if (info.type == "float") {
        *static_cast<float*>(buf->contents()) = static_cast<float>(value);
    } else if (info.type == "uint") {
        *static_cast<uint32_t*>(buf->contents()) = static_cast<uint32_t>(value);
    } else {
        *static_cast<int32_t*>(buf->contents()) = static_cast<int32_t>(value);
    }
}

std::vector<double> MetalDevice::download_buffer(int buffer_id) const {
    auto it = _gpu_buffers.find(buffer_id);
    if (it == _gpu_buffers.end())
        throw std::runtime_error("Unknown buffer id: " + std::to_string(buffer_id));

    const auto& info = it->second;
    std::vector<double> out;
    out.reserve(info.count);

    auto* buf = static_cast<MTL::Buffer*>(info.mtl_buffer);
    if (info.type == "float") {
        auto* src = static_cast<float*>(buf->contents());
        for (int i = 0; i < info.count; i++)
            out.push_back(static_cast<double>(src[i]));
    } else {
        auto* src = static_cast<int32_t*>(buf->contents());
        for (int i = 0; i < info.count; i++)
            out.push_back(static_cast<double>(src[i]));
    }

    return out;
}

void MetalDevice::run_kernel_with_buffers(
    const std::string& source,
    const std::string& kernel_name,
    int grid_size,
    const std::vector<int>& buffer_ids)
{
    auto* device = static_cast<MTL::Device*>(_device);
    auto* queue = static_cast<MTL::CommandQueue*>(_queue);

    PipelineBundle bundle = create_pipeline(device, source, kernel_name);

    std::vector<MTL::Buffer*> buffers;
    buffers.reserve(buffer_ids.size());
    for (int id : buffer_ids) {
        auto it = _gpu_buffers.find(id);
        if (it == _gpu_buffers.end()) {
            bundle.pipeline->release();
            bundle.library->release();
            throw std::runtime_error("Unknown buffer id: " + std::to_string(id));
        }
        buffers.push_back(static_cast<MTL::Buffer*>(it->second.mtl_buffer));
    }

    encode_and_dispatch(queue, bundle.pipeline, grid_size, buffers);

    bundle.pipeline->release();
    bundle.library->release();
}

void* MetalDevice::raw_device() const {
    return _device;
}

void* MetalDevice::raw_buffer(int buffer_id) const {
    auto it = _gpu_buffers.find(buffer_id);
    if (it == _gpu_buffers.end())
        throw std::runtime_error("Unknown buffer id: " + std::to_string(buffer_id));
    return it->second.mtl_buffer;
}

int MetalDevice::buffer_count(int buffer_id) const {
    auto it = _gpu_buffers.find(buffer_id);
    if (it == _gpu_buffers.end())
        throw std::runtime_error("Unknown buffer id: " + std::to_string(buffer_id));
    return it->second.count;
}

bool MetalDevice::buffer_is_scalar(int buffer_id) const {
    auto it = _gpu_buffers.find(buffer_id);
    if (it == _gpu_buffers.end())
        throw std::runtime_error("Unknown buffer id: " + std::to_string(buffer_id));
    return it->second.is_scalar;
}
