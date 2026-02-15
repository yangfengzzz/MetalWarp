#pragma once

#include <map>
#include <string>
#include <vector>

struct BufferConfig {
    std::string name;
    std::string type;              // "float", "int", "uint"
    std::vector<double> data;      // initial data (empty if not provided)
    int size = 0;                  // zero-initialized array size
    double value = 0;              // scalar value
    bool is_value = false;         // true when buffer represents a scalar
    bool is_sized = false;         // true when buffer is zero-initialized with size
};

class MetalDevice {
    struct GpuBuffer {
        void* mtl_buffer = nullptr; // MTL::Buffer*
        std::string type;           // "float", "int", "uint"
        int count = 0;              // element count (1 for scalar)
        bool is_scalar = false;
    };

    void* _device;  // MTL::Device*
    void* _queue;   // MTL::CommandQueue*
    int _next_buffer_id = 1;
    std::map<int, GpuBuffer> _gpu_buffers;

public:
    MetalDevice();
    ~MetalDevice();

    std::map<std::string, std::vector<double>> run_kernel(
        const std::string& source,
        const std::string& kernel_name,
        int grid_size,
        const std::vector<BufferConfig>& buffer_configs);

    int create_buffer(const std::string& type, int size);
    int create_buffer_with_data(const std::string& type, const std::vector<double>& data);
    int create_scalar_buffer(const std::string& type, double value);
    void upload_buffer(int buffer_id, const std::vector<double>& data);
    void set_scalar_buffer(int buffer_id, double value);
    std::vector<double> download_buffer(int buffer_id) const;
    void run_kernel_with_buffers(
        const std::string& source,
        const std::string& kernel_name,
        int grid_size,
        const std::vector<int>& buffer_ids);

    // Low-level accessors for zero-copy interop with renderer.
    void* raw_device() const;
    void* raw_buffer(int buffer_id) const;
    int buffer_count(int buffer_id) const;
    bool buffer_is_scalar(int buffer_id) const;
};
