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
    void* _device;  // MTL::Device*
    void* _queue;   // MTL::CommandQueue*

public:
    MetalDevice();
    ~MetalDevice();

    std::map<std::string, std::vector<double>> run_kernel(
        const std::string& source,
        const std::string& kernel_name,
        int grid_size,
        const std::vector<BufferConfig>& buffer_configs);
};
