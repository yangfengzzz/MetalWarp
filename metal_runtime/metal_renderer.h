#pragma once

#include <vector>

class MetalDevice;

class MetalRenderer {
    void* _device;    // MTL::Device*
    void* _queue;     // MTL::CommandQueue*
    void* _layer;     // CA::MetalLayer*
    void* _pipeline;  // MTL::RenderPipelineState*
    void* _window;    // NSWindow*
    int   _width;
    int   _height;
    bool  _closed;
    bool  _owns_device;

public:
    MetalRenderer(int width = 800, int height = 800);
    MetalRenderer(MetalDevice& device, int width = 800, int height = 800);
    ~MetalRenderer();

    void render_frame(const std::vector<float>& pos_x, const std::vector<float>& pos_y,
                      const std::vector<float>& vel_x, const std::vector<float>& vel_y);
    void render_frame_from_buffers(MetalDevice& device,
                                   int pos_x_buffer_id, int pos_y_buffer_id,
                                   int vel_x_buffer_id, int vel_y_buffer_id);
    bool poll_events();   // returns false if window closed
    bool is_open() const;
};
