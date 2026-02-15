#pragma once

#include <vector>

class MetalRenderer {
    void* _device;    // MTL::Device*
    void* _queue;     // MTL::CommandQueue*
    void* _layer;     // CA::MetalLayer*
    void* _pipeline;  // MTL::RenderPipelineState*
    void* _window;    // NSWindow*
    int   _width;
    int   _height;
    bool  _closed;

public:
    MetalRenderer(int width = 800, int height = 800);
    ~MetalRenderer();

    void render_frame(const std::vector<float>& pos_x, const std::vector<float>& pos_y,
                      const std::vector<float>& vel_x, const std::vector<float>& vel_y);
    bool poll_events();   // returns false if window closed
    bool is_open() const;
};
