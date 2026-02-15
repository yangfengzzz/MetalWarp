#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#import <Cocoa/Cocoa.h>

#include "metal_device.h"
#include "metal_renderer.h"

#include <stdexcept>

// -- Window delegate to detect close -----------------------------------------

@interface RendererWindowDelegate : NSObject <NSWindowDelegate>
@property (nonatomic) bool* closedFlag;
@end

@implementation RendererWindowDelegate
- (void)windowWillClose:(NSNotification*)notification {
    if (self.closedFlag) *self.closedFlag = true;
}
@end

// -- Embedded Metal shaders ---------------------------------------------------

static const char* kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float  point_size [[point_size]];
    float  speed;
};

vertex VertexOut vertex_main(const device float* pos_x [[buffer(0)]],
                             const device float* pos_y [[buffer(1)]],
                             const device float* vel_x [[buffer(2)]],
                             const device float* vel_y [[buffer(3)]],
                             const device float* uniforms [[buffer(4)]],
                             uint vid [[vertex_id]])
{
    float px = pos_x[vid];
    float py = pos_y[vid];
    float vx = vel_x[vid];
    float vy = vel_y[vid];

    VertexOut out;
    out.position = float4(px * 2.0 - 1.0, py * 2.0 - 1.0, 0.0, 1.0);
    out.point_size = uniforms[0];
    out.speed = sqrt(vx * vx + vy * vy);
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              float2 coord [[point_coord]])
{
    float2 c = coord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard_fragment();

    float t = clamp(in.speed * 2.0, 0.0, 1.0);
    float3 color = mix(float3(0.1, 0.3, 0.9), float3(1.0, 1.0, 1.0), t);

    float alpha = 1.0 - smoothstep(0.7, 1.0, sqrt(r2));
    return float4(color * alpha, alpha);
}
)";

static void draw_particles(MTL::Device* device,
                           MTL::CommandQueue* queue,
                           CA::MetalLayer* layer,
                           MTL::RenderPipelineState* pso,
                           MTL::Buffer* posXBuf,
                           MTL::Buffer* posYBuf,
                           MTL::Buffer* velXBuf,
                           MTL::Buffer* velYBuf,
                           size_t particle_count)
{
    if (particle_count == 0)
        return;

    float pointSize = 8.0f;
    MTL::Buffer* uniformBuf = device->newBuffer(&pointSize, sizeof(float), MTL::ResourceStorageModeShared);

    @autoreleasepool {
        CA::MetalDrawable* drawable = layer->nextDrawable();
        if (!drawable) {
            uniformBuf->release();
            return;
        }

        MTL::RenderPassDescriptor* rpd = MTL::RenderPassDescriptor::alloc()->init();
        rpd->colorAttachments()->object(0)->setTexture(drawable->texture());
        rpd->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
        rpd->colorAttachments()->object(0)->setClearColor(MTL::ClearColor(0.05, 0.05, 0.15, 1.0));
        rpd->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);

        MTL::CommandBuffer* cmd = queue->commandBuffer();
        MTL::RenderCommandEncoder* enc = cmd->renderCommandEncoder(rpd);

        enc->setRenderPipelineState(pso);
        enc->setVertexBuffer(posXBuf, 0, 0);
        enc->setVertexBuffer(posYBuf, 0, 1);
        enc->setVertexBuffer(velXBuf, 0, 2);
        enc->setVertexBuffer(velYBuf, 0, 3);
        enc->setVertexBuffer(uniformBuf, 0, 4);
        enc->drawPrimitives(MTL::PrimitiveTypePoint, NS::UInteger(0), NS::UInteger(particle_count));
        enc->endEncoding();

        cmd->presentDrawable(drawable);
        cmd->commit();
        cmd->waitUntilCompleted();

        rpd->release();
    }

    uniformBuf->release();
}

// -- MetalRenderer implementation --------------------------------------------

MetalRenderer::MetalRenderer(int width, int height)
    : _device(nullptr), _queue(nullptr), _layer(nullptr), _pipeline(nullptr), _window(nullptr),
      _width(width), _height(height), _closed(false), _owns_device(true)
{
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    NSRect frame = NSMakeRect(100, 100, width, height);
    NSWindow* window = [[NSWindow alloc]
        initWithContentRect:frame
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered
        defer:NO];
    [window setTitle:@"SPH Fluid Simulation"];

    RendererWindowDelegate* delegate = [[RendererWindowDelegate alloc] init];
    delegate.closedFlag = &_closed;
    [window setDelegate:delegate];

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device)
        throw std::runtime_error("No Metal device found");
    MTL::CommandQueue* queue = device->newCommandQueue();

    CA::MetalLayer* layer = CA::MetalLayer::layer();
    layer->setDevice(device);
    layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    layer->setDrawableSize(CGSizeMake(width, height));

    NSView* contentView = [window contentView];
    [contentView setWantsLayer:YES];
    [contentView setLayer:(__bridge CALayer*)layer];

    NS::Error* error = nullptr;
    auto src = NS::String::string(kShaderSource, NS::UTF8StringEncoding);
    auto opts = MTL::CompileOptions::alloc()->init();
    MTL::Library* library = device->newLibrary(src, opts, &error);
    opts->release();
    if (!library) {
        std::string msg = "Shader compile error";
        if (error) msg = error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }

    auto vertName = NS::String::string("vertex_main", NS::UTF8StringEncoding);
    auto fragName = NS::String::string("fragment_main", NS::UTF8StringEncoding);
    MTL::Function* vertFunc = library->newFunction(vertName);
    MTL::Function* fragFunc = library->newFunction(fragName);

    MTL::RenderPipelineDescriptor* pipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipeDesc->setVertexFunction(vertFunc);
    pipeDesc->setFragmentFunction(fragFunc);
    pipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    pipeDesc->colorAttachments()->object(0)->setBlendingEnabled(true);
    pipeDesc->colorAttachments()->object(0)->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
    pipeDesc->colorAttachments()->object(0)->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    pipeDesc->colorAttachments()->object(0)->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    pipeDesc->colorAttachments()->object(0)->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);

    MTL::RenderPipelineState* pipeline = device->newRenderPipelineState(pipeDesc, &error);
    pipeDesc->release();
    vertFunc->release();
    fragFunc->release();
    library->release();

    if (!pipeline) {
        std::string msg = "Pipeline creation failed";
        if (error) msg = error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }

    _device = device;
    _queue = queue;
    _layer = layer;
    _pipeline = pipeline;
    _window = (void*)window;

    [window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

MetalRenderer::MetalRenderer(MetalDevice& device, int width, int height)
    : _device(nullptr), _queue(nullptr), _layer(nullptr), _pipeline(nullptr), _window(nullptr),
      _width(width), _height(height), _closed(false), _owns_device(false)
{
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    NSRect frame = NSMakeRect(100, 100, width, height);
    NSWindow* window = [[NSWindow alloc]
        initWithContentRect:frame
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered
        defer:NO];
    [window setTitle:@"SPH Fluid Simulation"];

    RendererWindowDelegate* delegate = [[RendererWindowDelegate alloc] init];
    delegate.closedFlag = &_closed;
    [window setDelegate:delegate];

    MTL::Device* raw_device = static_cast<MTL::Device*>(device.raw_device());
    if (!raw_device)
        throw std::runtime_error("Invalid MetalDevice (null MTL::Device)");
    MTL::CommandQueue* queue = raw_device->newCommandQueue();

    CA::MetalLayer* layer = CA::MetalLayer::layer();
    layer->setDevice(raw_device);
    layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    layer->setDrawableSize(CGSizeMake(width, height));

    NSView* contentView = [window contentView];
    [contentView setWantsLayer:YES];
    [contentView setLayer:(__bridge CALayer*)layer];

    NS::Error* error = nullptr;
    auto src = NS::String::string(kShaderSource, NS::UTF8StringEncoding);
    auto opts = MTL::CompileOptions::alloc()->init();
    MTL::Library* library = raw_device->newLibrary(src, opts, &error);
    opts->release();
    if (!library) {
        std::string msg = "Shader compile error";
        if (error) msg = error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }

    auto vertName = NS::String::string("vertex_main", NS::UTF8StringEncoding);
    auto fragName = NS::String::string("fragment_main", NS::UTF8StringEncoding);
    MTL::Function* vertFunc = library->newFunction(vertName);
    MTL::Function* fragFunc = library->newFunction(fragName);

    MTL::RenderPipelineDescriptor* pipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipeDesc->setVertexFunction(vertFunc);
    pipeDesc->setFragmentFunction(fragFunc);
    pipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    pipeDesc->colorAttachments()->object(0)->setBlendingEnabled(true);
    pipeDesc->colorAttachments()->object(0)->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
    pipeDesc->colorAttachments()->object(0)->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    pipeDesc->colorAttachments()->object(0)->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    pipeDesc->colorAttachments()->object(0)->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);

    MTL::RenderPipelineState* pipeline = raw_device->newRenderPipelineState(pipeDesc, &error);
    pipeDesc->release();
    vertFunc->release();
    fragFunc->release();
    library->release();

    if (!pipeline) {
        std::string msg = "Pipeline creation failed";
        if (error) msg = error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }

    _device = raw_device;
    _queue = queue;
    _layer = layer;
    _pipeline = pipeline;
    _window = (void*)window;

    [window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

MetalRenderer::~MetalRenderer() {
    static_cast<MTL::RenderPipelineState*>(_pipeline)->release();
    static_cast<MTL::CommandQueue*>(_queue)->release();
    if (_owns_device)
        static_cast<MTL::Device*>(_device)->release();
    NSWindow* window = (NSWindow*)_window;
    [window close];
}

bool MetalRenderer::poll_events() {
    @autoreleasepool {
        while (true) {
            NSEvent* event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                                untilDate:nil
                                                   inMode:NSDefaultRunLoopMode
                                                  dequeue:YES];
            if (!event) break;
            [NSApp sendEvent:event];
        }
    }
    return !_closed;
}

bool MetalRenderer::is_open() const {
    return !_closed;
}

void MetalRenderer::render_frame(const std::vector<float>& pos_x,
                                 const std::vector<float>& pos_y,
                                 const std::vector<float>& vel_x,
                                 const std::vector<float>& vel_y)
{
    if (_closed)
        return;

    if (pos_x.size() != pos_y.size() || pos_x.size() != vel_x.size() || pos_x.size() != vel_y.size())
        throw std::runtime_error("render_frame size mismatch");

    auto* device = static_cast<MTL::Device*>(_device);
    auto* queue = static_cast<MTL::CommandQueue*>(_queue);
    auto* layer = static_cast<CA::MetalLayer*>(_layer);
    auto* pso = static_cast<MTL::RenderPipelineState*>(_pipeline);

    size_t n = pos_x.size();
    MTL::Buffer* posXBuf = device->newBuffer(pos_x.data(), n * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* posYBuf = device->newBuffer(pos_y.data(), n * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* velXBuf = device->newBuffer(vel_x.data(), n * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* velYBuf = device->newBuffer(vel_y.data(), n * sizeof(float), MTL::ResourceStorageModeShared);

    draw_particles(device, queue, layer, pso, posXBuf, posYBuf, velXBuf, velYBuf, n);

    posXBuf->release();
    posYBuf->release();
    velXBuf->release();
    velYBuf->release();
}

void MetalRenderer::render_frame_from_buffers(MetalDevice& device,
                                              int pos_x_buffer_id,
                                              int pos_y_buffer_id,
                                              int vel_x_buffer_id,
                                              int vel_y_buffer_id)
{
    if (_closed)
        return;

    if (device.raw_device() != _device)
        throw std::runtime_error("render_frame_from_buffers requires renderer/device on same MTLDevice");

    if (device.buffer_is_scalar(pos_x_buffer_id) ||
        device.buffer_is_scalar(pos_y_buffer_id) ||
        device.buffer_is_scalar(vel_x_buffer_id) ||
        device.buffer_is_scalar(vel_y_buffer_id)) {
        throw std::runtime_error("render_frame_from_buffers requires non-scalar buffers");
    }

    int n = device.buffer_count(pos_x_buffer_id);
    if (n != device.buffer_count(pos_y_buffer_id) ||
        n != device.buffer_count(vel_x_buffer_id) ||
        n != device.buffer_count(vel_y_buffer_id)) {
        throw std::runtime_error("render_frame_from_buffers buffer size mismatch");
    }

    auto* posXBuf = static_cast<MTL::Buffer*>(device.raw_buffer(pos_x_buffer_id));
    auto* posYBuf = static_cast<MTL::Buffer*>(device.raw_buffer(pos_y_buffer_id));
    auto* velXBuf = static_cast<MTL::Buffer*>(device.raw_buffer(vel_x_buffer_id));
    auto* velYBuf = static_cast<MTL::Buffer*>(device.raw_buffer(vel_y_buffer_id));

    auto* dev = static_cast<MTL::Device*>(_device);
    auto* queue = static_cast<MTL::CommandQueue*>(_queue);
    auto* layer = static_cast<CA::MetalLayer*>(_layer);
    auto* pso = static_cast<MTL::RenderPipelineState*>(_pipeline);

    draw_particles(dev, queue, layer, pso, posXBuf, posYBuf, velXBuf, velYBuf, static_cast<size_t>(n));
}
