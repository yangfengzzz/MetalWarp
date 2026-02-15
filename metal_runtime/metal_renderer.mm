#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#import <Cocoa/Cocoa.h>

#include "metal_renderer.h"
#include <cmath>
#include <stdexcept>

// ── Window delegate to detect close ──────────────────────────────────────────

@interface RendererWindowDelegate : NSObject <NSWindowDelegate>
@property (nonatomic) bool* closedFlag;
@end

@implementation RendererWindowDelegate
- (void)windowWillClose:(NSNotification*)notification {
    if (self.closedFlag) *self.closedFlag = true;
}
@end

// ── Embedded Metal shaders ───────────────────────────────────────────────────

static const char* kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float  point_size [[point_size]];
    float  speed;
};

vertex VertexOut vertex_main(const device float4* particles [[buffer(0)]],
                             const device float*  uniforms  [[buffer(1)]],
                             uint vid [[vertex_id]])
{
    float px = particles[vid].x;
    float py = particles[vid].y;
    float vx = particles[vid].z;
    float vy = particles[vid].w;

    VertexOut out;
    // Map [0,1] domain to [-1,1] clip space; flip y so 0 is bottom
    out.position = float4(px * 2.0 - 1.0, py * 2.0 - 1.0, 0.0, 1.0);
    out.point_size = uniforms[0];
    out.speed = sqrt(vx * vx + vy * vy);
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              float2 coord [[point_coord]])
{
    // Discard outside circle
    float2 c = coord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard_fragment();

    // Color by velocity: blue at rest, white at high speed
    float t = clamp(in.speed * 2.0, 0.0, 1.0);
    float3 color = mix(float3(0.1, 0.3, 0.9), float3(1.0, 1.0, 1.0), t);

    // Soft edge
    float alpha = 1.0 - smoothstep(0.7, 1.0, sqrt(r2));
    return float4(color * alpha, alpha);
}
)";

// ── MetalRenderer implementation ─────────────────────────────────────────────

MetalRenderer::MetalRenderer(int width, int height)
    : _width(width), _height(height), _closed(false)
{
    // Initialize NSApplication
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    // Create window
    NSRect frame = NSMakeRect(100, 100, width, height);
    NSWindow* window = [[NSWindow alloc]
        initWithContentRect:frame
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered
        defer:NO];
    [window setTitle:@"SPH Fluid Simulation"];

    // Window delegate for close detection
    RendererWindowDelegate* delegate = [[RendererWindowDelegate alloc] init];
    delegate.closedFlag = &_closed;
    [window setDelegate:delegate];

    // Create Metal device and command queue
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) throw std::runtime_error("No Metal device found");
    MTL::CommandQueue* queue = device->newCommandQueue();

    // Create CAMetalLayer and attach to window view
    CA::MetalLayer* layer = CA::MetalLayer::layer();
    layer->setDevice(device);
    layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    layer->setDrawableSize(CGSizeMake(width, height));

    NSView* contentView = [window contentView];
    [contentView setWantsLayer:YES];
    [contentView setLayer:(__bridge CALayer*)layer];

    // Compile shaders
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

    // Create render pipeline
    MTL::RenderPipelineDescriptor* pipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipeDesc->setVertexFunction(vertFunc);
    pipeDesc->setFragmentFunction(fragFunc);
    pipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    // Enable alpha blending
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

    // Store pointers
    _device   = device;
    _queue    = queue;
    _layer    = layer;
    _pipeline = pipeline;
    _window   = (void*)window;

    // Show window
    [window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

MetalRenderer::~MetalRenderer() {
    static_cast<MTL::RenderPipelineState*>(_pipeline)->release();
    static_cast<MTL::CommandQueue*>(_queue)->release();
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
    if (_closed) return;

    auto* device = static_cast<MTL::Device*>(_device);
    auto* queue  = static_cast<MTL::CommandQueue*>(_queue);
    auto* layer  = static_cast<CA::MetalLayer*>(_layer);
    auto* pso    = static_cast<MTL::RenderPipelineState*>(_pipeline);

    size_t N = pos_x.size();
    if (N == 0) return;

    // Pack interleaved float4 buffer: (px, py, vx, vy) per particle
    std::vector<float> packed(N * 4);
    for (size_t i = 0; i < N; i++) {
        packed[i * 4 + 0] = pos_x[i];
        packed[i * 4 + 1] = pos_y[i];
        packed[i * 4 + 2] = vel_x[i];
        packed[i * 4 + 3] = vel_y[i];
    }

    MTL::Buffer* vertexBuf = device->newBuffer(packed.data(), packed.size() * sizeof(float),
                                                MTL::ResourceStorageModeShared);

    // Uniform: point_size
    float pointSize = 10.0f;
    MTL::Buffer* uniformBuf = device->newBuffer(&pointSize, sizeof(float),
                                                 MTL::ResourceStorageModeShared);

    // Get next drawable
    @autoreleasepool {
        CA::MetalDrawable* drawable = layer->nextDrawable();
        if (!drawable) {
            vertexBuf->release();
            uniformBuf->release();
            return;
        }

        // Create render pass descriptor
        MTL::RenderPassDescriptor* rpd = MTL::RenderPassDescriptor::alloc()->init();
        rpd->colorAttachments()->object(0)->setTexture(drawable->texture());
        rpd->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
        rpd->colorAttachments()->object(0)->setClearColor(MTL::ClearColor(0.05, 0.05, 0.15, 1.0));
        rpd->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);

        MTL::CommandBuffer* cmd = queue->commandBuffer();
        MTL::RenderCommandEncoder* enc = cmd->renderCommandEncoder(rpd);

        enc->setRenderPipelineState(pso);
        enc->setVertexBuffer(vertexBuf, 0, 0);
        enc->setVertexBuffer(uniformBuf, 0, 1);
        enc->drawPrimitives(MTL::PrimitiveTypePoint, NS::UInteger(0), NS::UInteger(N));
        enc->endEncoding();

        cmd->presentDrawable(drawable);
        cmd->commit();
        cmd->waitUntilCompleted();

        rpd->release();
    }

    vertexBuf->release();
    uniformBuf->release();
}
