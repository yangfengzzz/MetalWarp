#include <metal_stdlib>
using namespace metal;

kernel void saxpy(
    const device int *a      [[buffer(0)]],
    const device int *x      [[buffer(1)]],
    const device int *y      [[buffer(2)]],
    device       int *result [[buffer(3)]],
    const device uint *n     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n[0]) {
        result[tid] = a[0] * x[tid] + y[tid];
    }
}
