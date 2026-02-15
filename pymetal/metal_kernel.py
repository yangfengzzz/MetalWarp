"""@metal_kernel decorator — compile Python functions to Metal GPU kernels.

Usage:
    from metal_kernel import metal_kernel

    @metal_kernel
    def saxpy(a, x, y, result, n, tid):
        if tid < n:
            result[tid] = a * x[tid] + y[tid]

    print(saxpy.metal_source)
    results = saxpy.launch(grid_size=8, buffers=[...])
"""

import ast
import inspect
import textwrap

from .codegen_metal import MetalCodeGenerator


_device = None


def _get_device():
    global _device
    if _device is None:
        import metal_backend
        _device = metal_backend.MetalDevice()
    return _device


class MetalKernel:
    def __init__(self, fn=None, *, metal_source=None, kernel_name=None):
        if fn is not None:
            self.kernel_name = fn.__name__

            source = inspect.getsource(fn)
            source = textwrap.dedent(source)
            # Strip decorator lines
            lines = [line for line in source.splitlines()
                     if not line.strip().startswith("@")]
            source = "\n".join(lines)

            tree = ast.parse(source)
            gen = MetalCodeGenerator()
            self.metal_source = gen.generate(tree)
        else:
            self.metal_source = metal_source
            self.kernel_name = kernel_name

    @classmethod
    def from_file(cls, path, kernel_name):
        with open(path, "r") as f:
            source = f.read()
        return cls(metal_source=source, kernel_name=kernel_name)

    def launch(self, grid_size, buffers):
        device = _get_device()
        return device.run_kernel(self.metal_source, self.kernel_name,
                                 grid_size, buffers)


def metal_kernel(fn):
    return MetalKernel(fn)
