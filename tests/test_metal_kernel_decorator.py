import tempfile
import unittest
from pathlib import Path

from metal_kernel import metal_kernel, MetalKernel


class MetalKernelDecoratorTests(unittest.TestCase):
    def test_decorator_generates_metal_source(self):
        @metal_kernel
        def inc(buf, n, tid):
            if tid < n:
                buf[tid] = buf[tid] + 1

        self.assertEqual(inc.kernel_name, "inc")
        self.assertIn("kernel void inc", inc.metal_source)

    def test_from_file_loads_native_source(self):
        source = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* a [[buffer(0)]], uint tid [[thread_position_in_grid]]) { a[tid] += 1.0; }
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "k.metal"
            p.write_text(source)
            k = MetalKernel.from_file(str(p), "k")
            self.assertEqual(k.kernel_name, "k")
            self.assertIn("kernel void k", k.metal_source)


if __name__ == "__main__":
    unittest.main()
