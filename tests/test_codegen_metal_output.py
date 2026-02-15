import ast
import unittest

from pymetal.codegen_metal import MetalCodeGenerator


class MetalCodegenTests(unittest.TestCase):
    def test_kernel_signature_uses_thread_index(self):
        src = """
def saxpy(a: float, x, y, out, n, tid):
    if tid < n:
        out[tid] = a * x[tid] + y[tid]
"""
        code = MetalCodeGenerator().generate(ast.parse(src))
        self.assertIn("kernel void saxpy", code)
        self.assertIn("[[thread_position_in_grid]]", code)
        self.assertIn("constant float& a", code)

    def test_print_in_kernel_is_rejected(self):
        src = """
def bad(tid):
    print(tid)
"""
        with self.assertRaises(ValueError):
            MetalCodeGenerator().generate(ast.parse(src))


if __name__ == "__main__":
    unittest.main()
