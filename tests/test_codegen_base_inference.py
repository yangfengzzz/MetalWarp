import ast
import unittest

from codegen_base import BaseCodeGenerator


class BaseInferenceTests(unittest.TestCase):
    def test_annotation_marks_float_param(self):
        source = """
def k(mass: float, n):
    x = mass / n
    return x
"""
        tree = ast.parse(source)
        gen = BaseCodeGenerator()
        gen._infer_types(tree)
        self.assertEqual(gen.func_param_types["k"]["mass"], gen.DOUBLE)
        self.assertEqual(gen.func_param_types["k"]["n"], gen.INT)

    def test_subscript_write_promotes_buffer_type(self):
        source = """
def k(buf, tid):
    buf[tid] = 1.25
"""
        tree = ast.parse(source)
        gen = BaseCodeGenerator()
        gen._infer_types(tree)
        self.assertEqual(gen.func_param_types["k"]["buf"], gen.DOUBLE)


if __name__ == "__main__":
    unittest.main()
