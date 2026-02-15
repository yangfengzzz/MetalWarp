import ast
import unittest

from pymetal.codegen_c import CCodeGenerator


class CCodegenTests(unittest.TestCase):
    def test_emits_main_and_printf(self):
        tree = ast.parse("print(1)\n")
        code = CCodeGenerator().generate(tree)
        self.assertIn("int main()", code)
        self.assertIn("printf", code)

    def test_pow_adds_math_header(self):
        tree = ast.parse("x = 2 ** 8\nprint(x)\n")
        code = CCodeGenerator().generate(tree)
        self.assertIn("#include <math.h>", code)
        self.assertIn("pow", code)


if __name__ == "__main__":
    unittest.main()
