import ast
import unittest

from tests.test_support import run_program


class InterpreterArithmeticTests(unittest.TestCase):
    def test_arithmetic_and_assignment(self):
        source = """
a = 2 + 3 * 4
b = a - 5
print(a)
print(b)
"""
        output, _ = run_program(source)
        self.assertEqual(output.strip().splitlines(), ["14", "9"])

    def test_floor_div_mod_pow(self):
        source = """
print(7 // 2)
print(7 % 2)
print(2 ** 5)
"""
        output, _ = run_program(source)
        self.assertEqual(output.strip().splitlines(), ["3", "1", "32"])


if __name__ == "__main__":
    unittest.main()
