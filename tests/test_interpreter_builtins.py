import unittest

from tests.test_support import run_program


class InterpreterBuiltinTests(unittest.TestCase):
    def test_list_append_len_and_minmax(self):
        source = """
xs = []
xs.append(3)
xs.append(1)
xs.append(7)
print(len(xs))
print(min(xs))
print(max(xs))
"""
        output, _ = run_program(source)
        self.assertEqual(output.strip().splitlines(), ["3", "1", "7"])

    def test_list_comprehension_and_abs(self):
        source = """
xs = [-2, -1, 0, 1]
ys = [abs(x) for x in xs if x < 1]
print(len(ys))
print(ys[0])
print(ys[1])
"""
        output, _ = run_program(source)
        self.assertEqual(output.strip().splitlines(), ["3", "2", "1"])


if __name__ == "__main__":
    unittest.main()
