import unittest

from tests.test_support import run_program


class InterpreterControlFlowTests(unittest.TestCase):
    def test_while_loop_accumulates(self):
        source = """
i = 0
s = 0
while i < 5:
    s = s + i
    i = i + 1
print(s)
"""
        output, _ = run_program(source)
        self.assertEqual(output.strip(), "10")

    def test_for_loop_and_range(self):
        source = """
total = 0
for i in range(1, 6):
    total = total + i
print(total)
"""
        output, _ = run_program(source)
        self.assertEqual(output.strip(), "15")


if __name__ == "__main__":
    unittest.main()
