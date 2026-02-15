import unittest

from mycompiler.interpreter import InterpreterError
from tests.test_support import run_program


class InterpreterFunctionTests(unittest.TestCase):
    def test_user_function_call_and_return(self):
        source = """
def mul_add(a, b, c):
    return a * b + c

print(mul_add(2, 3, 4))
"""
        output, _ = run_program(source)
        self.assertEqual(output.strip(), "10")

    def test_wrong_arity_raises(self):
        source = """
def f(x, y):
    return x + y

f(1)
"""
        with self.assertRaises(InterpreterError):
            run_program(source)


if __name__ == "__main__":
    unittest.main()
