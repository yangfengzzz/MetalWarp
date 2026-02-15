import ast
import io
from contextlib import redirect_stdout

from interpreter import Interpreter


def run_program(source: str):
    tree = ast.parse(source)
    interp = Interpreter()
    stream = io.StringIO()
    with redirect_stdout(stream):
        interp.run(tree)
    return stream.getvalue(), interp
