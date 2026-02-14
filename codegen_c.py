"""C code generator — compiles numeric Python subset to compilable C."""

import ast
from codegen_base import BaseCodeGenerator


class CCodeGenerator(BaseCodeGenerator):

    INT = "long long"
    DOUBLE = "double"

    def __init__(self):
        super().__init__()
        self._needs_math = False

    def _for_loop_type(self) -> str:
        return "long long"

    # ── top-level entry ──────────────────────────────────────────────────

    def generate(self, tree: ast.Module) -> str:
        self._infer_types(tree)
        self._refine_param_types(tree)

        func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        top_level = [n for n in tree.body if not isinstance(n, ast.FunctionDef)]

        self._needs_math = any(
            isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow)
            for n in ast.walk(tree)
        )

        # Headers
        self._emit("#include <stdio.h>")
        if self._needs_math:
            self._emit("#include <math.h>")
        self._emit("")

        # Forward declarations
        for fn in func_nodes:
            ret = self.func_return_types.get(fn.name, self.INT)
            params = ", ".join(
                f"{self.func_param_types.get(fn.name, {}).get(a.arg, self.INT)} {a.arg}"
                for a in fn.args.args
            )
            self._emit(f"{ret} {fn.name}({params});")
        if func_nodes:
            self._emit("")

        # Function definitions
        for fn in func_nodes:
            self._gen_function(fn)

        # main()
        self._emit("int main() {")
        self.indent += 1
        self._declared_vars = set()
        for node in top_level:
            self._gen_stmt(node)
        self._emit("return 0;")
        self.indent -= 1
        self._emit("}")
        self._emit("")

        return "\n".join(self.lines) + "\n"

    # ── C-specific binary ops ────────────────────────────────────────────

    def _gen_binop(self, node: ast.BinOp) -> str:
        left = self._gen_expr(node.left)
        right = self._gen_expr(node.right)

        if isinstance(node.op, ast.Pow):
            self._needs_math = True
            lt = self._infer_expr_type(node.left)
            rt = self._infer_expr_type(node.right)
            if lt == self.INT and rt == self.INT:
                return f"(long long)pow((double){left}, (double){right})"
            return f"pow((double){left}, (double){right})"

        if isinstance(node.op, ast.FloorDiv):
            lt = self._infer_expr_type(node.left)
            rt = self._infer_expr_type(node.right)
            if lt == self.DOUBLE or rt == self.DOUBLE:
                return f"(long long)((double){left} / (double){right})"
            return f"({left} / {right})"

        return f"({left} {self._op_symbol(node.op)} {right})"

    # ── C-specific call: print → printf ──────────────────────────────────

    def _gen_call(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            return self._gen_print(node)
        func = self._gen_expr(node.func)
        args = ", ".join(self._gen_expr(a) for a in node.args)
        return f"{func}({args})"

    def _gen_print(self, node: ast.Call) -> str:
        fmt_parts = []
        arg_exprs = []
        for arg in node.args:
            typ = self._infer_expr_type(arg)
            fmt_parts.append("%f" if typ == self.DOUBLE else "%lld")
            arg_exprs.append(self._gen_expr(arg))

        fmt = " ".join(fmt_parts) + "\\n"
        if not arg_exprs:
            return 'printf("\\n")'

        cast_args = []
        for expr_str, arg_node in zip(arg_exprs, node.args):
            typ = self._infer_expr_type(arg_node)
            if (typ == self.INT
                    and isinstance(arg_node, ast.Constant)
                    and isinstance(arg_node.value, int)):
                cast_args.append(f"(long long){expr_str}")
            else:
                cast_args.append(expr_str)
        return f'printf("{fmt}", {", ".join(cast_args)})'
