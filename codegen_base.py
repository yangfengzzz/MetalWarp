"""Base code generator — shared type inference and AST-to-C-like codegen."""

import ast
from typing import Dict, List, Optional


class BaseCodeGenerator:
    """Shared logic for C and Metal backends: type inference, statement and
    expression codegen, if/elif/else, for-range, while, subscripts."""

    INT = "int"
    DOUBLE = "double"

    def __init__(self):
        self.lines: List[str] = []
        self.indent = 0
        self.global_types: Dict[str, str] = {}
        self.func_param_types: Dict[str, Dict[str, str]] = {}
        self.func_return_types: Dict[str, str] = {}
        self.func_local_types: Dict[str, Dict[str, str]] = {}
        self._current_func: Optional[str] = None
        self._declared_vars: set = set()

    # ── output helpers ───────────────────────────────────────────────────

    def _emit(self, line: str):
        self.lines.append("    " * self.indent + line)

    # ── variable type bookkeeping ────────────────────────────────────────

    def _type_of_var(self, name: str) -> str:
        if self._current_func:
            ft = self.func_param_types.get(self._current_func, {})
            if name in ft:
                return ft[name]
            lt = self.func_local_types.get(self._current_func, {})
            if name in lt:
                return lt[name]
        return self.global_types.get(name, self.INT)

    def _set_var_type(self, name: str, typ: str):
        if self._current_func:
            self.func_local_types.setdefault(self._current_func, {})[name] = typ
        else:
            self.global_types[name] = typ

    def _merge_types(self, a: str, b: str) -> str:
        if a == self.DOUBLE or b == self.DOUBLE:
            return self.DOUBLE
        return self.INT

    # ── type inference: expressions ──────────────────────────────────────

    def _infer_expr_type(self, node: ast.expr) -> str:
        if isinstance(node, ast.Constant):
            return self.DOUBLE if isinstance(node.value, float) else self.INT

        if isinstance(node, ast.Name):
            return self._type_of_var(node.id)

        if isinstance(node, ast.BinOp):
            lt = self._infer_expr_type(node.left)
            rt = self._infer_expr_type(node.right)
            if isinstance(node.op, ast.Div):
                return self.DOUBLE
            if isinstance(node.op, ast.FloorDiv):
                return self.INT
            return self._merge_types(lt, rt)

        if isinstance(node, ast.UnaryOp):
            return self._infer_expr_type(node.operand)

        if isinstance(node, ast.IfExp):
            return self._merge_types(
                self._infer_expr_type(node.body),
                self._infer_expr_type(node.orelse),
            )

        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return self.INT

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in self.func_return_types:
                    return self.func_return_types[fname]
            return self.INT

        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                return self._type_of_var(node.value.id)
            return self.INT

        return self.INT

    # ── type inference: full-tree pass ───────────────────────────────────

    def _infer_types(self, tree: ast.Module):
        for node in tree.body:
            self._infer_stmt_types(node)

    def _infer_stmt_types(self, node: ast.stmt):
        if isinstance(node, ast.Assign):
            typ = self._infer_expr_type(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._set_var_type(target.id, typ)

        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                existing = self._type_of_var(node.target.id)
                rhs = self._infer_expr_type(node.value)
                self._set_var_type(node.target.id, self._merge_types(existing, rhs))

        elif isinstance(node, ast.FunctionDef):
            old_func = self._current_func
            self._current_func = node.name
            self.func_param_types[node.name] = {
                a.arg: self.INT for a in node.args.args
            }
            self.func_local_types[node.name] = {}
            for s in node.body:
                self._infer_stmt_types(s)
            self.func_return_types[node.name] = self._infer_return_type(node.body)
            self._current_func = old_func

        elif isinstance(node, ast.If):
            for s in node.body + node.orelse:
                self._infer_stmt_types(s)

        elif isinstance(node, ast.While):
            for s in node.body:
                self._infer_stmt_types(s)

        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                self._set_var_type(node.target.id, self.INT)
            for s in node.body:
                self._infer_stmt_types(s)

    def _infer_return_type(self, body: List[ast.stmt]) -> str:
        ret = self.INT
        for node in body:
            if isinstance(node, ast.Return) and node.value:
                ret = self._merge_types(ret, self._infer_expr_type(node.value))
            elif isinstance(node, ast.If):
                ret = self._merge_types(ret, self._infer_return_type(node.body))
                ret = self._merge_types(ret, self._infer_return_type(node.orelse))
            elif isinstance(node, (ast.While, ast.For)):
                ret = self._merge_types(ret, self._infer_return_type(node.body))
        return ret

    def _refine_param_types(self, tree: ast.Module):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in self.func_param_types:
                    ptypes = self.func_param_types[fname]
                    param_names = list(ptypes.keys())
                    for i, arg in enumerate(node.args):
                        if i < len(param_names):
                            arg_type = self._infer_expr_type(arg)
                            ptypes[param_names[i]] = self._merge_types(
                                ptypes[param_names[i]], arg_type
                            )

    # ── statement codegen ────────────────────────────────────────────────

    def _gen_stmt(self, node: ast.stmt):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    val = self._gen_expr(node.value)
                    vtype = self._type_of_var(target.id)
                    if target.id not in self._declared_vars:
                        self._declared_vars.add(target.id)
                        self._emit(f"{vtype} {target.id} = {val};")
                    else:
                        self._emit(f"{target.id} = {val};")
                elif isinstance(target, ast.Subscript):
                    self._emit(f"{self._gen_expr(target)} = {self._gen_expr(node.value)};")

        elif isinstance(node, ast.AugAssign):
            self._emit(
                f"{self._gen_target(node.target)} {self._op_symbol(node.op)}= "
                f"{self._gen_expr(node.value)};"
            )

        elif isinstance(node, ast.Expr):
            self._emit(f"{self._gen_expr(node.value)};")

        elif isinstance(node, ast.If):
            self._gen_if(node)

        elif isinstance(node, ast.While):
            self._emit(f"while ({self._gen_expr(node.test)}) {{")
            self.indent += 1
            for s in node.body:
                self._gen_stmt(s)
            self.indent -= 1
            self._emit("}")

        elif isinstance(node, ast.For):
            self._gen_for(node)

        elif isinstance(node, ast.Return):
            if node.value:
                self._emit(f"return {self._gen_expr(node.value)};")
            else:
                self._emit("return;")

        elif isinstance(node, ast.Pass):
            self._emit("// pass")
        elif isinstance(node, ast.Break):
            self._emit("break;")
        elif isinstance(node, ast.Continue):
            self._emit("continue;")

    # ── if / elif / else ─────────────────────────────────────────────────

    def _gen_if(self, node: ast.If):
        self._emit(f"if ({self._gen_expr(node.test)}) {{")
        self.indent += 1
        for s in node.body:
            self._gen_stmt(s)
        self.indent -= 1

        if not node.orelse:
            self._emit("}")
            return

        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            self._gen_elif_chain(node.orelse[0])
            self._emit("}")
        else:
            self._emit("} else {")
            self.indent += 1
            for s in node.orelse:
                self._gen_stmt(s)
            self.indent -= 1
            self._emit("}")

    def _gen_elif_chain(self, node: ast.If):
        self._emit(f"}} else if ({self._gen_expr(node.test)}) {{")
        self.indent += 1
        for s in node.body:
            self._gen_stmt(s)
        self.indent -= 1

        if not node.orelse:
            return
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            self._gen_elif_chain(node.orelse[0])
        else:
            self._emit("} else {")
            self.indent += 1
            for s in node.orelse:
                self._gen_stmt(s)
            self.indent -= 1

    # ── for-range ────────────────────────────────────────────────────────

    def _gen_for(self, node: ast.For):
        target = self._gen_target(node.target)
        if (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"):
            args = node.iter.args
            if len(args) == 1:
                start, end, step = "0", self._gen_expr(args[0]), "1"
                step_negative = False
            elif len(args) == 2:
                start, end = self._gen_expr(args[0]), self._gen_expr(args[1])
                step, step_negative = "1", False
            elif len(args) == 3:
                start = self._gen_expr(args[0])
                end = self._gen_expr(args[1])
                step = self._gen_expr(args[2])
                step_negative = self._is_negative_step(args[2])
            else:
                start, end, step, step_negative = "0", "0", "1", False

            cmp = ">" if step_negative else "<"
            loop_type = self._for_loop_type()
            self._declared_vars.add(target)
            self._emit(
                f"for ({loop_type} {target} = {start}; "
                f"{target} {cmp} {end}; {target} += {step}) {{"
            )
        else:
            self._emit("/* unsupported for-iter */ {")

        self.indent += 1
        for s in node.body:
            self._gen_stmt(s)
        self.indent -= 1
        self._emit("}")

    def _for_loop_type(self) -> str:
        return self.INT

    def _is_negative_step(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value < 0
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return isinstance(node.operand, ast.Constant)
        return False

    # ── expression codegen ───────────────────────────────────────────────

    def _gen_expr(self, node: ast.expr) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                return repr(node.value)
            if isinstance(node.value, bool):
                return "1" if node.value else "0"
            if isinstance(node.value, int):
                return repr(node.value)
            if isinstance(node.value, str):
                return f'"{node.value}"'
            return repr(node.value)

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.BinOp):
            return self._gen_binop(node)

        if isinstance(node, ast.UnaryOp):
            operand = self._gen_expr(node.operand)
            if isinstance(node.op, ast.USub):
                return f"(-{operand})"
            if isinstance(node.op, ast.UAdd):
                return f"(+{operand})"
            if isinstance(node.op, ast.Not):
                return f"(!{operand})"
            return operand

        if isinstance(node, ast.Compare):
            parts = [self._gen_expr(node.left)]
            for op, comp in zip(node.ops, node.comparators):
                parts.append(self._cmp_symbol(op))
                parts.append(self._gen_expr(comp))
            return "(" + " ".join(parts) + ")"

        if isinstance(node, ast.BoolOp):
            joiner = " && " if isinstance(node.op, ast.And) else " || "
            return "(" + joiner.join(self._gen_expr(v) for v in node.values) + ")"

        if isinstance(node, ast.Call):
            return self._gen_call(node)

        if isinstance(node, ast.IfExp):
            return (f"({self._gen_expr(node.test)} ? "
                    f"{self._gen_expr(node.body)} : {self._gen_expr(node.orelse)})")

        if isinstance(node, ast.Subscript):
            return f"{self._gen_expr(node.value)}[{self._gen_expr(node.slice)}]"

        return f"/* unsupported: {type(node).__name__} */"

    def _gen_binop(self, node: ast.BinOp) -> str:
        left = self._gen_expr(node.left)
        right = self._gen_expr(node.right)
        return f"({left} {self._op_symbol(node.op)} {right})"

    def _gen_call(self, node: ast.Call) -> str:
        func = self._gen_expr(node.func)
        args = ", ".join(self._gen_expr(a) for a in node.args)
        return f"{func}({args})"

    # ── function codegen ─────────────────────────────────────────────────

    def _gen_function(self, node: ast.FunctionDef):
        old_func = self._current_func
        self._current_func = node.name
        old_declared = self._declared_vars
        self._declared_vars = set()

        ret_type = self.func_return_types.get(node.name, self.INT)
        param_types = self.func_param_types.get(node.name, {})
        params = ", ".join(
            f"{param_types.get(a.arg, self.INT)} {a.arg}" for a in node.args.args
        )
        for a in node.args.args:
            self._declared_vars.add(a.arg)

        self._emit(f"{ret_type} {node.name}({params}) {{")
        self.indent += 1
        for s in node.body:
            self._gen_stmt(s)
        self.indent -= 1
        self._emit("}")
        self._emit("")

        self._current_func = old_func
        self._declared_vars = old_declared

    # ── symbol helpers ───────────────────────────────────────────────────

    def _gen_target(self, target: ast.expr) -> str:
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Subscript):
            return f"{self._gen_expr(target.value)}[{self._gen_expr(target.slice)}]"
        return "?"

    _OP_SYMBOLS = {
        ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
        ast.FloorDiv: "/", ast.Mod: "%",
        ast.LShift: "<<", ast.RShift: ">>",
        ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    }

    _CMP_SYMBOLS = {
        ast.Eq: "==", ast.NotEq: "!=",
        ast.Lt: "<", ast.LtE: "<=",
        ast.Gt: ">", ast.GtE: ">=",
    }

    def _op_symbol(self, op: ast.operator) -> str:
        return self._OP_SYMBOLS.get(type(op), "?")

    def _cmp_symbol(self, op: ast.cmpop) -> str:
        return self._CMP_SYMBOLS.get(type(op), "?")
