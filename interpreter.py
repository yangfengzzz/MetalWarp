"""Tree-walk interpreter over Python AST."""

import ast
import operator
from typing import Any, Dict, List, Optional


class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


class InterpreterError(Exception):
    pass


class Environment:
    """Scoped variable environment with parent chain."""

    def __init__(self, parent: Optional["Environment"] = None):
        self.vars: Dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise InterpreterError(f"Undefined variable '{name}'")

    def set(self, name: str, value: Any):
        self.vars[name] = value

    def has(self, name: str) -> bool:
        if name in self.vars:
            return True
        return self.parent.has(name) if self.parent else False


class Function:
    """A user-defined function."""

    def __init__(self, name: str, params: List[str], body: List[ast.stmt],
                 closure: Environment):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure

    def __repr__(self):
        return f"<function {self.name}({', '.join(self.params)})>"


class Interpreter:
    def __init__(self):
        self.global_env = Environment()
        self.builtins = {
            "print": self._builtin_print,
            "len": lambda args: len(args[0]),
            "range": lambda args: list(range(*args)),
            "int": lambda args: int(args[0]),
            "float": lambda args: float(args[0]),
            "str": lambda args: str(args[0]),
            "abs": lambda args: abs(args[0]),
            "min": lambda args: min(*args) if len(args) > 1 else min(args[0]),
            "max": lambda args: max(*args) if len(args) > 1 else max(args[0]),
            "type": lambda args: type(args[0]).__name__,
            "isinstance": lambda args: isinstance(args[0], args[1]),
            "input": lambda args: input(args[0] if args else ""),
            "append": None,
        }

    def _builtin_print(self, args, kwargs=None):
        kwargs = kwargs or {}
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        print(*args, sep=sep, end=end)

    def run(self, tree: ast.Module):
        for name, func in self.builtins.items():
            if func is not None:
                self.global_env.set(name, func)
        for node in tree.body:
            self._exec(node, self.global_env)

    # ── Statement execution ──────────────────────────────────────────────

    def _exec(self, node: ast.stmt, env: Environment):
        if isinstance(node, ast.Assign):
            value = self._eval(node.value, env)
            for target in node.targets:
                self._assign(target, value, env)

        elif isinstance(node, ast.AugAssign):
            target_val = self._eval(node.target, env)
            value = self._eval(node.value, env)
            result = self._apply_op(node.op, target_val, value)
            self._assign(node.target, result, env)

        elif isinstance(node, ast.Expr):
            self._eval(node.value, env)

        elif isinstance(node, ast.If):
            if self._eval(node.test, env):
                for stmt in node.body:
                    self._exec(stmt, env)
            else:
                for stmt in node.orelse:
                    self._exec(stmt, env)

        elif isinstance(node, ast.While):
            while self._eval(node.test, env):
                for stmt in node.body:
                    self._exec(stmt, env)

        elif isinstance(node, ast.For):
            iterable = self._eval(node.iter, env)
            for item in iterable:
                self._assign(node.target, item, env)
                for stmt in node.body:
                    self._exec(stmt, env)

        elif isinstance(node, ast.FunctionDef):
            params = [arg.arg for arg in node.args.args]
            func = Function(node.name, params, node.body, env)
            env.set(node.name, func)

        elif isinstance(node, ast.Return):
            value = self._eval(node.value, env) if node.value else None
            raise ReturnSignal(value)

        elif isinstance(node, ast.Pass):
            pass

        elif isinstance(node, ast.Break):
            raise _BreakSignal()

        elif isinstance(node, ast.Continue):
            raise _ContinueSignal()

        else:
            raise InterpreterError(
                f"Unsupported statement: {type(node).__name__} at line {node.lineno}"
            )

    def _assign(self, target: ast.expr, value: Any, env: Environment):
        if isinstance(target, ast.Name):
            env.set(target.id, value)
        elif isinstance(target, ast.Subscript):
            obj = self._eval(target.value, env)
            idx = self._eval(target.slice, env)
            obj[idx] = value
        elif isinstance(target, ast.Tuple):
            for t, v in zip(target.elts, value):
                self._assign(t, v, env)
        else:
            raise InterpreterError(
                f"Unsupported assignment target: {type(target).__name__}"
            )

    # ── Expression evaluation ────────────────────────────────────────────

    def _eval(self, node: ast.expr, env: Environment) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            return env.get(node.id)

        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, env)
            right = self._eval(node.right, env)
            return self._apply_op(node.op, left, right)

        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand, env)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Not):
                return not operand
            raise InterpreterError(f"Unsupported unary op: {type(node.op).__name__}")

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = True
                for val in node.values:
                    result = self._eval(val, env)
                    if not result:
                        return result
                return result
            if isinstance(node.op, ast.Or):
                result = False
                for val in node.values:
                    result = self._eval(val, env)
                    if result:
                        return result
                return result

        if isinstance(node, ast.Compare):
            left = self._eval(node.left, env)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator, env)
                if not self._compare(op, left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.Call):
            return self._eval_call(node, env)

        if isinstance(node, ast.IfExp):
            if self._eval(node.test, env):
                return self._eval(node.body, env)
            return self._eval(node.orelse, env)

        if isinstance(node, ast.List):
            return [self._eval(e, env) for e in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(self._eval(e, env) for e in node.elts)

        if isinstance(node, ast.Subscript):
            obj = self._eval(node.value, env)
            idx = self._eval(node.slice, env)
            return obj[idx]

        if isinstance(node, ast.Slice):
            lower = self._eval(node.lower, env) if node.lower else None
            upper = self._eval(node.upper, env) if node.upper else None
            step = self._eval(node.step, env) if node.step else None
            return slice(lower, upper, step)

        if isinstance(node, ast.Attribute):
            obj = self._eval(node.value, env)
            return getattr(obj, node.attr)

        if isinstance(node, ast.JoinedStr):
            parts = []
            for val in node.values:
                if isinstance(val, ast.Constant):
                    parts.append(str(val.value))
                elif isinstance(val, ast.FormattedValue):
                    parts.append(str(self._eval(val.value, env)))
                else:
                    parts.append(str(self._eval(val, env)))
            return "".join(parts)

        if isinstance(node, ast.ListComp):
            return self._eval_listcomp(node, env)

        raise InterpreterError(
            f"Unsupported expression: {type(node).__name__} "
            f"at line {getattr(node, 'lineno', '?')}"
        )

    def _eval_call(self, node: ast.Call, env: Environment) -> Any:
        if isinstance(node.func, ast.Attribute):
            obj = self._eval(node.func.value, env)
            method = getattr(obj, node.func.attr)
            args = [self._eval(a, env) for a in node.args]
            return method(*args)

        func = self._eval(node.func, env)
        args = [self._eval(a, env) for a in node.args]

        kwargs = {}
        for kw in node.keywords:
            kwargs[kw.arg] = self._eval(kw.value, env)

        if callable(func) and not isinstance(func, Function):
            if func == self._builtin_print:
                return func(args, kwargs)
            return func(args)

        if isinstance(node.func, ast.Name) and node.func.id in self.builtins:
            builtin = self.builtins[node.func.id]
            if builtin == self._builtin_print:
                return builtin(args, kwargs)
            return builtin(args)

        if isinstance(func, Function):
            if len(args) != len(func.params):
                raise InterpreterError(
                    f"{func.name}() expects {len(func.params)} args, got {len(args)}"
                )
            call_env = Environment(parent=func.closure)
            for param, arg in zip(func.params, args):
                call_env.set(param, arg)
            try:
                for stmt in func.body:
                    self._exec(stmt, call_env)
            except ReturnSignal as ret:
                return ret.value
            return None

        raise InterpreterError(f"'{func}' is not callable")

    def _eval_listcomp(self, node: ast.ListComp, env: Environment) -> list:
        result = []
        gen = node.generators[0]
        iterable = self._eval(gen.iter, env)
        for item in iterable:
            inner_env = Environment(parent=env)
            self._assign(gen.target, item, inner_env)
            passed = all(self._eval(if_, inner_env) for if_ in gen.ifs)
            if passed:
                result.append(self._eval(node.elt, inner_env))
        return result

    # ── Operators ────────────────────────────────────────────────────────

    BINOP_MAP = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.LShift: operator.lshift,
        ast.RShift: operator.rshift, ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_, ast.BitXor: operator.xor,
    }

    CMP_MAP = {
        ast.Eq: operator.eq, ast.NotEq: operator.ne,
        ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
        ast.Is: operator.is_, ast.IsNot: operator.is_not,
    }

    def _apply_op(self, op: ast.operator, left: Any, right: Any) -> Any:
        op_func = self.BINOP_MAP.get(type(op))
        if op_func:
            return op_func(left, right)
        raise InterpreterError(f"Unsupported binary op: {type(op).__name__}")

    def _compare(self, op: ast.cmpop, left: Any, right: Any) -> bool:
        cmp_func = self.CMP_MAP.get(type(op))
        if cmp_func:
            return cmp_func(left, right)
        raise InterpreterError(f"Unsupported comparison: {type(op).__name__}")
