"""Metal Shading Language code generator — compiles Python kernels to MSL.

Convention:
  - Any function with a `tid` parameter is a GPU kernel.
  - Buffer parameters (subscripted in body) → device T* [[buffer(N)]]
  - Scalar parameters                      → constant T& [[buffer(N)]]
  - tid                                    → uint [[thread_position_in_grid]]
"""

import ast
from codegen_base import BaseCodeGenerator


class MetalCodeGenerator(BaseCodeGenerator):

    INT = "int"
    DOUBLE = "float"  # GPU compute uses float, not double

    # ── top-level entry ──────────────────────────────────────────────────

    def generate(self, tree: ast.Module) -> str:
        self._infer_types(tree)
        self._refine_param_types(tree)

        kernels = []
        helpers = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            param_names = [a.arg for a in node.args.args]
            (kernels if "tid" in param_names else helpers).append(node)

        self._emit("#include <metal_stdlib>")
        self._emit("using namespace metal;")
        self._emit("")

        for fn in helpers:
            self._gen_function(fn)
        for fn in kernels:
            self._gen_kernel(fn)

        return "\n".join(self.lines) + "\n"

    # ── parameter classification ─────────────────────────────────────────

    def _classify_params(self, node: ast.FunctionDef):
        """Return [(name, kind), ...] for each parameter.

        kind is one of: 'tid', 'buffer_float', 'buffer_int',
        'scalar_float', 'scalar_uint', 'scalar_int'.
        """
        param_names = {a.arg for a in node.args.args}

        # Params used as array[expr]
        indexed = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                if isinstance(child.value, ast.Name) and child.value.id in param_names:
                    indexed.add(child.value.id)

        # Scalar params compared against tid → uint to avoid sign warnings
        tid_compared = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Compare):
                operands = [child.left] + child.comparators
                names = [n.id for n in operands if isinstance(n, ast.Name)]
                if "tid" in names:
                    tid_compared.update(n for n in names if n != "tid" and n in param_names)

        result = []
        for a in node.args.args:
            name = a.arg
            typ = self.func_param_types.get(node.name, {}).get(name, self.INT)
            is_float = (typ == self.DOUBLE)

            if name == "tid":
                result.append((name, "tid"))
            elif name in indexed:
                result.append((name, "buffer_float" if is_float else "buffer_int"))
            elif is_float:
                result.append((name, "scalar_float"))
            elif name in tid_compared:
                result.append((name, "scalar_uint"))
            else:
                result.append((name, "scalar_int"))
        return result

    # ── kernel codegen ───────────────────────────────────────────────────

    _PARAM_TEMPLATES = {
        "tid":          "    uint {name} [[thread_position_in_grid]]",
        "buffer_float": "    device float* {name} [[buffer({idx})]]",
        "buffer_int":   "    device int* {name} [[buffer({idx})]]",
        "scalar_float": "    constant float& {name} [[buffer({idx})]]",
        "scalar_uint":  "    constant uint& {name} [[buffer({idx})]]",
        "scalar_int":   "    constant int& {name} [[buffer({idx})]]",
    }

    def _gen_kernel(self, node: ast.FunctionDef):
        old_func, old_declared = self._current_func, self._declared_vars
        self._current_func = node.name
        self._declared_vars = set()

        classified = self._classify_params(node)

        params = []
        buf_idx = 0
        for name, kind in classified:
            tmpl = self._PARAM_TEMPLATES[kind]
            if kind == "tid":
                params.append(tmpl.format(name=name))
            else:
                params.append(tmpl.format(name=name, idx=buf_idx))
                buf_idx += 1
            self._declared_vars.add(name)

        self._emit(f"kernel void {node.name}(")
        self._emit(",\n".join(params))
        self._emit(") {")
        self.indent += 1
        for s in node.body:
            self._gen_stmt(s)
        self.indent -= 1
        self._emit("}")
        self._emit("")

        self._current_func, self._declared_vars = old_func, old_declared

    # ── Metal-specific binary ops ────────────────────────────────────────

    def _gen_binop(self, node: ast.BinOp) -> str:
        left = self._gen_expr(node.left)
        right = self._gen_expr(node.right)

        if isinstance(node.op, ast.Pow):
            return f"pow((float){left}, (float){right})"

        if isinstance(node.op, ast.FloorDiv):
            lt = self._infer_expr_type(node.left)
            rt = self._infer_expr_type(node.right)
            if lt == self.DOUBLE or rt == self.DOUBLE:
                return f"(int)((float){left} / (float){right})"
            return f"({left} / {right})"

        return f"({left} {self._op_symbol(node.op)} {right})"

    # ── Metal-specific call: reject print ────────────────────────────────

    def _gen_call(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            raise ValueError(
                "print() is not supported in Metal shaders (GPU has no stdout)"
            )
        func = self._gen_expr(node.func)
        args = ", ".join(self._gen_expr(a) for a in node.args)
        return f"{func}({args})"
