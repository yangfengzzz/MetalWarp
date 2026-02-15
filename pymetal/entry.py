"""Python DSL Compiler — CLI entry point.

Supported backends:
  run     — tree-walk interpreter
  ast     — dump the Python AST
  c       — emit compilable C code
  c-run   — emit C, compile with cc, and run
  metal   — emit Metal Shading Language
"""

import ast
import os
import subprocess
import sys
import tempfile

from .interpreter import Interpreter, InterpreterError
from .codegen_c import CCodeGenerator
from .codegen_metal import MetalCodeGenerator


DEMO_PROGRAM = """\
# Fibonacci sequence up to 100
a = 0
b = 1
while a < 100:
    print(a)
    temp = b
    b = a + b
    a = temp
"""


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Python DSL Compiler (using Python ast)")
    ap.add_argument("file", nargs="?", help="Python source file to compile/run")
    ap.add_argument(
        "--emit", choices=["ast", "c", "c-run", "metal", "metal-run", "run"], default="run",
        help="Output mode (default: run)",
    )
    ap.add_argument("--demo", action="store_true", help="Run the built-in demo program")
    args = ap.parse_args()

    # ── read source ──────────────────────────────────────────────────────
    if args.demo:
        source = DEMO_PROGRAM
    elif args.file:
        with open(args.file) as f:
            source = f.read()
    else:
        ap.print_help()
        sys.exit(1)

    # ── parse ────────────────────────────────────────────────────────────
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error: {e}", file=sys.stderr)
        sys.exit(1)

    # ── dispatch ─────────────────────────────────────────────────────────
    if args.emit == "ast":
        print(ast.dump(tree, indent=2))

    elif args.emit == "c":
        print(CCodeGenerator().generate(tree), end="")

    elif args.emit == "c-run":
        c_code = CCodeGenerator().generate(tree)
        tmp_dir = tempfile.mkdtemp()
        src_path = os.path.join(tmp_dir, "program.c")
        bin_path = os.path.join(tmp_dir, "program")
        try:
            with open(src_path, "w") as f:
                f.write(c_code)
            comp = subprocess.run(
                ["cc", "-o", bin_path, src_path, "-lm"],
                capture_output=True, text=True,
            )
            if comp.returncode != 0:
                print(f"Compilation failed:\n{comp.stderr}", file=sys.stderr)
                sys.exit(1)
            result = subprocess.run([bin_path], capture_output=False)
            sys.exit(result.returncode)
        finally:
            if os.path.exists(bin_path):
                os.remove(bin_path)
            if os.path.exists(src_path):
                os.remove(src_path)
            os.rmdir(tmp_dir)

    elif args.emit == "metal":
        print(MetalCodeGenerator().generate(tree), end="")

    elif args.emit == "metal-run":
        gen = MetalCodeGenerator()
        metal_source = gen.generate(tree)
        configs = gen.generate_config(tree)

        # Auto-build the native extension if missing
        try:
            import metal_backend
        except ImportError:
            print("Building metal_backend extension...", file=sys.stderr)
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            source_dir = os.path.join(project_dir, "metal_runtime")
            build_dir = os.path.join(source_dir, "build")
            os.makedirs(build_dir, exist_ok=True)
            # Get pybind11 cmake dir for find_package
            pybind11_dir = subprocess.run(
                ["python3", "-m", "pybind11", "--cmakedir"],
                capture_output=True, text=True).stdout.strip()
            cmake_args = ["cmake", ".."]
            if pybind11_dir:
                cmake_args.append(f"-Dpybind11_DIR={pybind11_dir}")
            comp = subprocess.run(cmake_args, cwd=build_dir,
                                  capture_output=True, text=True)
            if comp.returncode != 0:
                print(f"CMake configure failed:\n{comp.stderr}", file=sys.stderr)
                sys.exit(1)
            comp = subprocess.run(["cmake", "--build", "."], cwd=build_dir,
                                  capture_output=True, text=True)
            if comp.returncode != 0:
                print(f"CMake build failed:\n{comp.stderr}", file=sys.stderr)
                sys.exit(1)
            # Symlink the built .so into the project root so import works
            import glob as globmod
            so_files = globmod.glob(os.path.join(build_dir, "metal_backend*.so"))
            if so_files:
                link_path = os.path.join(project_dir, os.path.basename(so_files[0]))
                if os.path.exists(link_path):
                    os.remove(link_path)
                os.symlink(so_files[0], link_path)
            import metal_backend

        device = metal_backend.MetalDevice()
        for cfg in configs:
            results = device.run_kernel(
                metal_source, cfg["kernel"], cfg["grid_size"], cfg["buffers"]
            )
            print(f"=== {cfg['kernel']} (grid_size={cfg['grid_size']}) ===")
            for name, values in results.items():
                vals = [int(v) if isinstance(v, float) and v == int(v) else v for v in values]
                print(f"  {name}: {vals}")
            print()

    elif args.emit == "run":
        try:
            Interpreter().run(tree)
        except InterpreterError as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
