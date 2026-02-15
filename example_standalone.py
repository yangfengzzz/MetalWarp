"""Standalone example: compile a DSL kernel to Metal, configure buffers, and run on GPU.

This shows the full pipeline in plain Python — no CLI involved:
  1. Parse a DSL kernel string
  2. Use the compiler to emit Metal Shading Language
  3. Create a MetalDevice
  4. Build buffer configs by hand
  5. Launch the kernel and read back results
"""

import ast
from codegen_metal import MetalCodeGenerator

# ── Step 1: define a kernel in the DSL ────────────────────────────────────────

kernel_source = """\
def saxpy(a, x, y, result, n, tid):
    if tid < n:
        result[tid] = a * x[tid] + y[tid]
"""

# ── Step 2: compile DSL → Metal Shading Language ─────────────────────────────

tree = ast.parse(kernel_source)
gen = MetalCodeGenerator()
metal_code = gen.generate(tree)

print("--- Generated Metal code ---")
print(metal_code)

# ── Step 3: create the Metal device ──────────────────────────────────────────

import metal_backend

device = metal_backend.MetalDevice()

# ── Step 4: set up buffers ───────────────────────────────────────────────────
# The compiler infers int types for all parameters (no type annotations in DSL).
# Buffer order must match the kernel signature (excluding `tid`):
#   a       → buffer(0)  constant int&   (scalar)
#   x       → buffer(1)  device int*     (input array)
#   y       → buffer(2)  device int*     (input array)
#   result  → buffer(3)  device int*     (output, zero-initialized)
#   n       → buffer(4)  constant uint&  (scalar, compared with tid)

N = 8
alpha = 3
x = list(range(1, N + 1))          # [1, 2, 3, 4, 5, 6, 7, 8]
y = [10 * i for i in range(1, N + 1)]  # [10, 20, 30, 40, 50, 60, 70, 80]

buffers = [
    {"name": "a",      "type": "int",  "value": alpha},
    {"name": "x",      "type": "int",  "data": x},
    {"name": "y",      "type": "int",  "data": y},
    {"name": "result", "type": "int",  "size": N},
    {"name": "n",      "type": "uint", "value": N},
]

# ── Step 5: launch kernel and print results ──────────────────────────────────

results = device.run_kernel(metal_code, "saxpy", N, buffers)

print("--- GPU results ---")
for name, values in results.items():
    vals = [int(v) for v in values]
    print(f"  {name}: {vals}")

# Expected: result[i] = 3 * x[i] + y[i]
#   → [13, 26, 39, 52, 65, 78, 91, 104]
