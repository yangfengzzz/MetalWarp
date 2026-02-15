"""Example: load a native .metal file and run it on the GPU.

Uses MetalKernel.from_file() to load hand-written MSL and launch it
with the same API as the @metal_kernel decorator.
"""

from metal_kernel import MetalKernel

# ── Load a native .metal file ────────────────────────────────────────────────

saxpy = MetalKernel.from_file("kernels/saxpy.metal", "saxpy")

# ── Inspect the native Metal code ────────────────────────────────────────────

print("--- Native Metal code ---")
print(saxpy.metal_source)

# ── Set up buffers and launch ────────────────────────────────────────────────

N = 8
alpha = 3
x = list(range(1, N + 1))                # [1, 2, 3, 4, 5, 6, 7, 8]
y = [10 * i for i in range(1, N + 1)]    # [10, 20, 30, 40, 50, 60, 70, 80]

buffers = [
    {"name": "a",      "type": "int",  "value": alpha},
    {"name": "x",      "type": "int",  "data": x},
    {"name": "y",      "type": "int",  "data": y},
    {"name": "result", "type": "int",  "size": N},
    {"name": "n",      "type": "uint", "value": N},
]

results = saxpy.launch(grid_size=N, buffers=buffers)

print("--- GPU results ---")
for name, values in results.items():
    vals = [int(v) for v in values]
    print(f"  {name}: {vals}")

# Expected: result[i] = 3 * x[i] + y[i]
#   → [13, 26, 39, 52, 65, 78, 91, 104]
