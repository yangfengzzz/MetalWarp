# MyCompiler Documentation

This folder contains architecture documentation for the `MyCompiler` project.

## Table of Contents

1. `index.html` - Project Architecture Overview
2. `compiler-design.html` - Detailed DSL Compiler Design
3. `runtime-design.html` - Runtime and SPH Execution Flow

## How to View

From project root:

```bash
open docs/index.html
```

Or open files directly in any browser.

## Page Guide

### `index.html`
- Repository structure and major components
- Backend modes (`run`, `c`, `c-run`, `metal`, `metal-run`)
- Top-level system architecture diagram

### `compiler-design.html`
- End-to-end compiler pipeline
- Type inference design in `codegen_base.py`
- Metal kernel parameter classification in `codegen_metal.py`
- Notes on inference heuristics and constraints

### `runtime-design.html`
- Python ↔ pybind11 ↔ Metal runtime architecture
- SPH dam-break per-step execution graph
- GPU buffer lifecycle and renderer integration
- Current runtime limitations for scaling

## Source Mapping

The diagrams and descriptions are aligned to these files:
- `entry.py`
- `codegen_base.py`
- `codegen_c.py`
- `codegen_metal.py`
- `metal_kernel.py`
- `metal_runtime/bindings.mm`
- `metal_runtime/metal_device.h`
- `metal_runtime/metal_device.mm`
- `metal_runtime/metal_renderer.h`
- `metal_runtime/metal_renderer.mm`
- `sph_simulation.py`

## Notes

- The docs use self-contained HTML with inline SVG diagrams.
- No external JS/CSS dependencies are required.
