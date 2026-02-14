#!/bin/bash
# Build the metal_backend pybind11 extension module.
set -e
cd "$(dirname "$0")"

SUFFIX=$(python3-config --extension-suffix)

c++ -O2 -shared -std=c++17 \
    $(python3 -m pybind11 --includes) \
    -I./metal-cpp \
    -framework Metal -framework Foundation \
    metal_backend.mm \
    -o "metal_backend${SUFFIX}" \
    -undefined dynamic_lookup

echo "Built metal_backend${SUFFIX}"
