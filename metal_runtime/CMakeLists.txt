cmake_minimum_required(VERSION 3.18)
project(metal_backend LANGUAGES CXX OBJCXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(metal_backend metal_device.mm bindings.mm)

target_include_directories(metal_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../metal-cpp)

target_link_libraries(metal_backend PRIVATE
    "-framework Metal" "-framework Foundation")

target_compile_features(metal_backend PRIVATE cxx_std_17)
