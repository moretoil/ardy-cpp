cmake_minimum_required(VERSION 3.16)
project(ardy_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ONNX Runtime installation directory. Override with: -DONNXRUNTIME_DIR=/path/to/onnxruntime
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime directory containing include/ and lib/")
endif()

add_executable(ardy-cli
    src/main.cpp
    src/ardy_engine.cpp
)

target_include_directories(ardy-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(ONNXRUNTIME_DIR)
    target_include_directories(ardy-cli PRIVATE ${ONNXRUNTIME_DIR}/include)
    target_link_directories(ardy-cli PRIVATE ${ONNXRUNTIME_DIR}/lib)
endif()

target_link_libraries(ardy-cli PRIVATE onnxruntime)

if(WIN32)
    target_link_libraries(ardy-cli PRIVATE dxgi)
endif()

if(MSVC)
    target_compile_options(ardy-cli PRIVATE /O2 /W3 /utf-8)
else()
    target_compile_options(ardy-cli PRIVATE -O2 -Wall -Wextra)
endif()

# On Windows, copy ONNX Runtime and DirectML runtime DLLs next to the executable if present.
if(WIN32 AND ONNXRUNTIME_DIR)
    foreach(runtime_dll onnxruntime.dll DirectML.dll onnxruntime_providers_shared.dll)
        if(EXISTS "${ONNXRUNTIME_DIR}/lib/${runtime_dll}")
            add_custom_command(TARGET ardy-cli POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_DIR}/lib/${runtime_dll}"
                    "$<TARGET_FILE_DIR:ardy-cli>"
            )
        endif()
    endforeach()
endif()
