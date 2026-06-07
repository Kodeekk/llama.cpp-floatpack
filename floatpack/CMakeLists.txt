# floatpack: FP4-packed Vulkan matmul backend for Intel iGPUs

option(FLOATPACK_VULKAN "Build floatpack with Vulkan backend" ON)

# Try to find glslc for shader compilation
if (FLOATPACK_VULKAN)
    find_program(GLSLC glslc)
    if (GLSLC)
        set(FP4_SPV "${CMAKE_CURRENT_BINARY_DIR}/floatpack_fp4.spv")
        add_custom_command(
            OUTPUT ${FP4_SPV}
            COMMAND ${GLSLC} "${CMAKE_CURRENT_SOURCE_DIR}/floatpack_fp4.comp" -o ${FP4_SPV}
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/floatpack_fp4.comp"
            COMMENT "Compiling floatpack FP4 shader to SPIR-V"
        )
        add_custom_target(floatpack_shader ALL DEPENDS ${FP4_SPV})
        # Copy to source dir for runtime loading (at build time, not configure time)
        add_custom_command(TARGET floatpack_shader POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FP4_SPV} "${CMAKE_CURRENT_SOURCE_DIR}/floatpack_fp4.spv"
            COMMENT "Copying floatpack_fp4.spv to source dir")

        add_compile_definitions(FLOATPACK_VULKAN_ENABLED)
        message(STATUS "floatpack: glslc found, shader will be compiled")
    else()
        message(WARNING "floatpack: glslc not found. Install Vulkan SDK or run:\n"
                        "  glslc floatpack/floatpack_fp4.comp -o floatpack/floatpack_fp4.spv")
    endif()
endif()

# Main library
add_library(floatpack STATIC
    floatpack.c
)

set_target_properties(floatpack PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(floatpack PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if (FLOATPACK_VULKAN)
    find_package(Vulkan QUIET COMPONENTS)
    if (Vulkan_FOUND)
        target_link_libraries(floatpack PUBLIC Vulkan::Vulkan)
        target_compile_definitions(floatpack PRIVATE FLOATPACK_VULKAN=1)
    else()
        message(WARNING "floatpack: Vulkan not found, building CPU-only version")
    endif()
endif()

# Test executable
option(FLOATPACK_BUILD_TESTS "Build floatpack tests" ON)
if (FLOATPACK_BUILD_TESTS)
    add_executable(test_floatpack
        test_floatpack.c
    )
    target_link_libraries(test_floatpack PRIVATE floatpack)
    if (TARGET floatpack_shader)
        add_dependencies(test_floatpack floatpack_shader)
    endif()
endif()
