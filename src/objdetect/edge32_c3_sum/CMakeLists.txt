add_ops_compile_options(
        OP_NAME Edge32C3Sum
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
op_host/edge32_c3_sum.cpp
)

target_sources(optiling PRIVATE
        op_host/edge32_c3_sum.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/edge32_c3_sum.cpp
)

install(FILES op_kernel/edge32_c3_sum.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)