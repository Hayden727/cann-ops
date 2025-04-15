add_ops_compile_options(
        OP_NAME ForeachErfc
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
op_host/foreach_erfc.cpp
)

target_sources(optiling PRIVATE
        op_host/foreach_erfc.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/foreach_erfc.cpp
)

install(FILES op_kernel/foreach_erfc.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)