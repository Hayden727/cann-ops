# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

add_ops_compile_options(
        OP_NAME AddCustom
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
op_host/add_custom.cpp
)

# target_sources(opapi PRIVATE
#         op_host/add_custom.cpp
# )

target_sources(optiling PRIVATE
        op_host/add_custom.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/add_custom.cpp
)

# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_flash_attention_score.h
#         DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
# )
