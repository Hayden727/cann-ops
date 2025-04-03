# CMake lowest version requirement
cmake_minimum_required(VERSION 3.14)

# 设置工程名
project(expand_test)

# Compile options
add_compile_options(-std=c++17)

# 设置编译选项
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./")    
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 设置ASCEND_PATH（CANN软件包目录，请根据实际路径修改）和INCLUDE_BASE_DIR（头文件目录）
if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")      
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)

set(CUST_PKG_PATH "${ASCEND_PATH}/opp/vendors/customize/op_api")
set(LIB_PATH ${ASCEND_PATH}/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/lib64)

set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
include_directories(
    ${INCLUDE_BASE_DIR}
    ${INCLUDE_BASE_DIR}/aclnn
    ${CUST_PKG_PATH}/include
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

link_directories(
    ${LIB_PATH}
    ${CUST_PKG_PATH}/lib
)

# 设置可执行文件名（如opapi_test），并指定待运行算子文件*.cpp所在目录
add_executable(${PROJECT_NAME}
               main.cpp)

# 设置链接的库文件路径
# arch表示操作系统结构，os表示操作系统
target_link_libraries(${PROJECT_NAME} PRIVATE
	              Threads::Threads
                  ascendcl
                  nnopbase
                  cust_opapi
                  opapi)

# 可执行文件在CMakeLists文件所在目录的bin目录下
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})