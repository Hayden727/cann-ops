#!/bin/bash

# 定义构建目录
BUILD_DIR="build"

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 运行 CMake 配置和编译
cmake ..
make

# 查找生成的 .a 文件
A_FILE=$(find . -name "*.a" -type f)

# 检查是否找到了 .a 文件
if [ -z "$A_FILE" ]; then
    echo "未找到 .a 文件，编译可能失败。"
    exit 1
fi

# 复制头文件到 /usr/include
HEADER_FILES=$(find .. -name "*.h" -type f)
for header in $HEADER_FILES; do
    cp "$header" /usr/include/
done

# 复制 .a 文件到 /usr/local/lib
cp "$A_FILE" /usr/local/lib/

echo "构建完成，头文件和 .a 文件已复制到目标目录。"
    
