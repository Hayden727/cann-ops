#!/bin/bash


# 步骤1: 运行gen_data.py生成输入bin文件和golden标杆输出数据
echo "正在生成输入数据和golden标杆数据..."
mkdir -p script/input
mkdir -p script/output
python3 script/gen_data.py
if [ $? -ne 0 ]; then
    echo "生成数据失败，脚本终止。"
    exit 1
fi

# 步骤2: 创建构建目录并进入
mkdir -p build
cd build
if [ $? -ne 0 ]; then
    echo "无法进入构建目录，脚本终止。"
    exit 1
fi

# 步骤3: 使用CMake配置项目
echo "正在配置CMake项目..."
cmake ..
if [ $? -ne 0 ]; then
    echo "CMake配置失败，脚本终止。"
    exit 1
fi

# 步骤4: 编译代码
echo "正在编译代码..."
make
if [ $? -ne 0 ]; then
    echo "编译失败，脚本终止。"
    exit 1
fi

mv test_model ../
cd ..

# 步骤5: 运行可执行文件生成实际输出文件
echo "正在运行可执行文件生成实际输出..."
./test_model
if [ $? -ne 0 ]; then
    echo "运行可执行文件失败，脚本终止。"
    exit 1
fi

# 步骤6: 调用verify_result.py进行golden标杆数据和实际输出数据的比对
echo "正在验证结果..."
python3 script/verify_result.py script/output/output_0.bin script/output/golden0.bin

