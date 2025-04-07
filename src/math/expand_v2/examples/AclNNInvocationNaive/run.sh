#!/bin/bash
rm -rf $HOME/ascend/log/*
rm ./input/*.bin
rm ./output/*.bin

python3 gen_data.py

if [ $? -ne 0 ]; then
    echo "ERROR: generate input data failed!"
    return 1
fi
echo "INFO: generate input data success!"

set -e

rm -rf build
mkdir -p build
cmake -B build
cmake --build build -j
(
    cd build
    ./expand_test
)
ret=`python3 verify_result.py output/output_y.bin output/golden.bin`
echo $ret
if [ "x$ret" == "xtest pass" ]; then
    echo ""
    echo "#####################################"
    echo "INFO: you have passed the Precision!"
    echo "#####################################"
    echo ""
fi