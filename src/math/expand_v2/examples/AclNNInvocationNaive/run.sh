#!/bin/bash
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