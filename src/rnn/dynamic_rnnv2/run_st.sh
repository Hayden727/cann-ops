export CANN_ROOT=/home/cx/canndev
export install_path=/home/cx/Ascend/ascend-toolkit/latest

export PATH=$install_path/compiler/ccec_compiler/bin:$install_path/compiler/bin:$PATH
export PYTHONPATH=$install_path/compiler/python/site-packages:$CANN_ROOT/tools/op_test_frame/python:$PYTHONPATH
export LD_LIBRARY_PATH=$install_path/compiler/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=$install_path/opp
# acl
export DDK_PATH=$install_path
export NPU_HOST_LIB=$install_path/runtime/lib64/stub
export LD_LIBRARY_PATH=$install_path/runtime/lib64:$install_path/add-ons:$LD_LIBRARY_PATH

export msopst="$CANN_ROOT/tools/op_test_frame/python/op_test_frame/scripts/msopst"

export supported_soc=Ascend910

python "$msopst" run -i "$CANN_ROOT/ops/built-in/tests/st/YoloxBoundingBoxDecode/YoloxBoundingBoxDecode_case.json" -soc "$supported_soc" -out output

# python "$msopst" run -i "$CANN_ROOT/ops/built-in/tests/st/Add/Add_case.json" -soc "$supported_soc" -out ouput_path -conf "$CANN_ROOT/tools/op_test_frame/python/op_test_frame/scripts/msopst.ini" 