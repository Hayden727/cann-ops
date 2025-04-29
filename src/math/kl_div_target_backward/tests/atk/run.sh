# atk case -f op_kl_div_target_backward.yaml -p generate_reduce.py
# atk node --backend pyaclnn --devices 0 node --backend cpu task -c result/op_kl_div_target_backward/json/all_op_kl_div_target_backward.json -p aclnn_kl_div_target_backward.py --task accuracy -s 0 -e 200 -mt 100
# atk node --backend pyaclnn --devices 0 node --backend npu --devices 1 task -c result/op_kl_div_target_backward/json/all_op_kl_div_target_backward.json -p aclnn_kl_div_target_backward.py --task performance_device -s 0 -e 200 -mt 100
# atk node --backend pyaclnn --devices 0 node --backend cpu --is_compare False task -c result/op_kl_div_target_backward/json/all_op_kl_div_target_backward.json -p aclnn_kl_div_target_backward.py --task accuracy_dc -e 703 -rn 50

# broadcast
# atk case -f op_kl_div_target_backward_broadcast.yaml -p generate_broadcast.py
atk node --backend pyaclnn --devices 0 node --backend cpu task -c result/op_kl_div_target_backward_broadcast/json/all_op_kl_div_target_backward_broadcast.json -p aclnn_kl_div_target_backward.py --task accuracy -s 0 -e 200 -mt 100
# atk node --backend pyaclnn --devices 0 node --backend npu --devices 1 task -c result/op_kl_div_target_backward_broadcast/json/all_op_kl_div_target_backward_broadcast.json -p aclnn_kl_div_target_backward.py --task performance_device -s 0 -e 200 -mt 100
# atk node --backend pyaclnn --devices 0 node --backend cpu --is_compare False task -c result/op_kl_div_target_backward_broadcast/json/all_op_kl_div_target_backward_broadcast.json -p aclnn_kl_div_target_backward.py --task accuracy_dc -e 703 -rn 50
