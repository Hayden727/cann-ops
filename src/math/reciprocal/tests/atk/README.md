## 用例生成

```
atk case -f atk/tests/torch.reciprocal.yaml -p atk/case_generator/generator/generate_types/generate_reduce.py
```



# aclnn算子验证

## 精度比对

```
atk node --backend npu --devices 0 node --backend cpu task -c /root/yxh/cann-ops/src/math/reciprocal/tests/atk/result/torch.reciprocal/json/all_torch.reciprocal.json --task accuracy -s 0 -e 1
```



## 确定性计算

```
atk node --backend npu --devices 0  node --backend cpu --is_compare False task -c /root/yxh/cann-ops/src/math/reciprocal/tests/atk/result/torch.reciprocal/json/all_torch.reciprocal.json --task accuracy_dc -e 400 -rn 50
```



## 性能比对

### aclnn VS aclnn 性能比对

#### 第一步：

```
atk node --backend npu --devices 0 --name aclnn_base node --backend cpu task -c /root/yxh/cann-ops/src/math/reciprocal/tests/atk/result/torch.reciprocal/json/all_torch.reciprocal.json --task performance_device -s 0 -e 1
```



#### 第二步：

```
atk node --backend npu --devices 0 node --backend cpu --is_compare False task -c /root/yxh/cann-ops/src/math/reciprocal/tests/atk/result/torch.reciprocal/json/all_torch.reciprocal.json --task performance_device --bm_file /root/yxh/cann-ops/src/math/reciprocal/tests/atk/atk_output/all_torch.reciprocal_2025-03-25-13-40-42-189589/report/all_torch.reciprocal_reports_2025-03-25-05-40-44.xlsx --bm_backend pyaclnn --bm_name aclnn_base -s 0 -e 1
```



## 更新说明

| 时间 | 更新事项 |
|----|------|
| 2025/03/29 | 新增本readme |

