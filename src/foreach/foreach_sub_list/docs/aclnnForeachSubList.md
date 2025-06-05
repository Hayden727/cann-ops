# aclnnForeachSubList

## 支持的产品型号

- Atlas A2训练系列产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnForeachSubListGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnForeachSubList”接口执行计算。

- `aclnnStatus aclnnForeachSubListGetWorkspaceSize(const aclTensorList *x1, const aclTensorList *x2, const aclTensor *alpha, const aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnForeachSubList(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入的两个张量列表的相减运算的结果。

- 计算公式：
 
  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}]\\
  x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$
  
  $$
  y_i = x1_i-{x2_i}*alpha (i=0,1,...n-1)
  $$

## aclnnForeachSubListGetWorkspaceSize

- **参数说明**：

  - x1（aclTensorList*，计算输入）：公式中的`x1`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。数据格式支持ND，shape维度不高于8维。支持非连续的Tensor。该参数中所有tensor的数据类型保存一致。
  - x2（aclTensorList*，计算输入）：公式中的`x2`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32，数据格式支持ND，shape维度不高于8维。支持非连续的Tensor。数据类型、数据格式和shape跟入参`x1`的数据类型、数据格式和shape一致。该参数中所有tensor的数据类型保存一致。
  - alpha（aclTensor*，计算输入）：公式中的`alpha`，Host侧的aclTensor，数据格式支持ND，支持非连续的Tensor。元素个数为1。数据类型支持FLOAT、FLOAT16、INT32，且与入参`x1`的数据类型具有一定对应关系：
    - 当`x1`的数据类型为FLOAT、FLOAT16、INT32时，数据类型与`x1`的数据类型保持一致。
    - 当`x1`的数据类型为BFLOAT16时，数据类型支持FLOAT。
  - out（aclTensorList*，计算输出）：公式中的`y`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32，数据格式支持ND，shape维度不高于8维。支持非连续的Tensor。数据类型、数据格式和shape跟入参`x1`的数据类型、数据格式和shape一致。该参数中所有tensor的数据类型保存一致。
  - workspaceSize（uint64_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的x1、x2、alpha、out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. x1、x2、alpha和out的数据类型不在支持的范围之内。
                                         2. x1、x2、alpha和out无法做数据类型推导。
  返回561002（ACLNN_ERR_INNER_TILING_ERROR）:1. x1与out的数据类型或者shape不一致
  										   2. x1与x2的数据类型或者shape不一致
  										   3. x1、x2或out中的tensor的数据类型不一致。
  										   4. x1、x2、alpha或out中的tensor维度超过8维。
  										   5. alpha元素个数不为1。
  ```

## aclnnForeachSubList

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnForeachSubListGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码。

## 约束说明

无。