# aclnnForeachNonFiniteCheckAndUnscale

## 支持的产品型号

- Atlas A2训练系列产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnForeachNonFiniteCheckAndUnscale”接口执行计算。

- `aclnnStatus aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(const aclTensorList *scaledGrads, const aclTensor *foundInf, const aclTensor *inScale, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnForeachNonFiniteCheckAndUnscale(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：
  
  检查输入scaledGrads中是否存在inf或-inf，若有，foundInf更新为1；scaledGrads进行unscale操作，scaledGrads与inScale相乘。

- 计算公式：

  $$
  scaledGrads = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  $$

  $$
  foundInf = 
  \begin{cases}
  1, max(x) = inf \\
  1, min(x)=-inf \\
  0, otherwise 
  \end{cases}
  $$

  $$
  scaledGrads = inScale * scaledGrads = [inScale * {x_0}, inScale * {x_1}, ... inScale * {x_{n-1}}]
  $$

## aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize

- **参数说明**：

  - scaledGrads（aclTensorList*，计算输入/输出）：公式中的`scaledGrads`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。
  - foundInf（aclTensor*，计算输入/输出）：公式中的`foundInf`，Device侧的aclTensor，数据类型支持FLOAT，数据格式支持ND。
  - inScale（aclTensor*，计算输入/输出）：公式中的`inScale`，Device侧的aclTensor，数据类型支持FLOAT，数据格式支持ND。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的x、out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. x和out的数据类型不在支持的范围之内。
                                        2. x和out无法做数据类型推导。
  ```

## aclnnForeachNonFiniteCheckAndUnscale

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码。

## 约束与限制

无。
