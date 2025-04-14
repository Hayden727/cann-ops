# aclnnAscendQuant

## 支持的产品型号

- 昇腾310P AI处理器。
- 昇腾910B AI处理器。

## 接口原型
每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnAscendQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAscendQuant”接口执行计算。

- `aclnnStatus aclnnAscendQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset,
                                                bool sqrtMode, const char *roundMode, int32_t dstType, const aclTensor *y,
                                                uint64_t *workspaceSize, aclOpExecutor **executor)`

- `aclnnStatus aclnnAscendQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：对输入x进行量化操作，scale和offset的size需要是x的最后一维或1。
- 计算公式：
  $$
  y = round((x * scale) + offset)
  $$
  sqrt\_mode为true时，计算公式为:
  $$
  y = round((x * scale * scale) + offset)
  $$

## aclnnAscendQuantGetWorkspaceSize

- **参数说明：**

  - x（aclTensor*，计算输入）：Device侧的aclTensor，需要做量化的输入。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。如果dstType为3，Shape的最后一维需要能被8整除；如果dstType为29，Shape的最后一维需要能被2整除。
    - 昇腾910B AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    - 昇腾310P AI处理器：数据类型支持FLOAT32、FLOAT16。
  - scale（aclTensor*，计算输入）：Device侧的aclTensor，量化中的scale值。scale支持1维张量或多维张量（当shape为1维张量时，scale的第0维需要等于x的最后一维或等于1；当shape为多维张量时，scale的维度需要和x保持一致，最后一个维度的值需要和x保持一致，其他维度为1）。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。如果x的dtype不是FLOAT32，需要和x的dtype一致。
    - 昇腾910B AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    - 昇腾310P AI处理器：数据类型支持FLOAT32、FLOAT16。
  - offset（aclTensor*，计算输入）：可选参数，Device侧的aclTensor，反向量化中的offset值。offset支持1维张量或多维张量（当shape为1维张量时，scale的第0维需要等于x的最后一维或等于1；当shape为多维张量时，scale的维度需要和x保持一致，最后一个维度的值需要和x保持一致，其他维度为1）。数据类型和shape需要与scale保持一致。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - 昇腾910B AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    - 昇腾310P AI处理器：数据类型支持FLOAT32、FLOAT16。
  - sqrtMode（bool，计算输入）：host侧的bool，指定scale参与计算的逻辑，当取值为true时，公式为y = round((x * scale * scale) + offset)。数据类型支持BOOL。
  - roundMode（char\*，计算输入）：host侧的string，指定cast到int8输出的转换方式，支持取值round/ceil/trunc/floor。
  - dstType（int32_t，计算输入）：host侧的int32_t，指定输出的数据类型，该属性数据类型支持INT。
    - 昇腾910B AI处理器：支持取值2、3、29，分别表示INT8、INT32、INT4。
    - 昇腾310P AI处理器：支持取值2，表示INT8。
  - y（aclTensor\*，计算输出）：Device侧的aclTensor。类型为INT32时，Shape的最后一维是x最后一维的1/8，其余维度和x一致; 其他类型时，Shape与x一致。支持空Tensor，支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持ND。
    - 昇腾910B AI处理器：数据类型支持INT8、INT32、INT4。
    - 昇腾310P AI处理器：数据类型支持INT8。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR):  1. 传入的x、scale、y是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID):  1. x、scale、offset、y的数据类型或数据格式不在支持的范围之内。
                                     2. x、scale、offset、y的shape不满足限制条件。
                                     3. roundMode不在有效取值范围。
                                     4. dstType不在有效取值范围。
                                     5. y的数据类型为int4时，x的shape尾轴大小不是偶数。
                                     6. y的数据类型为int32时，y的shape尾轴不是x的shape尾轴大小的8倍，或者x与y的shape的非尾轴的大小不一致。
  ```

## aclnnAscendQuant

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAscendQuantGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制

无。

## 调用示例

详见[AscendQuantV2自定义算子样例说明算子调用章节](../README.md#算子调用)
