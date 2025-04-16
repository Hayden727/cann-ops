声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MulSigmoid

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：实现了两个数据相加，返回相加结果的功能。
- 计算公式：
  
  $$
  out_1 = 1 / (1 + e ^ (-(x1 * t1)))
  $$
  $$
  out_1 = [i * 2 if i < t2 else i for i in out_1]
  $$
  $$
  out = out_1 * x2 * t3
  $$
  
  **说明：**
  无。

## 实现原理

MulSigmoid由Mul + Sigmoid操作组成，计算过程只有2步：

1. out_mul = Sigmoid(x1[offset] * t1)
2. out_mul = where(tmp < t2, tmp, tmp *2)
3. out = Mul(out_mul, x2[offset]) * t3

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMulSigmoidGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMulSigmoid”接口执行计算。

* `aclnnStatus aclnnMulSigmoidGetWorkspaceSize(const aclTensor *x, const aclTensor *y, double t1, double t2, double t3, const aclTensor *out, uint64_t workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnMulSigmoid(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnMulSigmoidGetWorkspaceSize

- **参数说明：**
  
  - x1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x1，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - x2（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入s2，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - t1（Scalar\*，计算输入）：必选属性，标量，公式中的输入t1，数据类型支持FLOAT32。
  - t2（Scalar\*，计算输入）：必选属性，标量，公式中的输入t2，数据类型支持FLOAT32。
  - t3（Scalar\*，计算输入）：必选属性，标量，公式中的输入t3，数据类型支持FLOAT32。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出out，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

### aclnnMulSigmoid

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMulSigmoidGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- x，y，out的数据类型只支持FLOAT16，数据格式只支持ND

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">MulSigmoid</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">25 * 32768</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td rowspan="3" align="center">属性输入</td><td align="center">t1</td><td align="center">1</td><td align="center">float</td></tr>
<tr><td align="center">t2</td><td align="center">1</td><td align="center">float</td></tr>
<tr><td align="center">t3</td><td align="center">1</td><td align="center">float</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">mul_sigmoid</td></tr>
</table>

## 调用示例

详见[MulSigmoid自定义算子样例说明算子调用章节](../README.md#算子调用)
