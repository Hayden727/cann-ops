声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Threshol8u

## 支持的产品型号

Atlas 推理系列产品/Atlas A2训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：对输入图像 `input` 的每个像素进行阈值处理，将非零像素值设置为 1，零像素值保持为 0，并将结果存储到 `out` 中。
- 计算公式：
  
  $$
  out_{index} =
  \begin{cases}
  1, & \text{if } input_{index} \neq 0 \\
  0, & \text{otherwise}
  \end{cases}
  $$
  
  **说明：**
  其中 `index` 是图像中像素的索引，`height` 和 `width` 分别是图像的高度和宽度。


## 实现原理

Threshol8u由Mins操作实现，计算过程只有1步：

1. out = Mins(input, 1)

## 算子执行接口

每个算子分为[两段式接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用“aclnnThreshol8uGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnThreshol8u”接口执行计算。

* `aclnnStatus aclnnThreshol8uGetWorkspaceSize(const aclTensor *input, const aclTensor *out, uint64_t workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnThreshol8u(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnThreshol8uGetWorkspaceSize

- **参数说明：**
  
  - input（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入input，数据类型支持UINT8，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出out，数据类型支持UINT8，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、y、out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnThreshol8u

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnThreshol8uGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- input，out的数据类型只支持UINT8，数据格式只支持ND

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Threshol8u</td></tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">input</td><td align="center">height * width</td><td align="center">uint8</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">height * width</td><td align="center">uint8</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">threshol8u</td></tr>
</table>

## 调用示例

详见[Threshol8u自定义算子样例说明算子调用章节](../README.md#算子调用)
