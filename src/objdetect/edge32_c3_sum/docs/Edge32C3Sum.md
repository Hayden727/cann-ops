声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Edge32C3Sum

## 支持的产品型号

Atlas 推理系列产品/Atlas A2训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：对输入图像 `input` 的每个像素，在垂直方向上进行求和，并计算平均值，然后将结果存储到 `out` 中。对于图像的顶部和底部边界像素，不进行处理。
- 计算公式：
  
  $$
  sum_i = \sum_{j=-7}^{8} input_{y+j, x, i}
  $$
  
  $$
  out_{y, x, i} = \frac{sum_i}{16}
  $$
  
  **说明：**
  其中 `(y, x)` 表示图像中像素的坐标，`i` 表示通道索引，`width` 和 `height` 分别是图像的宽度和高度，`height`和`width`是定值，分别为3880和4887。

## 实现原理

Edge32C3Sum由多次Add操作实现求和。

## 算子执行接口

每个算子分为[两段式接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用“aclnnEdge32C3SumGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEdge32C3Sum”接口执行计算。

* `aclnnStatus aclnnEdge32C3SumGetWorkspaceSize(const aclTensor *input, const aclTensor *out, uint64_t workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnEdge32C3Sum(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnEdge32C3SumGetWorkspaceSize

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

### aclnnEdge32C3Sum

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnEdge32C3SumGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- input，out的数据类型只支持UINT8，数据格式只支持ND

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Edge32C3Sum</td></tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">input</td><td align="center">height * width * 3</td><td align="center">uchar</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">height * width * 3</td><td align="center">uchar</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">edge32_c3_sum</td></tr>
</table>

## 调用示例

详见[Edge32C3Sum自定义算子样例说明算子调用章节](../README.md#算子调用)
