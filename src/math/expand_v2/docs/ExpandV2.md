声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# ExpandV2

## 支持的产品型号

Atlas A2 训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：将输入张量x广播成指定shape的张量。该算子x参数仅支持int64输入。
- 例如：

  $$
  x = 
  \left[
  \begin{matrix}
  3
  \end{matrix}
  \right] \\
  shape = 
  \left[
  \begin{matrix}
  2,2
  \end{matrix}
  \right] \\
  y = 
  \left[
  \begin{matrix}
  3,3\\
  3,3
  \end{matrix}
  \right]
  $$

## 实现原理

调用`Ascend C`的`API`接口`ExpandV2`进行实现。

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnExpandV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnExpandV2”接口执行计算。

* `aclnnStatus aclnnExpandV2GetWorkspaceSize(const aclTensor* x, const aclIntArray* shape, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);`
* `aclnnStatus aclnnExpandV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnExpandV2GetWorkspaceSize

- **参数说明：**

  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持INT64，数据格式支持ND。
  - shape（aclIntArray\*，属性）：必选参数，公式中的输入shape，表示要广播后的维度。
  - y（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持INT64，数据格式支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、y的数据类型和数据格式不在支持的范围内。
  ```

### aclnnExpandV2

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnExpandV2GetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。


## 约束与限制

- x，y的数据类型仅支持INT64，数据格式只支持ND。

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">ExpandV2</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
 
<tr><td align="center">x</td><td align="center">tensor</td><td align="center">int64</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">属性</td>
<td align="center">shape</td><td align="center">IntArray</td><td align="center">int64</td><td align="center"></td></tr> 

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">int64</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">expand_v2</td></tr>  
</table>

## 调用示例

详见[ExpandV2自定义算子样例说明算子调用章节](../README.md#算子调用)